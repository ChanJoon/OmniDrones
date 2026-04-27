#!/usr/bin/env python3
"""Train a depth-based navigation policy using PPO.

Uses Isaac Lab's AppLauncher for safe simulator startup and Hydra for configuration.

Usage:
    python scripts/train_depth.py --task <task_name> [options]
    python scripts/train_depth.py task=<task_name> [hydra_overrides]

Arguments:
    --task               Task name (default: ForestDepth)
    --num_envs           Number of parallel environments
    --seed               Random seed
    --max_iters          Maximum training iterations
    --eval_interval      Evaluation interval (iterations)
    --save_interval      Checkpoint save interval (iterations)
    --video              Enable video recording during evaluation

Examples:
    # Using CLI arguments
    python scripts/train_depth.py --task ForestDepth --num_envs 128 --seed 42

    # Using Hydra overrides
    python scripts/train_depth.py task=ForestDepth env.num_envs=128 wandb.mode=disabled

    # With video recording
    python scripts/train_depth.py --task ForestDepth --video --headless

Logs saved to: outputs/<task_name>/<timestamp>_<run_name>/
"""

from __future__ import annotations

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. Parse Arguments & Launch Simulation App
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train depth-based navigation policy with PPO",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Supports both CLI arguments and Hydra overrides (key=value syntax)"
)

# Task configuration
parser.add_argument("--task", type=str, default=None, help="Task name (default: from config)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

# Training parameters
parser.add_argument("--max_iters", type=int, default=None, help="Maximum training iterations")
parser.add_argument("--eval_interval", type=int, default=None, help="Evaluation interval (iterations)")
parser.add_argument("--save_interval", type=int, default=None, help="Checkpoint save interval (iterations)")

# Video and debugging
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation")

# Add Isaac Lab standard arguments (headless, enable_cameras, device, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse known arguments; remaining args go to Hydra
args_cli, hydra_overrides = parser.parse_known_args()

# Pre-process Hydra overrides that affect AppLauncher
# AppLauncher needs headless setting BEFORE simulator starts
for override in hydra_overrides:
    if override.startswith('headless='):
        headless_value = override.split('=')[1].lower()
        if headless_value in ['true', '1', 'yes']:
            args_cli.headless = True
            print(f"[INFO] Headless mode enabled via Hydra override")
        elif headless_value in ['false', '0', 'no']:
            args_cli.headless = False
            print(f"[INFO] Headless mode disabled via Hydra override")

# IMPORTANT: IsaacLab cameras must be enabled for depth-based training
# Check if user explicitly disabled cameras
if hasattr(args_cli, 'enable_cameras') and args_cli.enable_cameras is False:
    # User explicitly set --enable_cameras=False
    print("[WARNING] Cameras are disabled but depth training requires them!")
    print("[WARNING] Training may fail. Use --enable_cameras or remove --enable_cameras=False")
else:
    # Enable cameras by default for depth training
    args_cli.enable_cameras = True
    if not args_cli.video:
        print("[INFO] Cameras enabled for depth-based training (use --enable_cameras=False to override)")

# Ensure cameras are enabled for video recording
if args_cli.video:
    args_cli.enable_cameras = True

# Launch the simulator (must be done before importing torch/isaac extensions)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 2. Imports (Safe to import torch/isaac extensions now)
# -----------------------------------------------------------------------------
import logging
import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import OmegaConf

# Set torch backends for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Isaac Lab & OmniDrones imports
from omni_drones.envs.isaac_env import IsaacEnv
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from omni_drones.learning import ALGOS

# -----------------------------------------------------------------------------
# 4. Main Training Logic
# -----------------------------------------------------------------------------

def main():
    """Main training loop."""
    # Load Hydra configuration with overrides from command line
    OmegaConf.register_new_resolver("eval", eval)

    with hydra.initialize(config_path=".", version_base=None):
        cfg = hydra.compose(config_name="train_depth", overrides=hydra_overrides)

    # Override config from command line arguments
    if args_cli.task is not None:
        cfg.task.name = args_cli.task
    if args_cli.num_envs is not None:
        cfg.env.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed
    if args_cli.max_iters is not None:
        cfg.max_iters = args_cli.max_iters
    if args_cli.eval_interval is not None:
        cfg.eval_interval = args_cli.eval_interval
    if args_cli.save_interval is not None:
        cfg.save_interval = args_cli.save_interval

    # Sync headless mode from AppLauncher
    cfg.headless = args_cli.headless

    # Sync video setting (CLI overrides config, but headless forces video off)
    if args_cli.video and not cfg.headless:
        cfg.video = True
    elif cfg.headless:
        # Force disable video in headless mode to avoid Replicator issues
        cfg.video = False
        if args_cli.video:
            print("[INFO] Video recording disabled in headless mode")
    elif not hasattr(cfg, 'video'):
        cfg.video = False

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Enable Replicator only if video recording is enabled (not in headless mode)
    if cfg.video:
        cfg.sim.enable_replicator = True
        print(f"[INFO] Video recording enabled - Replicator is active")
    
    # Initialize WandB and set process title
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    # Create environment
    import omni_drones.envs  # Ensure envs are registered

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # Apply environment transforms
    transforms = [InitTracker()]

    # Flatten composite observation specs (convert to flat tensors for MLP)
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
        and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    # Optional action space discretization
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # Note: We use the same environment for both training and evaluation
    # Isaac Sim doesn't support multiple environment instances on the same USD stage
    # To reduce memory usage during training, reduce env.num_envs in the config

    # Create policy
    policy_cls = ALGOS[cfg.algo.name]
    policy = policy_cls(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )

    # Training configuration
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    # Setup episode statistics tracking and data collector
    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    # Setup curriculum learning if enabled
    curriculum_manager = None
    if cfg.task.get("curriculum", {}).get("enabled", False):
        from omni_drones.utils.curriculum import CurriculumManager

        curriculum_cfg = cfg.task.curriculum
        curriculum_manager = CurriculumManager(
            num_envs=env.num_envs,
            device=base_env.device,
            window_size=curriculum_cfg.get("window_size", 20),
            success_threshold_up=curriculum_cfg.get("success_threshold_up", 0.75),
            success_threshold_down=curriculum_cfg.get("success_threshold_down", 0.25),
            collision_threshold=curriculum_cfg.get("collision_threshold", 0.60),
            cooldown_episodes=curriculum_cfg.get("cooldown_episodes", 30),
            num_terrain_levels=5,  # Fixed at 5 rows for ForestDepth terrain
        )
        base_env.curriculum_manager = curriculum_manager
        print(f"[INFO] Curriculum learning enabled with {curriculum_cfg.window_size} episode window")
    else:
        print("[INFO] Curriculum learning disabled")

    @torch.no_grad()
    def evaluate(seed: int = 0, exploration_type: ExplorationType = ExplorationType.MODE):
        """Evaluate policy and optionally record video."""

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        # Setup video recording if enabled (from config or CLI)
        render_callback = None
        if cfg.video or getattr(cfg.sim, "enable_replicator", False):
            render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video when available
        if render_callback is not None:
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        return info

    # Main training loop
    pbar = tqdm(collector)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        # Log episode statistics
        if len(episode_stats) >= base_env.num_envs:
            episode_data = episode_stats.pop()
            raw_stats = {
                (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_data.items(True, True)
            }
            reward_key_map = {
                "reward_distance": "distance",
                "reward_time": "time",
                "reward_heading": "heading",
                "reward_vel": "vel",
                "reward_penalty_action": "penalty_action",
                "reward_penalty_collision": "penalty_collision",
                "reward_penalty_depth": "penalty_depth",
                "reward_hover": "hover",
                "safety": "safety",
                "min_depth": "min_depth",
                "collision_depth": "collision_depth",
            }
            reward_keys = set(reward_key_map.keys())
            train_stats = {}
            reward_stats = {}
            for k, v in raw_stats.items():
                short = k.replace("stats.", "")
                if short in reward_keys:
                    reward_stats[f"reward/{reward_key_map[short]}"] = v
                else:
                    train_stats[f"train/{short}"] = v
            info.update(train_stats)
            info.update(reward_stats)

            # Update curriculum based on episode performance
            if curriculum_manager is not None:
                # Pass episode statistics to curriculum manager
                base_env.update_curriculum(episode_data)

                # Log curriculum metrics
                curriculum_stats = curriculum_manager.get_statistics()
                info["curriculum/mean_terrain_level"] = curriculum_stats["mean_terrain_level"]

                # Log level distribution as separate metrics
                for level_idx, count in enumerate(curriculum_stats["level_distribution"]):
                    info[f"curriculum/level_{level_idx}_count"] = count

        # Perform policy update
        info.update(policy.train_op(data.to_tensordict()))

        # Periodic evaluation
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Evaluating at {collector._frames} frames")
            info.update(evaluate())
            env.train()
            base_env.train()

        # Save checkpoint
        if save_interval > 0 and i % save_interval == 0:
            ckpt_data = {
                "policy": policy.state_dict(),
                "frames": collector._frames,
                "iteration": i,
            }
            if curriculum_manager is not None:
                ckpt_data["curriculum"] = curriculum_manager.state_dict()

            ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
            torch.save(ckpt_data, ckpt_path)
            logging.info(f"Saved checkpoint to {str(ckpt_path)}")

        run.log(info)

        # Print formatted training metrics
        print_dict = {k: v for k, v in info.items() if isinstance(v, float)}
        if print_dict:
            print(f"\n{'='*80}")
            print(f"Iteration {i} | Frames: {collector._frames} | FPS: {collector._fps:.1f}")
            print(f"{'-'*80}")

            # Group metrics by category
            train_metrics = {k.replace('train/', ''): v for k, v in print_dict.items() if k.startswith('train/')}
            other_metrics = {k: v for k, v in print_dict.items() if not k.startswith('train/') and not k.startswith('eval/')}

            if train_metrics:
                print("Training Metrics:")
                for k, v in sorted(train_metrics.items()):
                    print(f"  {k:30s}: {v:8.4f}")

            if other_metrics:
                print("\nOptimization Metrics:")
                for k, v in sorted(other_metrics.items()):
                    print(f"  {k:30s}: {v:8.4f}")
            print(f"{'='*80}\n")

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    # Final evaluation
    logging.info(f"Final evaluation at {collector._frames} frames")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    # Save final checkpoint and create artifact
    ckpt_data = {
        "policy": policy.state_dict(),
        "frames": collector._frames,
        "iteration": i,
    }
    if curriculum_manager is not None:
        ckpt_data["curriculum"] = curriculum_manager.state_dict()

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(ckpt_data, ckpt_path)

    model_artifact = wandb.Artifact(
        f"{cfg.task.name}-{cfg.algo.name.lower()}",
        type="model",
        description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
        metadata=dict(cfg))

    model_artifact.add_file(ckpt_path)
    wandb.save(ckpt_path)
    run.log_artifact(model_artifact)

    logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    wandb.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()
