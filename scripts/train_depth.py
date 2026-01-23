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

import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.func import vmap
from einops.layers.torch import Rearrange
from omni_drones.learning.ppo.ppo import PPOConfig, make_mlp, make_batch, Actor, IndependentNormal, GAE, ValueNorm1
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule, TensorDictModuleBase
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

# -----------------------------------------------------------------------------
# 3. Policy Definition
# -----------------------------------------------------------------------------

class PPODepthPolicy(TensorDictModuleBase):
    """
    PPO Policy with Depth Camera CNN encoder.
    Similar to PPOPolicy in train_lidar.py but adapted for depth image input.
    """

    def __init__(self, cfg: PPOConfig, observation_spec: CompositeSpec, action_spec: CompositeSpec, reward_spec: TensorSpec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        # action_spec is a Composite, need to access the actual tensor spec
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        # Wrapper to handle depth shape: (batch, n_agents, 1, h, w) -> CNN -> (batch, n_agents, feature_dim)
        class DepthCNN(nn.Module):
            def __init__(self, n_agents):
                super().__init__()
                self.n_agents = n_agents
                self.cnn = nn.Sequential(
                    nn.LazyConv2d(out_channels=16, kernel_size=5, stride=2, padding=2), nn.ELU(),
                    nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1), nn.ELU(),
                    nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1), nn.ELU(),
                    Rearrange("n c h w -> n (c h w)"),
                    nn.LazyLinear(128), nn.LayerNorm(128)
                )
            
            def forward(self, depth):
                # depth shape: (batch, n_agents, h, w) - no channel dimension yet
                batch_size = depth.shape[0]
                # Merge batch and n_agents, add channel dim: (batch*n_agents, 1, h, w)
                depth_flat = einops.rearrange(depth, "b n h w -> (b n) 1 h w")
                # Process with CNN: (batch*n_agents, feature_dim)
                features = self.cnn(depth_flat)
                # Restore structure: (batch, n_agents, feature_dim)
                features = einops.rearrange(features, "(b n) f -> b n f", b=batch_size, n=self.n_agents)
                return features
        
        depth_cnn = DepthCNN(self.n_agents)
        mlp = make_mlp([256, 256])

        self.encoder = TensorDictSequential(
            TensorDictModule(depth_cnn, [("agents", "observation", "depth")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(mlp, ["_feature"], ["_feature"]),
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]
        ).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.value_norm = ValueNorm1(1).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "_feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = vmap(self.encoder)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]

        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        # Reshape for GAE: (batch, steps, n_agents, 1) -> (batch, steps)
        # For single agent environment, squeeze agent and reward dimensions
        rewards = rewards.squeeze(-1).squeeze(-1)  # (batch, steps, 1, 1) -> (batch, steps)
        dones = dones.unsqueeze(-1) if dones.ndim == 2 else dones.squeeze(-1)  # Ensure (batch, steps)
        values = values.squeeze(-1).squeeze(-1)  # (batch, steps, 1, 1) -> (batch, steps) 
        next_values = next_values.squeeze(-1).squeeze(-1)  # (batch, steps, 1, 1) -> (batch, steps)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        
        # ValueNorm expects flattened input
        ret_shape = ret.shape
        ret_flat = ret.reshape(-1, 1)
        self.value_norm.update(ret_flat)
        ret_flat = self.value_norm.normalize(ret_flat)
        ret = ret_flat.reshape(ret_shape)

        # Expand back to (batch, steps, 1, 1) for tensordict
        adv = adv.unsqueeze(-1).unsqueeze(-1)
        ret = ret.unsqueeze(-1).unsqueeze(-1)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.encoder_opt.step()
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

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
    policy = PPODepthPolicy(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
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
            stats = {
                (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            # Remove "stats." prefix and add "train/" prefix
            stats = {
                "train/" + k.replace("stats.", ""): v
                for k, v in stats.items()
            }
            info.update(stats)

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
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

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
    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()
