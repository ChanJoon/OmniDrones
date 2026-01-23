import logging
import os
import time

import hydra
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from omni_drones.utils.torchrl import EpisodeStats
from omni_drones.learning.utils.valuenorm import ValueNorm1
from omni_drones.learning.ppo.common import GAE
from omni_drones.learning.ppo.ppo import make_mlp, Actor

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose, CatTensors
from torchrl.modules import ProbabilisticActor
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential
from tensordict import TensorDict
from omni_drones.learning.modules.distributions import IndependentNormal


FILE_PATH = os.path.dirname(__file__)


class PPODepthPolicy(TensorDictModuleBase):
    """
    PPO Policy with Depth Camera CNN encoder (for evaluation).
    Matches the architecture from train_depth.py.
    """

    def __init__(self, cfg, observation_spec: CompositeSpec, action_spec: CompositeSpec, reward_spec: TensorSpec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

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
                batch_size = depth.shape[0]
                depth_flat = einops.rearrange(depth, "b n h w -> (b n) 1 h w")
                features = self.cnn(depth_flat)
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

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(1).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "_feature", inplace=True)
        return tensordict


@hydra.main(config_path=FILE_PATH, config_name="train_depth", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Default checkpoint path
    default_checkpoint = "wandb/run/files/checkpoint_final.pt"

    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name + "_play")
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
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

    # Keep environment in train mode during policy initialization
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # Create PPODepthPolicy (must match train_depth.py)
    policy = PPODepthPolicy(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

    # Load trained checkpoint
    # Use +checkpoint=path to override, otherwise uses default
    checkpoint_relative = OmegaConf.select(cfg, "checkpoint", default=default_checkpoint)
    checkpoint_path = os.path.join(FILE_PATH, checkpoint_relative)

    if os.path.exists(checkpoint_path):
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=base_env.device)
        policy.load_state_dict(checkpoint)
        print(f"✓ Checkpoint loaded successfully!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print(f"Running with randomly initialized policy!")
        print(f"{'='*60}\n")

    # Switch to evaluation mode after loading checkpoint
    env.eval()
    policy.eval()

    # Set evaluation frames (default: enough for ~3-5 full episodes per environment)
    # With max_episode_length=800 and num_envs=128, use at least 800*128*3 = 307,200 frames
    eval_frames = cfg.get("eval_frames", 320000)
    frames_per_batch = env.num_envs * 32

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    # Use deterministic exploration for evaluation
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        collector = SyncDataCollector(
            env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=eval_frames,
            device=cfg.sim.device,
            return_same_td=True,
        )

        pbar = tqdm(collector, desc="Evaluating")
        episode_count = 0
        all_episode_stats = []

        for i, data in enumerate(pbar):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            episode_stats.add(data.to_tensordict())

            if len(episode_stats) >= base_env.num_envs:
                episode_count += 1
                stats = {
                    (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                    for k, v in episode_stats.pop().items(True, True)
                }
                # Remove "stats." prefix for cleaner display
                stats = {k.replace("stats.", ""): v for k, v in stats.items()}
                all_episode_stats.append(stats)

                # Add eval prefix
                stats_with_prefix = {"eval/" + k: v for k, v in stats.items()}
                info.update(stats_with_prefix)

            pbar.set_postfix({
                "rollout_fps": collector._fps,
                "frames": collector._frames,
                "episodes": episode_count
            })

        # Print summary statistics
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY ({episode_count} episodes)")
        print(f"{'='*80}")

        if all_episode_stats:
            # Group metrics by category
            performance_keys = ['return', 'episode_len', 'avg_velocity', 'forward_progress', 'success']
            safety_keys = ['collision', 'collision_depth', 'min_depth', 'safety']
            reset_keys = ['reset_collision', 'reset_altitude', 'reset_velocity', 'reset_nan', 'reset_truncated']

            def print_stats_group(title, keys):
                print(f"\n{title}:")
                print(f"{'-'*80}")
                for key in keys:
                    if key in all_episode_stats[0]:
                        values = [s[key] for s in all_episode_stats]
                        mean_val = sum(values) / len(values)
                        min_val = min(values)
                        max_val = max(values)
                        print(f"  {key:28s}: mean={mean_val:8.3f}  min={min_val:8.3f}  max={max_val:8.3f}")

            print_stats_group("Performance Metrics", performance_keys)
            print_stats_group("Safety Metrics", safety_keys)
            print_stats_group("Reset Reasons", reset_keys)

            # Print remaining metrics
            other_keys = [k for k in all_episode_stats[0].keys()
                         if k not in performance_keys + safety_keys + reset_keys]
            if other_keys:
                print_stats_group("Other Metrics", other_keys)

        print(f"\n{'='*80}\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
