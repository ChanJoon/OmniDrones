import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from torchrl.data import CompositeSpec, TensorSpec

from .ppo_depth_common import PPODepthBase, PPODepthConfig


class DepthCNNSmall(nn.Module):
    """3-layer CNN encoder for depth images.

    Architecture: Conv(16) -> Conv(32) -> Conv(32) -> Linear(feature_dim)
    Input:  (batch, n_agents, H, W)
    Output: (batch, n_agents, feature_dim)
    """

    def __init__(self, n_agents: int, feature_dim: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.cnn = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            Rearrange("n c h w -> n (c h w)"),
            nn.LazyLinear(feature_dim), nn.LayerNorm(feature_dim),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        batch_size = depth.shape[0]
        depth_flat = einops.rearrange(depth, "b n h w -> (b n) 1 h w")
        features = self.cnn(depth_flat)
        return einops.rearrange(features, "(b n) f -> b n f", b=batch_size, n=self.n_agents)


class PPODepthCNNSmall(PPODepthBase):
    """PPO depth policy with small 3-layer CNN encoder."""

    def __init__(
        self,
        cfg: PPODepthConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device)
        depth_cnn = DepthCNNSmall(self.n_agents, cfg.encoder_feature_dim)
        self._build_encoder_pipeline(depth_cnn)
