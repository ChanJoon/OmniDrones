import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap
from dataclasses import dataclass, field
from typing import Dict, List, Union

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from ..modules.distributions import IndependentNormal
from ..utils.valuenorm import ValueNorm1
from .common import GAE, make_mlp
from .ppo import PPOConfig, Actor, make_batch


@dataclass
class PPODepthConfig(PPOConfig):
    name: str = "ppo_depth_cnn_small"
    encoder_feature_dim: int = 128
    mlp_units: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 1e-4
    entropy_coef: float = 0.001
    clip_param: float = 0.1
    checkpoint_path: Union[str, None] = None


class PPODepthBase(TensorDictModuleBase):
    """Base class for PPO policies with depth image encoder.

    Subclasses define the CNN encoder and call ``_build_encoder_pipeline``
    to wire it into the shared actor-critic architecture.
    """

    def __init__(
        self,
        cfg: PPODepthConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        _reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.entropy_coef = cfg.entropy_coef
        self.clip_param = cfg.clip_param
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.gae = GAE(0.99, 0.95)

        self._observation_spec = observation_spec
        self._action_spec = action_spec

    def _build_encoder_pipeline(self, depth_cnn: nn.Module):
        """Wire depth CNN into TensorDict encoder pipeline and build actor/critic.

        Args:
            depth_cnn: Encoder module with signature (batch, n_agents, H, W) -> (batch, n_agents, feature_dim)
        """
        mlp = make_mlp(self.cfg.mlp_units)

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
            log_prob_key="sample_log_prob",
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]
        ).to(self.device)

        # Materialize lazy layers
        fake_input = self._observation_spec.zero()
        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.value_norm = ValueNorm1(1).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "_feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict) -> Dict[str, float]:
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = vmap(self.encoder)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]

        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]
        values = tensordict["state_value"]

        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        # Reshape for GAE: single-agent depth tasks use (batch, steps) layout
        rewards = rewards.squeeze(-1).squeeze(-1)
        dones = dones.squeeze(-1)
        values = values.squeeze(-1).squeeze(-1)
        next_values = next_values.squeeze(-1).squeeze(-1)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / adv.std().clip(1e-7)

        ret_flat = ret.reshape(-1, 1)
        self.value_norm.update(ret_flat)
        ret = self.value_norm.normalize(ret_flat).reshape(ret.shape)

        # Expand back to (batch, steps, 1, 1) for tensordict
        adv = adv.unsqueeze(-1).unsqueeze(-1)
        ret = ret.unsqueeze(-1).unsqueeze(-1)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _ in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict) -> TensorDict:
        self.encoder(tensordict)
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss = torch.max(
            self.critic_loss_fn(b_returns, values_clipped),
            self.critic_loss_fn(b_returns, values),
        )

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
            "explained_var": explained_var,
        }, [])
