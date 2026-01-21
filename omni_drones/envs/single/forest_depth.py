import math
import torch
import torch.distributions as D
import einops

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate, quat_mul

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec

from isaacsim.core.utils.viewports import set_camera_view


def exponential_reward_function(alpha: float, offset: float, x: torch.Tensor) -> torch.Tensor:
    """Exponential reward function: alpha * exp(-offset * x)"""
    return alpha * torch.exp(-offset * x)


class ForestDepth(IsaacEnv):
    r"""
    This is a single-agent task where the agent is required to navigate a randomly
    generated cluttered environment. The agent needs to fly at a commanded speed
    along the positive direction while avoiding collisions with obstacles.

    The agent utilizes a depth camera to perceive its surroundings.

    ## Observation

    The observation is given by a `Composite` containing the following values:

    - `"state"` (16 + `num_rotors`): The basic information of the drone
      (except its position), containing its rotation (in quaternion), velocities
      (linear and angular), heading and up vectors, and the current throttle.
    - `"depth"` (1, h, w) : The depth image from the camera. The size is decided by the
      resolution configuration.

    ## Reward

    - `vel`: Reward computed from the position error to the target position.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `survive`: Reward of a constant value to encourage collision avoidance.
    - `depth_penalty`: Exponential penalty based on minimum depth to nearby obstacles.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{vel} + r_\text{up} + r_\text{survive} + r_\text{depth_penalty}
    ```

    ## Episode End

    The episode ends when the drone misbehaves, e.g., when the drone collides
    with the ground or obstacles, or when the drone flies out of the boundary.

    ## Config

    | Parameter               | Type  | Default      | Description                                                 |
    | ----------------------- | ----- | ------------ | ----------------------------------------------------------- |
    | `drone_model`           | str   | "firefly"    | Specifies the model of the drone being used.                |
    | `depth_range`           | float | 10.0         | Specifies the maximum range of the depth camera.            |
    | `depth_resolution`      | tuple | [64, 64]     | Specifies the resolution of the depth image (h, w).         |
    | `time_encoding`         | bool  | True         | Whether to include time encoding in the observation space.  |
    """

    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        
        # Depth camera configuration (support both new and legacy config)
        if hasattr(cfg.task, 'depth_camera'):
            # New unified config
            from omni_drones.sensors import DepthCameraCfg
            self.depth_cfg = cfg.task.depth_camera
            self.depth_resolution = tuple(self.depth_cfg.resolution)
            self.depth_range = self.depth_cfg.range
        else:
            # Legacy flat config (backward compatibility)
            from omni_drones.sensors import DepthCameraCfg, DepthProcessingCfg
            self.depth_resolution = tuple(cfg.task.depth_resolution)
            self.depth_range = cfg.task.depth_range
            backend = cfg.task.get("depth_backend", "isaaclab")
            
            # Build DepthCameraCfg from legacy config
            self.depth_cfg = DepthCameraCfg(
                backend=backend,
                resolution=self.depth_resolution,
                range=self.depth_range,
                offset_pos=cfg.task.get("simple_raycaster_depth_camera_offset_pos", [0.1, 0.0, 0.0]),
                offset_rot_wxyz=cfg.task.get("simple_raycaster_depth_camera_offset_rot_wxyz", [0.5, -0.5, 0.5, -0.5]),
                convention="ros",
                focal_length=cfg.task.get("simple_raycaster_depth_camera_focal_length", 24.0),
                horizontal_aperture=cfg.task.get("simple_raycaster_depth_camera_horizontal_aperture", 20.955),
                data_type="distance_to_camera",  # Legacy default
                mesh_prim_paths=cfg.task.get("simple_raycaster_mesh_prim_paths", ["/World/ground"]),
                simplify_factor=cfg.task.get("simple_raycaster_simplify_factor", None),
                mesh_poses_from_stage=cfg.task.get("simple_raycaster_mesh_poses_from_stage", True),
                mesh_count=cfg.task.get("simple_raycaster_mesh_count", None),
                processing=DepthProcessingCfg(),
            )

        super().__init__(cfg, headless)

        # Initialize depth camera sensor
        self.depth_sensor.initialize()

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = 24.
            self.target_pos[:, 0, 2] = 2.

        self.alpha = 0.8
        
        # For depth-based penalty
        self.min_pixel_dist = torch.zeros(self.num_envs, device=self.device)

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        from isaaclab.terrains import (
            TerrainImporterCfg,
            TerrainImporter,
            TerrainGeneratorCfg,
            HfDiscreteObstaclesTerrainCfg,
        )

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0., 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=5,
                num_cols=5,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(8.0, 8.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=40,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 0.8),
                        obstacle_height_range=(3.0, 4.0),
                        platform_width=1.5,
                    )
                },
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        # Create unified depth camera sensor
        from omni_drones.sensors import DepthCamera
        
        drone_prim_path = f"/World/envs/env_.*/{self.drone.name}_0"
        self.depth_sensor = DepthCamera(
            cfg=self.depth_cfg,
            num_envs=self.num_envs,
            device=self.device,
            drone_prim_path=drone_prim_path
        )
        
        return ["/World/ground"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        depth_h, depth_w = self.depth_resolution

        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Composite({
                    "state": Unbounded((1, drone_state_dim), device=self.device),
                    "depth": Unbounded((1, depth_h, depth_w), device=self.device)
                }),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": Composite({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "action_smoothness": Unbounded(1),
            "safety": Unbounded(1),
            "min_depth": Unbounded(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
        pos[:, 0, 1] = -24.
        pos[:, 0, 2] = 2.

        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        # Update the depth camera after physics step
        self.depth_sensor.update(self.dt)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]

        # Get depth image from unified sensor
        if self.depth_cfg.backend == "isaaclab":
            self.depth_image = self.depth_sensor.get_depth()
        else:  # simple_raycaster
            pos_w = self.drone.pos.squeeze(1)
            rot_w = self.drone.rot.squeeze(1)
            self.depth_image = self.depth_sensor.get_depth(pos_w, rot_w)
        
        # Compute depth range pixels for reward computation (inverted: closer = higher value)
        self.depth_range_pixels = (self.depth_range - self.depth_image) / self.depth_range

        distance = self.rpos.norm(dim=-1, keepdim=True)
        rpos_clipped = self.rpos / distance.clamp(1e-6)
        state = torch.cat([rpos_clipped, self.drone_state[..., 3:]], dim=-1)  # (num_envs, 1, state_dim)

        if self._should_render(0) and self.enable_viewport:
            self.debug_draw.clear()
            x = self.drone.pos[0, 0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

        return TensorDict(
            {
                "agents": {
                    "observation": {
                        "state": state,
                        "depth": self.depth_image
                    },
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pose reward
        distance = self.rpos.norm(dim=-1, keepdim=True)
        vel_direction = self.rpos / distance.clamp_min(1e-6)

        # Compute minimum depth distance for safety penalty
        # depth_image: (num_envs, 1, h, w), min over spatial dimensions
        depth_obs = 10.0 * self.depth_range_pixels.squeeze(1)  # (num_envs, h, w)
        depth_obs[depth_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(depth_obs, dim=(1, 2))  # (num_envs,)
        
        # Safety reward based on minimum depth
        reward_safety = torch.log(self.depth_image.clamp_min(0.1)).mean(dim=(1, 2, 3)).unsqueeze(-1)  # (num_envs, 1)
        
        # Velocity reward
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1).clip(max=2.0)

        # Uprightness reward
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # Depth-based penalty (exponential penalty for close obstacles)
        reward_depth_penalty = -exponential_reward_function(
            4.0, 1.0, self.min_pixel_dist
        ).unsqueeze(-1)  # (num_envs, 1)

        reward = reward_vel + reward_up + 1. + reward_safety * 0.2 + reward_depth_penalty

        # Termination conditions
        min_depth_per_env = self.depth_image.amin(dim=(1, 2, 3))  # (num_envs,)
        
        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 4.)
            | (self.drone.vel_w[..., :3].norm(dim=-1) > 2.5)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["safety"].add_(reward_safety)
        self.stats["min_depth"][:] = min_depth_per_env.unsqueeze(-1)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )


