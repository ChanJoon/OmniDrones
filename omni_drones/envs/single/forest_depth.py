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

    - `vel`: Reward computed from the velocity along the target direction.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `safety`: Logarithmic reward based on depth image to encourage safe flying.
    - `depth_penalty`: Exponential penalty based on minimum depth to nearby obstacles.
    - `collision`: Large negative penalty when physical collision is detected (-10.0 by default).

    The total reward is computed as follows:

    ```{math}
        r = r_\text{vel} + r_\text{up} + 1.0 + 0.2 \cdot r_\text{safety} + r_\text{depth_penalty} + r_\text{collision}
    ```

    ## Episode End

    The episode ends when:
    - The drone collides with obstacles or terrain (detected via contact forces)
    - The drone flies too low (< 0.2m) or too high (> 4m)
    - The drone's velocity exceeds 2.5 m/s
    - NaN values are detected in the state

    ## Config

    | Parameter               | Type  | Default      | Description                                                 |
    | ----------------------- | ----- | ------------ | ----------------------------------------------------------- |
    | `drone_model`           | str   | "firefly"    | Specifies the model of the drone being used.                |
    | `depth_range`           | float | 10.0         | Specifies the maximum range of the depth camera.            |
    | `depth_resolution`      | tuple | [64, 64]     | Specifies the resolution of the depth image (h, w).         |
    | `time_encoding`         | bool  | True         | Whether to include time encoding in the observation space.  |
    | `reset_on_collision`    | bool  | True         | Whether to reset the environment when collision occurs.     |
    | `collision_penalty`     | float | -10.0        | Reward penalty applied when collision is detected.          |
    | `collision_force_threshold` | float | 0.3      | Contact force threshold (N) for collision detection.        |
    """

    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()

        # Collision detection settings
        self.reset_on_collision = cfg.task.get("reset_on_collision", True)
        self.collision_penalty = cfg.task.get("collision_penalty", -10.0)
        
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

        # Initialize drone with contact force tracking enabled for collision detection
        self.drone.initialize(track_contact_forces=self.reset_on_collision)

        # Apply PhysX contact reporting API for reliable collision detection
        # Track BOTH base_link AND rotors since rotors collide first
        if self.reset_on_collision:
            from pxr import PhysxSchema
            from isaacsim.core.utils.stage import get_current_stage

            stage = get_current_stage()
            base_link_success = 0
            rotor_success = 0

            for i in range(self.num_envs):
                # Enable contact reporting for base_link
                base_link_path = f"/World/envs/env_{i}/{self.drone.name}_0/base_link"
                base_link_prim = stage.GetPrimAtPath(base_link_path)

                if base_link_prim.IsValid():
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(base_link_prim)
                    cr_api.CreateThresholdAttr().Set(0.0)  # Report all contacts
                    base_link_success += 1

                # Enable contact reporting for all rotors (critical for obstacle collision!)
                for rotor_idx in range(self.drone.num_rotors):
                    rotor_path = f"/World/envs/env_{i}/{self.drone.name}_0/rotor_{rotor_idx}"
                    rotor_prim = stage.GetPrimAtPath(rotor_path)

                    if rotor_prim.IsValid():
                        cr_api = PhysxSchema.PhysxContactReportAPI.Apply(rotor_prim)
                        cr_api.CreateThresholdAttr().Set(0.0)
                        rotor_success += 1

            print("=" * 60)
            print("COLLISION DETECTION SETUP:")
            print(f"  PhysX Contact Reporting enabled:")
            print(f"    - Base links: {base_link_success}/{self.num_envs} envs")
            print(f"    - Rotors: {rotor_success}/{self.num_envs * self.drone.num_rotors} total ({self.drone.num_rotors} per drone)")
            print(f"  Collision force threshold: {cfg.task.get('collision_force_threshold', 0.0001)}N")
            print(f"  Collision penalty: {cfg.task.get('collision_penalty', -10.0)}")
            print(f"  Reset on collision: {self.reset_on_collision}")
            print("=" * 60)

        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        # Collision force threshold (in Newtons) - tune this based on your drone scale
        self.collision_force_threshold = cfg.task.get("collision_force_threshold", 0.0001)

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
            "safety": Unbounded(1),
            "min_depth": Unbounded(1),
            "collision": Unbounded(1),
            "collision_depth": Unbounded(1),  # Inverse depth metric for collision severity
            # Reset reason tracking (mutually exclusive)
            "reset_collision": Unbounded(1),  # Reset due to collision
            "reset_altitude": Unbounded(1),   # Reset due to altitude violation
            "reset_velocity": Unbounded(1),   # Reset due to velocity violation
            "reset_nan": Unbounded(1),        # Reset due to NaN in state
            "reset_truncated": Unbounded(1),  # Reset due to max episode length
            # Performance metrics
            "avg_velocity": Unbounded(1),     # Average velocity magnitude
            "forward_progress": Unbounded(1), # Forward displacement
            "success": Unbounded(1),          # Completed episode without collision/violation
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

        # Track initial position for forward progress calculation
        if not hasattr(self, 'init_pos'):
            self.init_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.init_pos[env_ids] = pos

        # Initialize velocity accumulator for average velocity metric
        if not hasattr(self, 'vel_accumulator'):
            self.vel_accumulator = torch.zeros(self.num_envs, 1, device=self.device)
        self.vel_accumulator[env_ids] = 0.
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

        # Calculate min depth for debug logging (used below)
        min_depth_per_env = self.depth_image.amin(dim=(1, 2, 3))  # (num_envs,)

        # Check for collision using contact forces only
        collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        force_magnitude = torch.zeros(self.num_envs, 1, device=self.device)

        if self.reset_on_collision:
            # Get contact forces from BOTH base_link AND rotors
            # Base link forces: (num_envs, 1, 3) for single-agent tasks
            base_forces = self.drone.base_link.get_net_contact_forces()

            if base_forces is None:
                # Fallback if contact tracking not enabled
                print("[WARNING] Base link contact forces not available - contact tracking may not be enabled")
                base_force_mag = torch.zeros(self.num_envs, 1, device=self.device)
            else:
                # Squeeze agent dimension for single-agent: (num_envs, 1, 3) -> (num_envs, 3)
                base_forces = base_forces.squeeze(1)
                base_force_mag = base_forces.norm(dim=-1, keepdim=True)  # (num_envs, 1)

            # Rotor forces: (num_envs, 1, num_rotors, 3) for single-agent tasks
            rotor_forces = self.drone.rotors_view.get_net_contact_forces()

            if rotor_forces is None:
                # Fallback if rotor contact tracking not enabled
                print("[WARNING] Rotor contact forces not available - using base_link only")
                max_rotor_force = torch.zeros(self.num_envs, 1, device=self.device)
            else:
                # Squeeze agent dimension for single-agent: (num_envs, 1, num_rotors, 3) -> (num_envs, num_rotors, 3)
                rotor_forces = rotor_forces.squeeze(1)
                rotor_force_mag = rotor_forces.norm(dim=-1)  # (num_envs, num_rotors)
                max_rotor_force = rotor_force_mag.max(dim=-1, keepdim=True)[0]  # (num_envs, 1)

            # Total collision force = max of base_link or any rotor
            force_magnitude = torch.maximum(base_force_mag, max_rotor_force)
            collision = force_magnitude > self.collision_force_threshold

            # Apply collision penalty to reward
            reward = reward + self.collision_penalty * collision.float()

        # Termination conditions
        # Track individual termination reasons for metrics
        altitude_violation = (self.drone.pos[..., 2] < 0.2) | (self.drone.pos[..., 2] > 4.)
        velocity_violation = (self.drone.vel_w[..., :3].norm(dim=-1) > 2.5)
        misbehave = altitude_violation | velocity_violation
        hasnan = torch.isnan(self.drone_state).any(-1)

        # Terminate on collision or misbehavior
        terminated = misbehave | hasnan | collision
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # Track performance metrics
        vel_magnitude = self.drone.vel_w[..., :3].norm(dim=-1)  # (num_envs, 1)
        self.vel_accumulator += vel_magnitude
        avg_velocity = self.vel_accumulator / (self.progress_buf.unsqueeze(1) + 1)

        # Forward progress (Y-axis displacement from initial position)
        forward_progress = (self.drone.pos[..., 1] - self.init_pos[..., 1]).abs()

        # Success: completed episode without collision or misbehavior
        success = truncated & ~collision & ~misbehave & ~hasnan

        # Track reset reasons (mutually exclusive)
        # Priority: collision > altitude > velocity > nan > truncated
        reset_collision = collision & terminated
        reset_altitude = ~collision & altitude_violation & terminated
        reset_velocity = ~collision & ~altitude_violation & velocity_violation & terminated
        reset_nan = ~collision & ~misbehave & hasnan & terminated
        reset_truncated = truncated
        

        # Update stats
        self.stats["safety"].add_(reward_safety)
        self.stats["min_depth"][:] = min_depth_per_env.unsqueeze(-1)
        # Accumulate collision during episode (once True, stays True)
        self.stats["collision"] = torch.maximum(
            self.stats["collision"], collision.float()
        )
        self.stats["collision_depth"][:] = (1.0 / (min_depth_per_env.unsqueeze(-1) + 0.01))  # Inverse depth

        # Reset reason tracking
        # Accumulate reset reasons during episode (use max to keep True once set)
        self.stats['reset_collision'] = torch.maximum(
            self.stats['reset_collision'], reset_collision.float()
        )
        self.stats['reset_altitude'] = torch.maximum(
            self.stats['reset_altitude'], reset_altitude.float()
        )
        self.stats['reset_velocity'] = torch.maximum(
            self.stats['reset_velocity'], reset_velocity.float()
        )
        self.stats['reset_nan'] = torch.maximum(
            self.stats['reset_nan'], reset_nan.float()
        )
        self.stats["reset_truncated"] = torch.maximum(
            self.stats["reset_truncated"], reset_truncated.float()
        )

        # Performance metrics
        self.stats["avg_velocity"][:] = avg_velocity
        self.stats["forward_progress"][:] = forward_progress
        self.stats["success"][:] = success.float()

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
                "stats": self.stats.clone(),  # Include stats BEFORE reset
            },
            self.batch_size,
        )


