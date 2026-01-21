# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import torch.distributions as D
import einops

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec

from isaacsim.core.utils.viewports import set_camera_view


class Forest(IsaacEnv):
    r"""
    This is a single-agent task where the agent is required to navigate a randomly
    generated cluttered environment. The agent needs to fly at a commanded speed
    along the positive direction while avoiding collisions with obstacles.

    The agent utilizes Lidar sensors to perceive its surroundings. The Lidar has
    a horizontal field of view (FOV) of 360 degrees and a the vertical view can be
    specified.

    ## Observation

    The observation is given by a `Composite` containing the following values:

    - `"state"` (16 + `num_rotors`): The basic information of the drone
      (except its position), containing its rotation (in quaternion), velocities
      (linear and angular), heading and up vectors, and the current throttle.
    - `"lidar"` (1, w, h) : The lidar scan of the drone. The size is decided by the
      field of view and resolution.

    ## Reward

    - `vel`: Reward computed from the position error to the target position.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `survive`: Reward of a constant value to encourage collision avoidance.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions,
      computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{vel} + r_\text{up} + r_\text{survive} + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the drone misbehaves, e.g., when the drone collides
    with the ground or obstacles, or when the drone flies out of the boundary:

    ```{math}
        d_\text{ground} < 0.2 \text{ or } d_\text{ground} > 4.0 \text{ or } v_\text{drone} > 2.5
    ```

    or when the episode reaches the maximum length.

    ## Config

    | Parameter       | Type  | Default   | Description                                                                                                                                                                                                                             |
    | --------------- | ----- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`   | str   | "firefly" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `lidar_range`   | float | 4.0       | Specifies the maximum range of the lidar.                                                                                                                                                                                               |
    | `lidar_vfov`    | float | [-10, 20] | Specifies the vertical field of view of the lidar.                                                                                                                                                                                      |
    | `time_encoding` | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.lidar_backend = cfg.task.get("lidar_backend", "isaaclab")
        self.lidar_resolution = (36, 4)

        super().__init__(cfg, headless)

        if self.lidar_backend == "isaaclab":
            self.lidar._initialize_impl()

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

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
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
            # visual_material=sim_utils.MdlFileCfg(
            #     mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            #     project_uvw=True,
            # ),
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        self.lidar_vfov = (
            max(-89., self.cfg.task.lidar_vfov[0]),
            min(89., self.cfg.task.lidar_vfov[1])
        )
        self.lidar_range = self.cfg.task.lidar_range
        if self.lidar_backend == "isaaclab":
            ray_caster_cfg = RayCasterCfg(
                prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ray_alignment="base",
                pattern_cfg=patterns.BpearlPatternCfg(
                    vertical_ray_angles=torch.linspace(*self.lidar_vfov, 4)
                ),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
            self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        elif self.lidar_backend == "simple_raycaster":
            try:
                from simple_raycaster.raycaster import MultiMeshRaycaster
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "simple_raycaster backend selected but the package is not installed."
                ) from exc
            import isaacsim.core.utils.stage as stage_utils
            from omni_drones.envs.utils.simple_raycaster import (
                create_raycaster_from_stage,
                select_mesh_prims,
                compute_mesh_poses,
                build_mesh_pose_tensors,
                default_mesh_pose_tensors,
            )

            mesh_patterns = self.cfg.task.get("simple_raycaster_mesh_prim_paths", ["/World/ground"])
            simplify_factor = self.cfg.task.get("simple_raycaster_simplify_factor", None)
            stage = stage_utils.get_current_stage()
            self._simple_lidar_raycaster = create_raycaster_from_stage(
                MultiMeshRaycaster,
                stage=stage,
                paths=mesh_patterns,
                device=self.device,
                simplify_factor=simplify_factor,
            )
            self._simple_lidar_dirs_body = self._build_lidar_dirs_body()
            mesh_pos, mesh_quat = None, None
            if self.cfg.task.get("simple_raycaster_mesh_poses_from_stage", True):
                mesh_prims = select_mesh_prims(mesh_patterns)
                mesh_pos, mesh_quat = compute_mesh_poses(mesh_prims)
            self._simple_lidar_mesh_pos_local = mesh_pos
            self._simple_lidar_mesh_quat_local = mesh_quat
            self._simple_lidar_mesh_pos_w, self._simple_lidar_mesh_quat_w = (
                build_mesh_pose_tensors(self.num_envs, mesh_pos, mesh_quat, self.device)
            )
            if self._simple_lidar_mesh_pos_w is None:
                mesh_count = self.cfg.task.get("simple_raycaster_mesh_count", None)
                if mesh_count is not None:
                    self._simple_lidar_mesh_pos_w, self._simple_lidar_mesh_quat_w = (
                        default_mesh_pose_tensors(self.num_envs, mesh_count, self.device)
                    )
            offset = self.cfg.task.get("simple_raycaster_lidar_offset", [0.0, 0.0, 0.0])
            self._simple_lidar_offset = torch.tensor(offset, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown lidar_backend: {self.lidar_backend}")
        return ["/World/ground"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        lidar_h, lidar_w = 4, 36  # 4 vertical angles, 36 horizontal rays

        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Composite({
                    "state": Unbounded((1, drone_state_dim), device=self.device),
                    "lidar": Unbounded((1, lidar_w, lidar_h), device=self.device)
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
                "reward": Unbounded((1,1))
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
            "safety": Unbounded(1)
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
        if self.lidar_backend == "isaaclab":
            self.lidar.update(self.dt)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]

        if self.lidar_backend == "isaaclab":
            self.lidar_scan = self.lidar_range - (
                (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
                .norm(dim=-1)
                .clamp_max(self.lidar_range)
                .reshape(self.num_envs, 1, *self.lidar_resolution)
            )
            lidar_hits_w = self.lidar.data.ray_hits_w
            lidar_pos_w = self.lidar.data.pos_w
        else:
            from omni_drones.envs.utils.simple_raycaster import raycast_fused
            pos_w = self.drone.pos.squeeze(1)
            rot_w = self.drone.rot.squeeze(1)
            ray_dirs_body = self._simple_lidar_dirs_body
            num_rays = ray_dirs_body.shape[0]
            ray_dirs_body = ray_dirs_body.unsqueeze(0).expand(self.num_envs, -1, -1)
            rot_rep = rot_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4)
            ray_dirs_w = quat_rotate(rot_rep, ray_dirs_body.reshape(-1, 3)).reshape(self.num_envs, num_rays, 3)
            offset = self._simple_lidar_offset.expand(self.num_envs, -1)
            ray_origin = pos_w + quat_rotate(rot_w, offset)
            ray_starts_w = ray_origin.unsqueeze(1).expand(-1, num_rays, -1)
            ray_hits_w, ray_dists = raycast_fused(
                self._simple_lidar_raycaster,
                ray_starts_w=ray_starts_w,
                ray_dirs_w=ray_dirs_w,
                min_dist=0.0,
                max_dist=self.lidar_range,
                mesh_pos_w=self._simple_lidar_mesh_pos_w,
                mesh_quat_w=self._simple_lidar_mesh_quat_w,
            )
            ray_dists = torch.nan_to_num(ray_dists, nan=self.lidar_range, posinf=self.lidar_range, neginf=0.0)
            self.lidar_scan = self.lidar_range - ray_dists.clamp_max(self.lidar_range).reshape(
                self.num_envs, 1, *self.lidar_resolution
            )
            lidar_hits_w = ray_hits_w
            lidar_pos_w = ray_origin

        distance = self.rpos.norm(dim=-1, keepdim=True)
        rpos_clipped = self.rpos / distance.clamp(1e-6)
        state = torch.cat([rpos_clipped, self.drone_state[..., 3:]], dim=-1)  # (num_envs, 1, state_dim)

        if self._should_render(0) and self.enable_viewport:
            self.debug_draw.clear()
            x = lidar_pos_w[0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )
            v = (lidar_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])

        return TensorDict(
            {
                "agents": {
                    "observation": {
                        "state": state,
                        "lidar": self.lidar_scan
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

        reward_safety = torch.log(self.lidar_range-self.lidar_scan).mean(dim=(2, 3))
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1).clip(max=2.0)

        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # effort
        # reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        reward = reward_vel + reward_up + 1. + reward_safety * 0.2

        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 4.)
            | (self.drone.vel_w[..., :3].norm(dim=-1) > 2.5)
            | (einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3))
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["safety"].add_(reward_safety)
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

    def _build_lidar_dirs_body(self) -> torch.Tensor:
        lidar_w, lidar_h = self.lidar_resolution
        h_angles = torch.linspace(0.0, 2.0 * math.pi, lidar_w + 1, device=self.device)[:-1]
        v_angles = torch.deg2rad(
            torch.linspace(self.lidar_vfov[0], self.lidar_vfov[1], lidar_h, device=self.device)
        )
        h_grid, v_grid = torch.meshgrid(h_angles, v_angles, indexing="ij")
        x = torch.cos(v_grid) * torch.cos(h_grid)
        y = torch.cos(v_grid) * torch.sin(h_grid)
        z = torch.sin(v_grid)
        dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
