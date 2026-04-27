import math

import numpy as np
import torch
import torch.distributions as D
import einops
import trimesh

import isaacsim.core.api.objects as objects
import omni_drones.utils.kit as kit_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate, quat_mul

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec

from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field.hf_terrains import discrete_obstacles_terrain
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaacsim.core.utils.viewports import set_camera_view


def exponential_reward_function(alpha: float, offset: float, x: torch.Tensor) -> torch.Tensor:
    """Exponential reward function: alpha * exp(-offset * x)"""
    return alpha * torch.exp(-offset * x)


def curriculum_obstacles_terrain(difficulty: float, cfg: "CurriculumObstaclesTerrainCfg"):
    # NOTE: difficulty는 row 정보를 직접 받지 않으므로 5단계로 양자화해서
    # 난이도를 안정화시킴. (row=7로 늘렸을 때 1,1,2,3,4,5,5 패턴에 가깝게)
    # 정확한 row 패턴 보장은 불가. 롤백하려면 아래 level/t 계산 제거하고
    # 기존 difficulty로 바로 선형 보간하면 됨.
    # 5단계로 양자화
    num_levels = 5
    level = int(difficulty * num_levels) + 1  # 1..6
    level = max(1, min(num_levels, level))    # 1..5

    # level -> [0,1] 스칼라로 변환 (원래 difficulty 대신 t 사용)
    t = (level - 1) / (num_levels - 1)

    num_obstacles = int(cfg.num_obstacles_range[0] + t * (cfg.num_obstacles_range[1] - cfg.num_obstacles_range[0]))
    min_height = cfg.min_height_range[0] + t * (cfg.min_height_range[1] - cfg.min_height_range[0])
    max_height = cfg.max_height_range[0] + t * (cfg.max_height_range[1] - cfg.max_height_range[0])

    cfg.num_obstacles = num_obstacles
    cfg.obstacle_height_range = (min_height, max_height)

    return discrete_obstacles_terrain(difficulty=t, cfg=cfg)



@configclass
class CurriculumObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Height-field obstacles terrain config with curriculum-scaled parameters."""

    function = curriculum_obstacles_terrain

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = (0.4, 0.8)
    obstacle_height_range: tuple[float, float] = (2.5, 4.5)
    num_obstacles: int = 8
    num_obstacles_range: tuple[int, int] = (8, 50)
    min_height_range: tuple[float, float] = (2.5, 3.5)
    max_height_range: tuple[float, float] = (3.5, 4.5)
    platform_width: float = 1.5


def _make_box_mesh(size: tuple[float, float, float], center: tuple[float, float, float], yaw: float = 0.0):
    transform = trimesh.transformations.euler_matrix(0.0, 0.0, yaw, axes="rxyz")
    transform[:3, 3] = np.asarray(center)
    return trimesh.creation.box(extents=size, transform=transform)


def _make_rect_frame_mesh(
    outer_size: tuple[float, float, float],
    inner_size: tuple[float, float, float],
    center: tuple[float, float, float],
    yaw: float = 0.0,
) -> list[trimesh.Trimesh]:
    outer_x, outer_z, depth = outer_size
    inner_x, inner_z, _ = inner_size
    side_w = max((outer_x - inner_x) * 0.5, 0.02)
    top_h = max((outer_z - inner_z) * 0.5, 0.02)
    cx, cy, cz = center
    pieces = [
        _make_box_mesh((side_w, depth, outer_z), (cx - inner_x * 0.5 - side_w * 0.5, cy, cz), yaw),
        _make_box_mesh((side_w, depth, outer_z), (cx + inner_x * 0.5 + side_w * 0.5, cy, cz), yaw),
        _make_box_mesh((inner_x, depth, top_h), (cx, cy, cz + inner_z * 0.5 + top_h * 0.5), yaw),
        _make_box_mesh((inner_x, depth, top_h), (cx, cy, cz - inner_z * 0.5 - top_h * 0.5), yaw),
    ]
    return pieces


def _make_orbit_mesh(
    rng: np.random.Generator,
    center: tuple[float, float, float],
    yaw: float = 0.0,
) -> trimesh.Trimesh:
    """Small floating obstacle mixture inspired by MasterRacing's orbit objects."""
    primitive = rng.choice(("box", "cylinder", "sphere", "capsule"), p=(0.25, 0.25, 0.25, 0.25))
    if primitive == "box":
        mesh = trimesh.creation.box(extents=rng.uniform(0.1, 0.5, 3))
    elif primitive == "cylinder":
        mesh = trimesh.creation.cylinder(radius=rng.uniform(0.1, 0.3), height=rng.uniform(0.2, 0.6), sections=8)
    elif primitive == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=rng.uniform(0.1, 0.3))
    else:
        mesh = trimesh.creation.capsule(radius=rng.uniform(0.1, 0.3), height=rng.uniform(0.2, 0.6), count=[8, 8])
    mesh.apply_transform(trimesh.transformations.euler_matrix(0.0, rng.uniform(-0.8, 0.8), yaw, axes="rxyz"))
    mesh.apply_translation(center)
    return mesh


def composite_forest_terrain(difficulty: float, cfg: "CompositeForestTerrainCfg"):
    """Generate a static mesh forest with mixed primitive obstacles."""
    t = float(np.clip(difficulty, 0.0, 1.0))
    seed = None if cfg.seed is None else int(cfg.seed + round(t * 1_000_000))
    rng = np.random.default_rng(seed)
    meshes = [make_plane(cfg.size, height=0.0, center_zero=False)]
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], cfg.spawn_height), dtype=np.float64)

    num_obstacles = int(cfg.num_obstacles_range[0] + t * (cfg.num_obstacles_range[1] - cfg.num_obstacles_range[0]))
    min_clearance = cfg.clear_corridor_width_range[0] + t * (
        cfg.clear_corridor_width_range[1] - cfg.clear_corridor_width_range[0]
    )
    x_min, x_max = cfg.spawn_x_range
    y_min, y_max = cfg.spawn_y_range
    x_mid = 0.5 * cfg.size[0]
    y_mid = 0.5 * cfg.size[1]
    primitive_weights = np.asarray(cfg.primitive_weights, dtype=np.float64)
    if len(cfg.primitives) != len(primitive_weights):
        raise ValueError(
            "CompositeForestTerrainCfg.primitives and primitive_weights must have the same length."
        )
    primitive_weights = primitive_weights / primitive_weights.sum()

    for _ in range(num_obstacles):
        for _sample_attempt in range(64):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if abs(x - x_mid) >= min_clearance * 0.5 or abs(y - y_mid) >= cfg.spawn_keepout_y:
                break
        primitive = rng.choice(cfg.primitives, p=primitive_weights)
        yaw = rng.uniform(-math.pi, math.pi)

        if primitive == "column":
            radius = rng.uniform(*cfg.column_radius_range)
            height = rng.uniform(*cfg.height_range)
            transform = trimesh.transformations.translation_matrix((x, y, height * 0.5))
            meshes.append(trimesh.creation.cylinder(radius=radius, height=height, sections=8, transform=transform))
        elif primitive == "box":
            sx = rng.uniform(*cfg.box_size_xy_range)
            sy = rng.uniform(*cfg.box_size_xy_range)
            sz = rng.uniform(*cfg.height_range)
            meshes.append(_make_box_mesh((sx, sy, sz), (x, y, sz * 0.5), yaw))
        elif primitive == "wall":
            length = rng.uniform(*cfg.wall_length_range)
            thickness = rng.uniform(*cfg.wall_thickness_range)
            height = rng.uniform(*cfg.height_range)
            meshes.append(_make_box_mesh((length, thickness, height), (x, y, height * 0.5), yaw))
        elif primitive == "sphere":
            radius = rng.uniform(*cfg.sphere_radius_range)
            z = rng.uniform(cfg.floating_z_range[0], cfg.floating_z_range[1])
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
            sphere.apply_translation((x, y, z))
            meshes.append(sphere)
        elif primitive == "capsule":
            radius = rng.uniform(*cfg.sphere_radius_range)
            height = rng.uniform(*cfg.capsule_height_range)
            capsule = trimesh.creation.capsule(radius=radius, height=height, count=[8, 8])
            capsule.apply_transform(trimesh.transformations.euler_matrix(0.0, rng.uniform(-0.6, 0.6), yaw, axes="rxyz"))
            capsule.apply_translation((x, y, rng.uniform(cfg.floating_z_range[0], cfg.floating_z_range[1])))
            meshes.append(capsule)
        elif primitive == "cone":
            radius = rng.uniform(*cfg.column_radius_range)
            height = rng.uniform(*cfg.height_range)
            transform = trimesh.transformations.translation_matrix((x, y, height * 0.5))
            meshes.append(trimesh.creation.cone(radius=radius, height=height, sections=8, transform=transform))
        elif primitive == "rect_frame":
            outer_w = rng.uniform(*cfg.frame_outer_width_range)
            outer_h = rng.uniform(*cfg.frame_outer_height_range)
            inner_w = max(outer_w - rng.uniform(*cfg.frame_bar_width_range) * 2.0, 0.2)
            inner_h = max(outer_h - rng.uniform(*cfg.frame_bar_width_range) * 2.0, 0.2)
            depth = rng.uniform(*cfg.wall_thickness_range)
            center_z = rng.uniform(cfg.floating_z_range[0], cfg.floating_z_range[1])
            meshes.extend(
                _make_rect_frame_mesh((outer_w, outer_h, depth), (inner_w, inner_h, depth), (x, y, center_z), yaw)
            )
        elif primitive == "orbit":
            center_z = rng.uniform(cfg.floating_z_range[0], cfg.floating_z_range[1])
            meshes.append(_make_orbit_mesh(rng, (x, y, center_z), yaw))
        elif primitive == "ground_little":
            if rng.random() < 0.5:
                sx, sy = rng.uniform(0.1, 1.0, 2)
                sz = rng.uniform(0.1, 1.0)
                z = sz * 0.5 + rng.uniform(-0.1, 0.4)
                meshes.append(_make_box_mesh((sx, sy, sz), (x, y, z), yaw))
            else:
                radius = rng.uniform(0.05, 0.5)
                z = rng.uniform(-radius, radius) + rng.uniform(-0.1, 0.4)
                sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
                sphere.apply_translation((x, y, z))
                meshes.append(sphere)

    return meshes, origin


@configclass
class CompositeForestTerrainCfg(SubTerrainBaseCfg):
    """Static mesh forest with DiffLab-style mixed primitive obstacles."""

    function = composite_forest_terrain

    spawn_height: float = 2.0
    num_obstacles_range: tuple[int, int] = (8, 28)
    clear_corridor_width_range: tuple[float, float] = (3.0, 1.6)
    spawn_keepout_y: float = 0.8
    spawn_x_range: tuple[float, float] = (0.4, 7.6)
    spawn_y_range: tuple[float, float] = (0.4, 7.6)
    primitives: tuple[str, ...] = (
        "column",
        "box",
        "wall",
        "sphere",
        "capsule",
        "rect_frame",
        "orbit",
        "ground_little",
    )
    primitive_weights: tuple[float, ...] = (0.22, 0.16, 0.20, 0.10, 0.08, 0.06, 0.14, 0.04)
    height_range: tuple[float, float] = (1.0, 3.0)
    column_radius_range: tuple[float, float] = (0.05, 0.5)
    box_size_xy_range: tuple[float, float] = (0.05, 1.0)
    wall_length_range: tuple[float, float] = (0.4, 1.0)
    wall_thickness_range: tuple[float, float] = (0.04, 0.08)
    sphere_radius_range: tuple[float, float] = (0.1, 0.3)
    capsule_height_range: tuple[float, float] = (0.2, 0.6)
    floating_z_range: tuple[float, float] = (0.8, 2.0)
    frame_outer_width_range: tuple[float, float] = (0.8, 1.3)
    frame_outer_height_range: tuple[float, float] = (0.8, 1.3)
    frame_bar_width_range: tuple[float, float] = (0.15, 0.25)


def _apply_terrain_scene_overrides(sub_terrain_cfg: SubTerrainBaseCfg, scene_cfg) -> None:
    for key, value in scene_cfg.items():
        if key == "type":
            continue
        if not hasattr(sub_terrain_cfg, key):
            raise ValueError(f"Unknown terrain_scene option for {type(sub_terrain_cfg).__name__}: {key}")
        current = getattr(sub_terrain_cfg, key)
        if isinstance(current, tuple) and not isinstance(value, tuple):
            value = tuple(value)
        setattr(sub_terrain_cfg, key, value)


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

    ## Reward (structured navigation reward)

    - `distance`: Encourage approaching goal using exponential distance + progress term.
    - `time`: Encourage efficient completion with per-step penalty + remaining-time bonus.
    - `heading`: Encourage heading alignment toward the target.
    - `vel`: Velocity shaping around desired speed along goal direction.
    - `penalty`: Discourage unsafe/inefficient behaviors (action changes, collision, depth proximity).
    - `hover`: Reward stable hovering near the target.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{distance} + r_\text{time} + r_\text{heading} + r_\text{vel} + r_\text{penalty} + r_\text{hover}
    ```

    ## Episode End

    The episode ends when:
    - The drone collides with obstacles or terrain (detected via contact forces)
    - The drone flies too low (< 0.2m) or too high (> 4m)
    - NaN values are detected in the state

    ## Success Condition

    Success is achieved when the drone reaches the goal position within 0.4m on each axis
    (x, y, z) without collision, misbehavior (altitude/velocity violations), or NaN states.
    This can occur at any point during the episode or at truncation/timeout.

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
        # structured navigation reward parameters (defaults set in cfg/task/ForestDepth.yaml)
        self.reward_distance_lambda1 = cfg.task.get("reward_distance_lambda1", 1.0)
        self.reward_distance_lambda2 = cfg.task.get("reward_distance_lambda2", 0.5)
        self.reward_distance_alpha = cfg.task.get("reward_distance_alpha", 1.0)
        self.reward_time_eta = cfg.task.get("reward_time_eta", 0.01)
        self.reward_time_beta = cfg.task.get("reward_time_beta", 0.1)
        self.reward_heading_lambda3 = cfg.task.get("reward_heading_lambda3", 0.5)
        self.reward_vel_desired = cfg.task.get("reward_vel_desired", 1.0)
        self.reward_vel_k_v1 = cfg.task.get("reward_vel_k_v1", 0.8)
        self.reward_vel_k_v2 = cfg.task.get("reward_vel_k_v2", 1.2)
        self.reward_vel_sigma = cfg.task.get("reward_vel_sigma", 0.5)
        self.reward_vel_lambda = cfg.task.get("reward_vel_lambda", 1.0)
        self.reward_penalty_lambda4 = cfg.task.get("reward_penalty_lambda4", 0.1)
        self.reward_penalty_lambda5 = cfg.task.get("reward_penalty_lambda5", 5.0)
        self.reward_penalty_lambda6 = cfg.task.get("reward_penalty_lambda6", 1.0)
        self.reward_hover_lambda7 = cfg.task.get("reward_hover_lambda7", 1.0)
        self.reward_hover_distance = cfg.task.get("reward_hover_distance", 0.5)
        self.reward_hover_velocity = cfg.task.get("reward_hover_velocity", 0.2)
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

        self.target = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False,
        )
        self.target.initialize()

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

        target = objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            translation=(0.0, 0.0, 2.0),
            radius=0.08,
            color=torch.tensor([0.0, 0.0, 1.0])
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)

        terrain_scene_cfg = self.cfg.task.get("terrain_scene", {})
        terrain_scene_type = terrain_scene_cfg.get("type", "discrete_obstacles")
        if terrain_scene_type == "discrete_obstacles":
            sub_terrain_cfg = CurriculumObstaclesTerrainCfg()
        elif terrain_scene_type == "composite_mesh":
            sub_terrain_cfg = CompositeForestTerrainCfg()
        else:
            raise ValueError(f"Unknown ForestDepth terrain_scene.type: {terrain_scene_type}")
        _apply_terrain_scene_overrides(sub_terrain_cfg, terrain_scene_cfg)

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=7,
                num_cols=5,
                color_scheme="height",
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                curriculum=True,  # Enable row-based difficulty progression
                sub_terrains={
                    terrain_scene_type: sub_terrain_cfg
                },
            ),
            max_init_terrain_level=0,  # All environments start at row 0 (easiest)
            collision_group=-1,
            debug_vis=False,
        )
        self.terrain_size = terrain_cfg.terrain_generator.size
        self.max_terrain_level = max(0, terrain_cfg.terrain_generator.num_rows - 1)
        self.terrain = terrain_cfg.class_type(terrain_cfg)
        if self.terrain.terrain_origins is not None:
            y_vals = self.terrain.terrain_origins[..., 1]
            self.global_y_min = y_vals.amin().item()
            self.global_y_max = y_vals.amax().item()
            self._env_x_offsets = torch.zeros(self.num_envs, device=self.device)

        # Track per-environment terrain levels for curriculum learning
        self.env_terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Force all envs to start at a fixed row, regardless of column.
        # NOTE: num_rows를 늘리면 지형이 center 기준으로 재배치되어 row 0 시작점이
        # 더 멀어질 수 있음. 기존 row1 위치에서 시작하려면 start_row=1 유지.
        # 롤백하려면 start_row=0으로 되돌리면 됨.
        if self.terrain.terrain_origins is not None:
            start_row = 1  # 기존 row1 위치에서 시작하고 싶으면 1
            self.terrain.terrain_levels[:] = start_row
            self.terrain.env_origins[:] = self.terrain.terrain_origins[
                self.terrain.terrain_levels, self.terrain.terrain_types
            ]
            # Keep env_terrain_levels aligned with start_row so offsets/targets don't assume row 0.
            self.env_terrain_levels[:] = start_row
            origins = self.terrain.env_origins
            min_xyz = origins.amin(dim=0).tolist()
            max_xyz = origins.amax(dim=0).tolist()
            print(f"[ForestDepth] env_origins min xyz: {min_xyz} max xyz: {max_xyz}")
            if not hasattr(self, "target_pos"):
                self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
            self._refresh_row_x_offsets()
            self._update_env_targets(torch.arange(self.num_envs, device=self.device))

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

    def _refresh_row_x_offsets(self) -> None:
        if not hasattr(self, "_env_x_offsets") or self._env_x_offsets is None:
            self._env_x_offsets = torch.zeros(self.num_envs, device=self.device)
        if self.terrain.terrain_origins is None:
            return
        for level in range(self.max_terrain_level + 1):
            row_mask = self.env_terrain_levels == level
            if not row_mask.any():
                continue
            row_env_ids = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
            row_env_ids, _ = torch.sort(row_env_ids)
            row_x_vals = self.terrain.terrain_origins[level, :, 0]
            row_x_min = row_x_vals.amin() - 0.5 * self.terrain_size[0]
            row_x_max = row_x_vals.amax() + 0.5 * self.terrain_size[0]
            target_x = torch.linspace(
                row_x_min.item(),
                row_x_max.item(),
                row_env_ids.numel(),
                device=self.device,
            )
            origins_x = self.terrain.env_origins[row_env_ids, 0]
            self._env_x_offsets[row_env_ids] = target_x - origins_x

    def _get_env_x_offsets(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_env_x_offsets") or self._env_x_offsets is None:
            self._env_x_offsets = torch.zeros(self.num_envs, device=self.device)
        return self._env_x_offsets[env_ids]

    def update_curriculum(self, episode_stats: TensorDict) -> None:
        """Update terrain difficulty based on per-environment performance.

        Called after episode completion to progressively move environments between
        terrain rows based on their success/collision rates.

        Args:
            episode_stats: TensorDict with per-environment episode statistics
                - "success": (num_envs, 1) - 1.0 if goal reached without violations
                - "collision": (num_envs, 1) - 1.0 if collision occurred
        """
        if not hasattr(self, 'curriculum_manager'):
            return

        # Update performance tracking
        self.curriculum_manager.update_metrics(episode_stats)

        # Get terrain update decisions (per-environment)
        move_up, move_down, new_levels = self.curriculum_manager.get_terrain_updates()

        # Apply terrain reassignment via IsaacLab API
        if move_up.any() or move_down.any():
            env_ids = torch.arange(self.num_envs, device=self.device)
            moved = move_up | move_down
            if moved.any():
                moved_env_ids = env_ids[moved]
                self.env_terrain_levels[moved_env_ids] = new_levels[moved_env_ids].to(
                    self.env_terrain_levels.dtype
                )
                self.terrain.terrain_levels[moved_env_ids] = self.env_terrain_levels[moved_env_ids]
                self.terrain.env_origins[moved_env_ids] = self.terrain.terrain_origins[
                    self.terrain.terrain_levels[moved_env_ids],
                    self.terrain.terrain_types[moved_env_ids],
                ]
                self._refresh_row_x_offsets()
                self._update_env_targets(moved_env_ids)

            # Log transitions
            num_up = move_up.sum().item()
            num_down = move_down.sum().item()
            if num_up > 0:
                print(f"[Curriculum] {num_up} environments moved to harder terrain")
            if num_down > 0:
                print(f"[Curriculum] {num_down} environments moved to easier terrain")

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
            "collision": Unbounded(1),
            # Reward term tracking (already weighted as used in total reward)
            "reward_distance": Unbounded(1),
            "reward_time": Unbounded(1),
            "reward_heading": Unbounded(1),
            "reward_vel": Unbounded(1),
            "reward_penalty_action": Unbounded(1),
            "reward_penalty_collision": Unbounded(1),
            "reward_penalty_depth": Unbounded(1),
            "reward_hover": Unbounded(1),
            # Reset reason tracking (mutually exclusive - sum to 1.0)
            "reset_collision": Unbounded(1),  # Reset due to collision
            "reset_altitude": Unbounded(1),   # Reset due to altitude violation
            "reset_x_bound": Unbounded(1),    # Reset due to X-axis corridor violation
            "reset_nan": Unbounded(1),        # Reset due to NaN in state
            "reset_truncated": Unbounded(1),  # Reset due to timeout (no violations)
            # Performance metrics
            "avg_velocity": Unbounded(1),     # Average velocity magnitude
            "forward_progress": Unbounded(1), # Forward displacement
            "goal_reached": Unbounded(1),     # Drone within 0.4m of goal on all axes
            "success": Unbounded(1),          # Goal reached without collision/violation
            "z_pos": Unbounded(1),            # Current altitude
            "z_vel": Unbounded(1),            # Current vertical velocity
            "z_margin_low": Unbounded(1),     # Distance above lower altitude reset boundary
            "z_margin_high": Unbounded(1),    # Distance below upper altitude reset boundary
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        # Explicitly create stats with correct shape (num_envs, 1)
        self.stats = TensorDict({
            "return": torch.zeros(self.num_envs, 1, device=self.device),
            "episode_len": torch.zeros(self.num_envs, 1, device=self.device),
            "collision": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_distance": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_time": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_heading": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_vel": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_penalty_action": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_penalty_collision": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_penalty_depth": torch.zeros(self.num_envs, 1, device=self.device),
            "reward_hover": torch.zeros(self.num_envs, 1, device=self.device),
            "reset_collision": torch.zeros(self.num_envs, 1, device=self.device),
            "reset_altitude": torch.zeros(self.num_envs, 1, device=self.device),
            "reset_x_bound": torch.zeros(self.num_envs, 1, device=self.device),
            "reset_nan": torch.zeros(self.num_envs, 1, device=self.device),
            "reset_truncated": torch.zeros(self.num_envs, 1, device=self.device),
            "avg_velocity": torch.zeros(self.num_envs, 1, device=self.device),
            "forward_progress": torch.zeros(self.num_envs, 1, device=self.device),
            "goal_reached": torch.zeros(self.num_envs, 1, device=self.device),
            "success": torch.zeros(self.num_envs, 1, device=self.device),
            "z_pos": torch.zeros(self.num_envs, 1, device=self.device),
            "z_vel": torch.zeros(self.num_envs, 1, device=self.device),
            "z_margin_low": torch.zeros(self.num_envs, 1, device=self.device),
            "z_margin_high": torch.zeros(self.num_envs, 1, device=self.device),
        }, batch_size=[self.num_envs])
        self.prev_action = self.action_spec[("agents", "action")].zero()
        self.action_delta = self.reward_spec[("agents", "reward")].zero().squeeze(-1)
        self.prev_distance = torch.zeros(self.num_envs, 1, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.prev_action[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0

        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        origins = self.terrain.env_origins[env_ids]
        x_offsets = self._get_env_x_offsets(env_ids)
        pos[:, 0, 0] = origins[:, 0] + x_offsets
        pos[:, 0, 1] = -24.0
        pos[:, 0, 2] = 2.0
        self._update_env_targets(env_ids)

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

        # Initialize previous distance for distance-delta reward
        rpos = self.target_pos[env_ids] - pos
        self.prev_distance[env_ids] = rpos.norm(dim=-1)

        self.stats[env_ids] = 0.

    def _update_env_targets(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "target_pos"):
            self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        origins = self.terrain.env_origins[env_ids]
        x_offsets = self._get_env_x_offsets(env_ids)
        self.target_pos[env_ids, 0, 0] = origins[:, 0] + x_offsets
        self.target_pos[env_ids, 0, 1] = 24.0
        self.target_pos[env_ids, 0, 2] = 2.0
        if hasattr(self, "target"):
            self.target.set_world_poses(self.target_pos[env_ids], env_indices=env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.action_delta[:] = (actions - self.prev_action).abs().mean(dim=-1)
        self.prev_action[:] = actions
        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        # Update the depth camera after physics step
        self.depth_sensor.update(self.dt)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]

        # Get depth image from unified sensor
        if self.depth_cfg.backend in ("isaaclab", "isaaclab_raycaster"):
            self.depth_image = self.depth_sensor.get_depth()
        else:  # simple_raycaster
            pos_w = self.drone.pos.squeeze(1)
            rot_w = self.drone.rot.squeeze(1)
            self.depth_image = self.depth_sensor.get_depth(pos_w, rot_w)
        
        # Compute depth range pixels for reward computation.
        # We need DISTANCE (0=Close, 1=Far) because:
        # 1. We take torch.amin() later to find the closest obstacle
        # 2. The exponential reward exp(-dist) gives high penalty when dist is small
        if self.depth_cfg.processing.normalize:
             self.depth_range_pixels = self.depth_image
        else:
             self.depth_range_pixels = self.depth_image / self.depth_range

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
        distance = self.rpos.norm(dim=-1)  # (num_envs, 1)
        vel_direction = self.rpos / distance.clamp_min(1e-6).unsqueeze(-1)

        # Distance-related reward: exponential + progress
        reward_distance = (
            self.reward_distance_lambda1 * torch.exp(-self.reward_distance_alpha * distance)
            + self.reward_distance_lambda2 * (self.prev_distance - distance)
        )

        # Time-related reward
        t_rem = (self.max_episode_length - self.progress_buf).clamp_min(0).unsqueeze(-1)
        reward_time = -self.reward_time_eta + self.reward_time_beta * (t_rem / self.max_episode_length)

        # Heading reward: alignment with target direction
        heading_vec = self.drone.heading
        heading_alignment = (heading_vec * vel_direction).sum(-1).clamp(-1.0, 1.0)
        reward_heading = self.reward_heading_lambda3 * heading_alignment

        # Velocity shaping reward around desired speed
        v_des = self.reward_vel_desired
        v_mag = self.drone.vel_w[..., :3].norm(dim=-1)
        v_dot = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)
        upper_excess = torch.clamp(v_mag - self.reward_vel_k_v2 * v_des, min=0.0)
        lower_excess = torch.clamp(self.reward_vel_k_v1 * v_des - v_mag, min=0.0)
        in_band = (v_mag >= self.reward_vel_k_v1 * v_des) & (v_mag <= self.reward_vel_k_v2 * v_des)
        reward_vel = (
            v_dot
            - upper_excess.pow(2)
            - lower_excess.pow(2)
            + in_band.float() * torch.exp(-((v_mag - v_des) ** 2) / (2.0 * self.reward_vel_sigma ** 2))
        ) * self.reward_vel_lambda

        # Penalty reward: action change + collision + depth proximity
        reward_penalty_action = -self.reward_penalty_lambda4 * self.action_delta

        # Depth proximity penalty from minimum depth (0=close, 1=far)
        min_depth = torch.amin(self.depth_range_pixels, dim=(2, 3))
        reward_penalty_depth = -4.0 * self.reward_penalty_lambda6 * torch.exp(-(min_depth ** 2))

        # Check for collision using contact forces only
        collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        force_magnitude = torch.zeros(self.num_envs, 1, device=self.device)

        reward_penalty_collision = torch.zeros_like(reward_penalty_action)
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

            # Collision penalty term
            reward_penalty_collision = -self.reward_penalty_lambda5 * collision.float()

        # Termination conditions
        # Track individual termination reasons for metrics
        z_pos = self.drone.pos[..., 2]
        z_vel = self.drone.vel_w[..., 2]
        z_margin_low = z_pos - 0.2
        z_margin_high = 4.0 - z_pos
        altitude_violation = (z_pos < 0.2) | (z_pos > 4.)
        # X-axis bound check (drone should stay within corridor as it moves along Y-axis)
        x_bound_violation = (self.drone.pos[..., 0] < -22.0) | (self.drone.pos[..., 0] > 22.0)
        misbehave = altitude_violation | x_bound_violation
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

        # Goal reached: within 0.4m on each axis (x, y, z)
        # self.rpos shape: (num_envs, 1, 3)
        goal_reached = (self.rpos.abs() < 0.4).all(dim=-1)  # (num_envs, 1)

        # Success: reached goal without collision, misbehavior, or NaN
        # Can occur at truncation/timeout OR at any point during episode
        success = goal_reached & ~collision & ~misbehave & ~hasnan

        # Hover reward: stable hover near target
        vel_magnitude = self.drone.vel_w[..., :3].norm(dim=-1)
        hover_mask = (distance < self.reward_hover_distance) & (vel_magnitude < self.reward_hover_velocity)
        reward_hover = self.reward_hover_lambda7 * hover_mask.float()

        # Total reward
        reward_penalty = reward_penalty_action + reward_penalty_collision + reward_penalty_depth
        reward = reward_distance + reward_time + reward_heading + reward_vel + reward_penalty + reward_hover

        # Track reset reasons (mutually exclusive - sum to 1.0)
        # Priority: collision > altitude > x_bound > nan > truncated (timeout without violations)
        reset_collision = collision & terminated
        reset_altitude = ~collision & altitude_violation & terminated
        reset_x_bound = ~collision & ~altitude_violation & x_bound_violation & terminated
        reset_nan = ~collision & ~misbehave & hasnan & terminated
        # Truncated ONLY if timeout without any violations
        reset_truncated = truncated & ~terminated
        

        # Update stats
        self.stats["reward_distance"].add_(reward_distance)
        self.stats["reward_time"].add_(reward_time)
        self.stats["reward_heading"].add_(reward_heading)
        self.stats["reward_vel"].add_(reward_vel)
        self.stats["reward_penalty_action"].add_(reward_penalty_action)
        self.stats["reward_penalty_collision"].add_(reward_penalty_collision)
        self.stats["reward_penalty_depth"].add_(reward_penalty_depth)
        self.stats["reward_hover"].add_(reward_hover)
        # Accumulate collision during episode (once True, stays True)
        self.stats["collision"] = torch.maximum(
            self.stats["collision"], collision.float()
        )

        # Reset reason tracking (only update at episode end for mutual exclusivity)
        # These are mutually exclusive and sum to 1.0 across all episodes
        episode_ended = terminated | truncated
        self.stats['reset_collision'] = torch.where(
            episode_ended, reset_collision.float(), self.stats['reset_collision']
        )
        self.stats['reset_altitude'] = torch.where(
            episode_ended, reset_altitude.float(), self.stats['reset_altitude']
        )
        self.stats['reset_x_bound'] = torch.where(
            episode_ended, reset_x_bound.float(), self.stats['reset_x_bound']
        )
        self.stats['reset_nan'] = torch.where(
            episode_ended, reset_nan.float(), self.stats['reset_nan']
        )
        self.stats["reset_truncated"] = torch.where(
            episode_ended, reset_truncated.float(), self.stats["reset_truncated"]
        )

        # Performance metrics
        self.stats["avg_velocity"][:] = avg_velocity
        self.stats["forward_progress"][:] = forward_progress
        self.stats["z_pos"][:] = z_pos
        self.stats["z_vel"][:] = z_vel
        self.stats["z_margin_low"][:] = z_margin_low
        self.stats["z_margin_high"][:] = z_margin_high
        # Accumulate goal_reached during episode (once True, stays True)
        self.stats["goal_reached"] = torch.maximum(
            self.stats["goal_reached"], goal_reached.float()
        )
        # Accumulate success during episode (once True, stays True)
        self.stats["success"] = torch.maximum(
            self.stats["success"], success.float()
        )

        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        # Update previous distance for next step
        self.prev_distance = distance.detach()

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
