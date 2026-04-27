"""
Unified depth camera sensor supporting multiple backends.

Backends:
- IsaacLab TiledCamera
- IsaacLab RayCasterCamera
- IsaacLab MultiMeshRayCasterCamera
- simple_raycaster
Provides consistent API with optional preprocessing pipeline (noise, normalization) for DEAN compatibility.
"""

import torch
from typing import Optional
import re

from .depth_camera_cfg import DepthCameraCfg, DepthProcessingCfg


class DepthCamera:
    """
    Unified depth camera sensor with backend abstraction.
    
    Supports:
    - IsaacLab TiledCamera backend (GPU-accelerated multi-env rendering)
    - IsaacLab RayCasterCamera backend (static mesh raycasting)
    - IsaacLab MultiMeshRayCasterCamera backend (multi/dynamic mesh raycasting)
    - simple_raycaster backend (Warp-based raycasting)
    
    Features:
    - Configurable mount position and orientation
    - Multiple depth data types (distance_to_camera, distance_to_image_plane, depth)
    - Optional preprocessing: clamping, noise injection, normalization
    - DEAN-compatible configuration
    
    Usage:
        cfg = DepthCameraCfg(
            backend="isaaclab",
            resolution=(64, 64),
            range=10.0,
            data_type="distance_to_image_plane"
        )
        camera = DepthCamera(cfg, num_envs=64, device="cuda", drone_prim_path="/World/envs/env_.*/Hummingbird_0")
        camera.initialize()
        
        # In environment loop:
        camera.update(dt)
        depth = camera.get_depth()  # (num_envs, 1, H, W)
    """
    
    def __init__(
        self,
        cfg: DepthCameraCfg,
        num_envs: int,
        device: torch.device,
        drone_prim_path: str
    ):
        """
        Initialize depth camera sensor.
        
        Args:
            cfg: Depth camera configuration
            num_envs: Number of parallel environments
            device: Torch device (cuda/cpu)
            drone_prim_path: Prim path pattern for drone (e.g., "/World/envs/env_.*/Drone_0")
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.drone_prim_path = drone_prim_path
        
        self._backend = None
        self._sensor_prim_path = f"{self.drone_prim_path}/base_link/DepthCamera"
        self._depth_image = torch.zeros(
            num_envs, 1, cfg.resolution[0], cfg.resolution[1],
            device=device
        )
        
        # Setup backend-specific components
        if cfg.backend == "isaaclab":
            self._setup_isaaclab_backend()
        elif cfg.backend == "isaaclab_raycaster":
            self._setup_isaaclab_raycaster_backend()
        elif cfg.backend in ["isaaclab_multimesh_raycaster", "isaaclab_multimesh"]:
            self._setup_isaaclab_multimesh_raycaster_backend()
        elif cfg.backend == "simple_raycaster":
            self._setup_raycaster_backend()
        else:
            raise ValueError(f"Unknown depth backend: {cfg.backend}")

    def _cfg_value(self, name: str, default=None):
        """Read config values from dataclasses or OmegaConf containers."""
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(self.cfg, DictConfig):
                value = self.cfg.get(name, default)
                if OmegaConf.is_config(value):
                    return OmegaConf.to_container(value, resolve=True)
                return value
        except ImportError:
            pass
        return getattr(self.cfg, name, default)
    
    def _setup_isaaclab_backend(self):
        """Setup IsaacLab TiledCamera backend."""
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        
        # Determine data type for TiledCamera
        if self.cfg.data_type == "distance_to_camera":
            data_types = ["distance_to_camera"]
        elif self.cfg.data_type == "distance_to_image_plane":
            data_types = ["distance_to_image_plane"]
        elif self.cfg.data_type == "depth":
            # Depth (z-buffer) - DEAN compatible
            data_types = ["distance_to_image_plane"]  # Use this and convert if needed
        else:
            raise ValueError(f"Unknown data_type: {self.cfg.data_type}")
        
        
        # Create TiledCamera configuration
        # Convert OmegaConf config to plain dict to avoid recursion in IsaacLab validation
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(self.cfg, DictConfig):
                cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
                offset_pos = cfg_dict["offset_pos"]
                offset_rot = cfg_dict["offset_rot_wxyz"]
                clipping_range = cfg_dict["clipping_range"]
                focal_length = cfg_dict["focal_length"]
                horizontal_aperture = cfg_dict["horizontal_aperture"]
                resolution = cfg_dict["resolution"]
                convention = cfg_dict["convention"]
            else:
                offset_pos = self.cfg.offset_pos
                offset_rot = self.cfg.offset_rot_wxyz
                clipping_range = self.cfg.clipping_range
                focal_length = self.cfg.focal_length
                horizontal_aperture = self.cfg.horizontal_aperture
                resolution = self.cfg.resolution
                convention = self.cfg.convention
        except ImportError:
            offset_pos = self.cfg.offset_pos
            offset_rot = self.cfg.offset_rot_wxyz
            clipping_range = self.cfg.clipping_range
            focal_length = self.cfg.focal_length
            horizontal_aperture = self.cfg.horizontal_aperture
            resolution = self.cfg.resolution
            convention = self.cfg.convention

        # Ensure USD-expected types (tuples of floats/ints)
        offset_pos = tuple(float(x) for x in offset_pos)
        offset_rot = tuple(float(x) for x in offset_rot)
        clipping_range = tuple(float(x) for x in clipping_range)
        resolution = tuple(int(x) for x in resolution)
        focal_length = float(focal_length)
        horizontal_aperture = float(horizontal_aperture)
        convention = str(convention)
        
        depth_camera_cfg = TiledCameraCfg(
            prim_path=self._sensor_prim_path,
            offset=TiledCameraCfg.OffsetCfg(
                pos=offset_pos,
                rot=offset_rot,
                convention=convention,
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
            ),
            width=int(resolution[1]),
            height=int(resolution[0]),
            data_types=data_types,
        )

        
        self._backend = depth_camera_cfg.class_type(depth_camera_cfg)
        self._backend_type = "isaaclab"

    def _setup_isaaclab_raycaster_backend(self):
        """Setup IsaacLab RayCasterCamera backend."""
        from isaaclab.sensors import RayCasterCameraCfg, patterns

        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(self.cfg, DictConfig):
                cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
                offset_pos = cfg_dict["offset_pos"]
                offset_rot = cfg_dict["offset_rot_wxyz"]
                focal_length = cfg_dict["focal_length"]
                horizontal_aperture = cfg_dict["horizontal_aperture"]
                resolution = cfg_dict["resolution"]
                convention = cfg_dict["convention"]
                mesh_prim_paths = cfg_dict["mesh_prim_paths"]
                data_type = cfg_dict["data_type"]
                depth_clipping_behavior = cfg_dict["depth_clipping_behavior"]
            else:
                offset_pos = self.cfg.offset_pos
                offset_rot = self.cfg.offset_rot_wxyz
                focal_length = self.cfg.focal_length
                horizontal_aperture = self.cfg.horizontal_aperture
                resolution = self.cfg.resolution
                convention = self.cfg.convention
                mesh_prim_paths = self.cfg.mesh_prim_paths
                data_type = self.cfg.data_type
                depth_clipping_behavior = self.cfg.depth_clipping_behavior
        except ImportError:
            offset_pos = self.cfg.offset_pos
            offset_rot = self.cfg.offset_rot_wxyz
            focal_length = self.cfg.focal_length
            horizontal_aperture = self.cfg.horizontal_aperture
            resolution = self.cfg.resolution
            convention = self.cfg.convention
            mesh_prim_paths = self.cfg.mesh_prim_paths
            data_type = self.cfg.data_type
            depth_clipping_behavior = self.cfg.depth_clipping_behavior

        if data_type not in ["distance_to_camera", "distance_to_image_plane", "depth"]:
            raise ValueError(f"Unknown data_type: {data_type}")

        height = int(resolution[0])
        width = int(resolution[1])
        focal_length = float(focal_length)
        horizontal_aperture = float(horizontal_aperture)
        fx = width * focal_length / horizontal_aperture
        fy = fx
        cx = width * 0.5
        cy = height * 0.5
        raycaster_data_type = (
            data_type if data_type != "depth" else "distance_to_image_plane"
        )
        data_types = [raycaster_data_type]
        if self._cfg_value("include_distance_to_camera", True) and "distance_to_camera" not in data_types:
            data_types.append("distance_to_camera")
        raycaster_clipping_behavior = (
            "none" if depth_clipping_behavior == "nan" else depth_clipping_behavior
        )

        # RayCasterCamera registers a play-event initialize callback when the
        # backend object is constructed. Create the tracked Xform prims first so
        # the callback cannot race ahead and fail on the regex path.
        self._ensure_raycaster_tracking_prims()

        raycaster_camera_cfg = RayCasterCameraCfg(
            prim_path=self._sensor_prim_path,
            debug_vis=False,
            mesh_prim_paths=list(mesh_prim_paths),
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=tuple(float(x) for x in offset_pos),
                rot=tuple(float(x) for x in offset_rot),
                convention=str(convention),
            ),
            attach_yaw_only=False,
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                width=width,
                height=height,
                intrinsic_matrix=[
                    fx, 0.0, cx,
                    0.0, fy, cy,
                    0.0, 0.0, 1.0,
                ],
            ),
            data_types=data_types,
            max_distance=float(self.cfg.range),
            depth_clipping_behavior=str(raycaster_clipping_behavior),
        )

        self._backend = raycaster_camera_cfg.class_type(raycaster_camera_cfg)
        self._backend_type = "isaaclab_raycaster"

    def _setup_isaaclab_multimesh_raycaster_backend(self):
        """Setup IsaacLab MultiMeshRayCasterCamera backend."""
        from isaaclab.sensors import patterns
        from isaaclab.sensors.ray_caster import (
            MultiMeshRayCasterCameraCfg,
            MultiMeshRayCasterCfg,
        )

        offset_pos = self._cfg_value("offset_pos")
        offset_rot = self._cfg_value("offset_rot_wxyz")
        focal_length = self._cfg_value("focal_length")
        horizontal_aperture = self._cfg_value("horizontal_aperture")
        resolution = self._cfg_value("resolution")
        convention = self._cfg_value("convention")
        mesh_prim_paths = self._cfg_value("mesh_prim_paths")
        data_type = self._cfg_value("data_type")
        depth_clipping_behavior = self._cfg_value("depth_clipping_behavior")

        if data_type not in ["distance_to_camera", "distance_to_image_plane", "depth"]:
            raise ValueError(f"Unknown data_type: {data_type}")

        height = int(resolution[0])
        width = int(resolution[1])
        focal_length = float(focal_length)
        horizontal_aperture = float(horizontal_aperture)
        fx = width * focal_length / horizontal_aperture
        fy = fx
        cx = width * 0.5
        cy = height * 0.5
        raycaster_data_type = (
            data_type if data_type != "depth" else "distance_to_image_plane"
        )
        data_types = [raycaster_data_type]
        if self._cfg_value("include_distance_to_camera", True) and "distance_to_camera" not in data_types:
            data_types.append("distance_to_camera")
        raycaster_clipping_behavior = (
            "none" if depth_clipping_behavior == "nan" else depth_clipping_behavior
        )

        self._ensure_raycaster_tracking_prims()
        mesh_targets = self._build_multimesh_targets(
            MultiMeshRayCasterCfg.RaycastTargetCfg,
            mesh_prim_paths,
        )

        raycaster_camera_cfg = MultiMeshRayCasterCameraCfg(
            prim_path=self._sensor_prim_path,
            debug_vis=False,
            mesh_prim_paths=mesh_targets,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=tuple(float(x) for x in offset_pos),
                rot=tuple(float(x) for x in offset_rot),
                convention=str(convention),
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                width=width,
                height=height,
                intrinsic_matrix=[
                    fx, 0.0, cx,
                    0.0, fy, cy,
                    0.0, 0.0, 1.0,
                ],
            ),
            data_types=data_types,
            max_distance=float(self._cfg_value("range")),
            depth_clipping_behavior=str(raycaster_clipping_behavior),
            update_mesh_ids=bool(self._cfg_value("update_mesh_ids", False)),
            reference_meshes=bool(self._cfg_value("reference_meshes", True)),
        )

        self._backend = raycaster_camera_cfg.class_type(raycaster_camera_cfg)
        self._backend_type = "isaaclab_multimesh_raycaster"

    def _build_multimesh_targets(self, target_cfg_type, mesh_prim_paths):
        """Build MultiMesh raycast targets from config or legacy paths."""
        target_dicts = self._cfg_value("raycast_targets", None)
        if target_dicts:
            targets = []
            for target in target_dicts:
                target = dict(target)
                targets.append(
                    target_cfg_type(
                        prim_expr=target["prim_expr"],
                        is_shared=bool(target.get("is_shared", True)),
                        merge_prim_meshes=bool(target.get("merge_prim_meshes", True)),
                        track_mesh_transforms=bool(target.get("track_mesh_transforms", False)),
                    )
                )
            return targets

        return [
            target_cfg_type(
                prim_expr=str(path),
                is_shared=True,
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            )
            for path in mesh_prim_paths
        ]

    def _ensure_raycaster_tracking_prims(self):
        """Create empty Xform prims that RayCasterCamera can track."""
        from isaaclab.sim.utils import prims as prim_utils

        is_prim_path_valid = getattr(prim_utils, "is_prim_path_valid", None)
        if is_prim_path_valid is None:
            from isaaclab.sim.utils.stage import get_current_stage

            def is_prim_path_valid(prim_path):
                stage = get_current_stage()
                return stage is not None and stage.GetPrimAtPath(prim_path).IsValid()

        if ".*" in self._sensor_prim_path:
            sensor_prim_paths = [
                re.sub(r"env_\.\*", f"env_{env_id}", self._sensor_prim_path)
                for env_id in range(self.num_envs)
            ]
        else:
            sensor_prim_paths = [self._sensor_prim_path]

        for prim_path in sensor_prim_paths:
            if not is_prim_path_valid(prim_path):
                prim_utils.create_prim(prim_path, prim_type="Xform")
    
    def _setup_raycaster_backend(self):
        """Setup simple_raycaster backend."""
        # Import raycaster backend module (will be created in Step 3)
        from .raycaster_depth import RaycasterDepthBackend
        
        self._backend = RaycasterDepthBackend(
            cfg=self.cfg,
            num_envs=self.num_envs,
            device=self.device
        )
        self._backend_type = "simple_raycaster"
    
    def initialize(self):
        """Initialize the depth camera sensor."""
        if self._backend_type in ["isaaclab_raycaster", "isaaclab_multimesh_raycaster"]:
            self._ensure_raycaster_tracking_prims()
            self._backend._initialize_impl()
        elif self._backend_type == "isaaclab":
            self._backend._initialize_impl()
        elif self._backend_type == "simple_raycaster":
            self._backend.initialize()
    
    def update(self, dt: float):
        """
        Update sensor data.
        
        Args:
            dt: Time step in seconds
        """
        if self._backend_type in ["isaaclab", "isaaclab_raycaster", "isaaclab_multimesh_raycaster"]:
            self._backend.update(dt)
        # Simple raycaster doesn't need periodic updates
    
    def get_depth(self, drone_pos: Optional[torch.Tensor] = None, 
                  drone_rot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get processed depth image.
        
        Args:
            drone_pos: Drone positions (num_envs, 3), required for raycaster backend
            drone_rot: Drone rotations as quaternions (num_envs, 4), required for raycaster backend
        
        Returns:
            Depth image tensor (num_envs, 1, H, W)
        """
        # Get raw depth data from backend
        if self._backend_type in ["isaaclab", "isaaclab_raycaster", "isaaclab_multimesh_raycaster"]:
            depth_data = self._get_isaaclab_depth()
        elif self._backend_type == "simple_raycaster":
            if drone_pos is None or drone_rot is None:
                raise ValueError("drone_pos and drone_rot required for raycaster backend")
            depth_data = self._backend.raycast(drone_pos, drone_rot)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")
        
        # Apply preprocessing pipeline
        processed_depth = self._apply_depth_processing(depth_data)
        
        return processed_depth
    
    def _get_isaaclab_depth(self) -> torch.Tensor:
        """Extract depth data from IsaacLab TiledCamera."""
        # Get appropriate data type
        if self.cfg.data_type in ["distance_to_camera", "distance_to_image_plane"]:
            depth_data = self._backend.data.output[self.cfg.data_type]
        else:  # depth
            # Use distance_to_image_plane as proxy for depth
            depth_data = self._backend.data.output["distance_to_image_plane"]
        
        # Handle NaN and Inf values
        # Set depth_clipping_behavior: "max" fills with max range, "nan" keeps NaNs
        if self.cfg.depth_clipping_behavior == "max":
            depth_data = torch.nan_to_num(
                depth_data, 
                nan=self.cfg.range, 
                posinf=self.cfg.range, 
                neginf=0.0
            )
        elif self.cfg.depth_clipping_behavior == "nan":
            # Keep NaNs for explicit handling downstream
            pass
        
        # Reshape: (num_envs, h, w, 1) -> (num_envs, 1, h, w)
        depth_image = depth_data.permute(0, 3, 1, 2).clamp(0, self.cfg.range)
        
        return depth_image
    
    def _apply_depth_processing(self, raw_depth: torch.Tensor) -> torch.Tensor:
        """
        Apply preprocessing pipeline to raw depth data.
        
        Pipeline (DEAN-compatible):
        1. Clamp to valid range
        2. Add noise (if enabled)
        3. Normalize (if enabled)
        
        Args:
            raw_depth: Raw depth tensor (num_envs, 1, H, W)
        
        Returns:
            Processed depth tensor (num_envs, 1, H, W)
        """
        depth = raw_depth.clone()
        
        # Step 1: Clamp to valid range
        min_val = self.cfg.processing.clamp_min if self.cfg.processing.clamp_min is not None else 0.0
        max_val = self.cfg.processing.clamp_max if self.cfg.processing.clamp_max is not None else self.cfg.range
        depth = depth.clamp(min_val, max_val)
        
        # Step 2: Add noise (if enabled)
        if self.cfg.processing.add_noise:
            depth = self._add_noise(depth)
        
        # Step 3: Normalize (if enabled)
        if self.cfg.processing.normalize:
            # Normalize from [min_val, max_val] to normalize_range
            norm_min, norm_max = self.cfg.processing.normalize_range
            depth = (depth - min_val) / (max_val - min_val)  # to [0, 1]
            depth = depth * (norm_max - norm_min) + norm_min  # to [norm_min, norm_max]
        
        return depth
    
    def _add_noise(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Add noise to depth measurements.
        
        Args:
            depth: Depth tensor (num_envs, 1, H, W)
        
        Returns:
            Noisy depth tensor
        """
        if self.cfg.processing.noise_type == "gaussian":
            std = self.cfg.processing.noise_params.get("std", 0.01)
            noise = torch.randn_like(depth) * std
            return depth + noise
        
        elif self.cfg.processing.noise_type == "dropout":
            prob = self.cfg.processing.noise_params.get("prob", 0.1)
            mask = torch.rand_like(depth) > prob
            return depth * mask
        
        else:
            raise ValueError(f"Unknown noise type: {self.cfg.processing.noise_type}")
    
    @property
    def data(self):
        """Access to underlying backend data (for IsaacLab backend)."""
        if self._backend_type in ["isaaclab", "isaaclab_raycaster", "isaaclab_multimesh_raycaster"]:
            return self._backend.data
        else:
            raise NotImplementedError("data property only available for IsaacLab backends")
