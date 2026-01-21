"""
Simple raycaster backend for depth camera using warp-based raycasting.
"""

import math
import torch
from typing import Tuple, Optional

from .depth_camera_cfg import DepthCameraCfg
from omni_drones.utils.torch import quat_rotate, quat_mul


class RaycasterDepthBackend:
    """
    Raycaster-based depth camera backend.
    
    Uses simple_raycaster package for warp-based depth estimation.
    Supports configurable camera offset, rotation, and FOV.
    
    Args:
        cfg: Depth camera configuration
        num_envs: Number of parallel environments
        device: Torch device
    """
    
    def __init__(
        self,
        cfg: DepthCameraCfg,
        num_envs: int,
        device: torch.device
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        # Will be initialized in initialize()
        self._raycaster = None
        self._ray_dirs_cam = None
        self._mesh_pos_w = None
        self._mesh_quat_w = None
        self._offset_pos = None
        self._offset_rot = None
    
    def initialize(self):
        """Initialize raycaster and camera geometry."""
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
        
        # Create raycaster from stage meshes
        stage = stage_utils.get_current_stage()
        self._raycaster = create_raycaster_from_stage(
            MultiMeshRaycaster,
            stage=stage,
            paths=self.cfg.mesh_prim_paths,
            device=self.device,
            simplify_factor=self.cfg.simplify_factor,
        )
        
        # Setup mesh poses
        mesh_pos, mesh_quat = None, None
        if self.cfg.mesh_poses_from_stage:
            mesh_prims = select_mesh_prims(self.cfg.mesh_prim_paths)
            mesh_pos, mesh_quat = compute_mesh_poses(mesh_prims)
        
        self._mesh_pos_w, self._mesh_quat_w = build_mesh_pose_tensors(
            self.num_envs, mesh_pos, mesh_quat, self.device
        )
        
        if self._mesh_pos_w is None and self.cfg.mesh_count is not None:
            self._mesh_pos_w, self._mesh_quat_w = default_mesh_pose_tensors(
                self.num_envs, self.cfg.mesh_count, self.device
            )
        
        # Build ray directions in camera frame
        self._ray_dirs_cam = self._build_depth_dirs_cam()
        
        # Setup camera offset
        self._offset_pos = torch.tensor(
            self.cfg.offset_pos, device=self.device, dtype=torch.float32
        )
        self._offset_rot = torch.tensor(
            self.cfg.offset_rot_wxyz, device=self.device, dtype=torch.float32
        )
    
    def raycast(
        self, 
        drone_pos: torch.Tensor, 
        drone_rot: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform raycasting to get depth image.
        
        Args:
            drone_pos: Drone positions (num_envs, 3)
            drone_rot: Drone rotations as quaternions (num_envs, 4) in wxyz format
        
        Returns:
            Depth image (num_envs, 1, H, W)
        """
        from omni_drones.envs.utils.simple_raycaster import raycast_fused
        
        # Compute camera pose in world frame
        offset_pos = self._offset_pos.expand(self.num_envs, -1)
        offset_rot = self._offset_rot.expand(self.num_envs, -1)
        
        cam_pos_w = drone_pos + quat_rotate(drone_rot, offset_pos)
        cam_rot_w = quat_mul(drone_rot, offset_rot)
        
        # Transform ray directions to world frame
        num_rays = self._ray_dirs_cam.shape[0]
        ray_dirs_cam = self._ray_dirs_cam.unsqueeze(0).expand(self.num_envs, -1, -1)
        rot_rep = cam_rot_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4)
        ray_dirs_w = quat_rotate(
            rot_rep, 
            ray_dirs_cam.reshape(-1, 3)
        ).reshape(self.num_envs, num_rays, 3)
        
        # Ray starts
        ray_starts_w = cam_pos_w.unsqueeze(1).expand(-1, num_rays, -1)
        
        # Perform raycasting
        _, ray_dists = raycast_fused(
            self._raycaster,
            ray_starts_w=ray_starts_w,
            ray_dirs_w=ray_dirs_w,
            min_dist=0.0,
            max_dist=self.cfg.range,
            mesh_pos_w=self._mesh_pos_w,
            mesh_quat_w=self._mesh_quat_w,
        )
        
        # Handle NaN/Inf values
        ray_dists = torch.nan_to_num(
            ray_dists, 
            nan=self.cfg.range, 
            posinf=self.cfg.range, 
            neginf=0.0
        )
        
        # Reshape to image format: (num_envs, 1, H, W)
        depth_image = ray_dists.reshape(
            self.num_envs, 
            self.cfg.resolution[0], 
            self.cfg.resolution[1]
        ).unsqueeze(1).clamp(0, self.cfg.range)
        
        return depth_image
    
    def _build_depth_dirs_cam(self) -> torch.Tensor:
        """
        Build ray directions in camera frame.
        
        Creates a grid of ray directions based on camera resolution and FOV.
        Directions are normalized and point forward in camera frame.
        
        Returns:
            Ray directions tensor (H*W, 3)
        """
        depth_h, depth_w = self.cfg.resolution
        
        # Compute horizontal and vertical FOV
        # FOV can be computed from focal length and aperture
        focal_length = self.cfg.focal_length
        horizontal_aperture = self.cfg.horizontal_aperture
        
        hfov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
        aspect = depth_w / depth_h
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)
        
        # Create pixel grid in normalized coordinates [-1, 1]
        xs = (torch.arange(depth_w, device=self.device) + 0.5 - depth_w / 2.0) / (depth_w / 2.0)
        ys = (torch.arange(depth_h, device=self.device) + 0.5 - depth_h / 2.0) / (depth_h / 2.0)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        
        # Convert to 3D ray directions
        # Camera frame: X-forward, Y-right, Z-down (or up depending on convention)
        x = torch.ones_like(grid_x)  # Forward component
        y = grid_x * math.tan(hfov / 2.0)  # Horizontal component
        z = -grid_y * math.tan(vfov / 2.0)  # Vertical component
        
        # Stack and reshape to (H*W, 3)
        dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        
        # Normalize directions
        return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
