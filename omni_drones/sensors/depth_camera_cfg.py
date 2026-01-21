"""
Depth camera configuration dataclasses for unified sensor API.

This module provides configuration structures for the DepthCamera sensor,
supporting multiple backends with consistent configuration interface.

## Backends

### IsaacLab TiledCamera
- GPU-accelerated rendering using Isaac Sim's replicator
- Efficient multi-environment parallel depth rendering
- Supports distance_to_camera, distance_to_image_plane, depth
- Requires enable_replicator=true in sim config

### SimpleRaycaster
- Warp-based CPU/GPU raycasting
- Faster startup, lower memory overhead
- Best for simpler geometry
- Requires simple_raycaster package

## Coordinate Conventions

### ROS Convention (default)
- X-forward, Y-left, Z-up
- Default rotation for forward-facing camera: [0.5, -0.5, 0.5, -0.5] (wxyz)

### OpenGL Convention  
- X-right, Y-up, Z-backward
- DEAN-compatible rotation: [0.5, 0.5, -0.5, -0.5] (wxyz)

## Depth Data Types

- **distance_to_camera**: Euclidean distance from camera origin (legacy)
- **distance_to_image_plane**: Distance along optical axis (recommended)
- **depth**: Z-buffer depth (DEAN-compatible)

## DEAN Compatibility

To configure for DEAN-style depth preprocessing:
```python
from omni_drones.sensors import DepthCameraCfg, DepthProcessingCfg

cfg = DepthCameraCfg(
    backend="isaaclab",
    convention="opengl",
    offset_rot_wxyz=[0.5, 0.5, -0.5, -0.5],
    data_type="depth",
    processing=DepthProcessingCfg(
        normalize=True,
        normalize_range=(0.0, 1.0),
        add_noise=True,
        noise_type="gaussian",
        noise_params={"std": 0.01}
    )
)
```

Or in YAML:
```yaml
depth_camera:
  backend: isaaclab
  convention: opengl
  offset_rot_wxyz: [0.5, 0.5, -0.5, -0.5]
  data_type: depth
  processing:
    normalize: true
    add_noise: true
    noise_type: gaussian
    noise_params:
      std: 0.01
```
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class DepthProcessingCfg:
    """Configuration for depth image preprocessing.
    
    This configures optional processing steps applied to raw depth data:
    - Clamping to valid range
    - Adding noise (Gaussian, pixel dropout)
    - Normalization to [0, 1]
    
    DEAN-compatible settings:
        normalize=True, normalize_range=(0.0, 1.0), add_noise=True,
        noise_type="gaussian", noise_params={"std": 0.01}
    """
    
    normalize: bool = False
    """Whether to normalize depth values to [0, 1] range."""
    
    normalize_range: Tuple[float, float] = (0.0, 1.0)
    """Target range for normalization (min, max)."""
    
    add_noise: bool = False
    """Whether to add noise to depth measurements."""
    
    noise_type: str = "gaussian"
    """Type of noise: 'gaussian' or 'dropout'."""
    
    noise_params: Dict[str, Any] = field(default_factory=lambda: {"std": 0.01})
    """Noise parameters. For gaussian: {'std': float}. For dropout: {'prob': float}."""
    
    clamp_min: Optional[float] = None
    """Minimum depth value (None = use sensor min)."""
    
    clamp_max: Optional[float] = None
    """Maximum depth value (None = use sensor max)."""


@dataclass
class DepthCameraCfg:
    """Configuration for depth camera sensor.
    
    Supports multiple backends:
    - "isaaclab": Uses IsaacLab's TiledCamera for GPU-accelerated multi-env rendering
    - "simple_raycaster": Uses warp-based raycasting for depth estimation
    
    Key features:
    - Configurable mount position and orientation
    - Convention switching (ROS vs OpenGL)
    - Depth data type selection (distance_to_camera, distance_to_image_plane, depth)
    - Optional preprocessing pipeline
    
    Example (ROS convention, default):
        backend="isaaclab"
        convention="ros"
        offset_rot_wxyz=[0.5, -0.5, 0.5, -0.5]  # Forward-facing
    
    Example (DEAN-compatible, OpenGL convention):
        backend="isaaclab"
        convention="opengl"
        offset_rot_wxyz=[0.5, 0.5, -0.5, -0.5]  # Forward-facing
        data_type="depth"
        processing=DepthProcessingCfg(normalize=True, add_noise=True)
    """
    
    backend: str = "isaaclab"
    """Depth sensor backend: 'isaaclab' or 'simple_raycaster'."""
    
    resolution: Tuple[int, int] = (64, 64)
    """Depth image resolution (height, width)."""
    
    range: float = 10.0
    """Maximum sensing range in meters."""
    
    offset_pos: Tuple[float, float, float] = (0.1, 0.0, 0.0)
    """Camera offset position relative to drone base (x, y, z) in meters."""
    
    offset_rot_wxyz: Tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)
    """Camera offset rotation as quaternion (w, x, y, z). 
    Default is forward-facing in ROS convention."""
    
    convention: str = "ros"
    """Coordinate convention: 'ros' or 'opengl'.
    ROS: X-forward, Y-left, Z-up
    OpenGL: X-right, Y-up, Z-backward"""
    
    focal_length: float = 24.0
    """Camera focal length in mm."""
    
    horizontal_aperture: float = 20.955
    """Camera horizontal aperture in mm."""
    
    clipping_range: Tuple[float, float] = (0.1, 1000.0)
    """Near and far clipping planes (meters)."""
    
    data_type: str = "distance_to_image_plane"
    """Depth data type: 'distance_to_camera', 'distance_to_image_plane', or 'depth'.
    - distance_to_camera: Euclidean distance from camera origin
    - distance_to_image_plane: Distance along optical axis (recommended)
    - depth: Z-buffer depth (DEAN-compatible)"""
    
    depth_clipping_behavior: str = "max"
    """How to handle rays that don't hit: 'nan' or 'max'.
    'max' sets to max range, reducing need for nan_to_num."""
    
    processing: DepthProcessingCfg = field(default_factory=DepthProcessingCfg)
    """Optional depth preprocessing configuration."""
    
    # Simple raycaster specific options
    mesh_prim_paths: List[str] = field(default_factory=lambda: ["/World/ground"])
    """Mesh prim paths for raycasting (simple_raycaster backend only)."""
    
    simplify_factor: Optional[float] = None
    """Mesh simplification factor (simple_raycaster backend only)."""
    
    mesh_poses_from_stage: bool = True
    """Whether to extract mesh poses from stage (simple_raycaster backend only)."""
    
    mesh_count: Optional[int] = None
    """Expected mesh count if not from stage (simple_raycaster backend only)."""
