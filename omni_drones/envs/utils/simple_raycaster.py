# MIT License
#
# Copyright (c) 2026
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

from __future__ import annotations

import inspect
import re
from typing import Iterable, Sequence, Tuple

import torch

_REGEX_CHARS = set(".*+?[](){}|^$")


def _is_regex(pattern: str) -> bool:
    return any(ch in pattern for ch in _REGEX_CHARS)


def _match_path(path: str, pattern: str) -> bool:
    if _is_regex(pattern):
        return re.fullmatch(pattern, path) is not None
    return path == pattern


def select_mesh_prims(patterns: Sequence[str]):
    from pxr import UsdGeom
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    prims = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = prim.GetPath().pathString
        if any(_match_path(path, p) for p in patterns):
            prims.append(prim)
    return prims


def compute_mesh_poses(prims) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    from pxr import Usd, UsdGeom

    if not prims:
        return None, None
    positions = []
    quats = []
    for prim in prims:
        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        trans = mat.ExtractTranslation()
        rot = mat.ExtractRotationQuat()
        positions.append([trans[0], trans[1], trans[2]])
        imag = rot.GetImaginary()
        quats.append([rot.GetReal(), imag[0], imag[1], imag[2]])
    return (
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(quats, dtype=torch.float32),
    )


def build_mesh_pose_tensors(
    num_envs: int,
    mesh_pos: torch.Tensor | None,
    mesh_quat: torch.Tensor | None,
    device: torch.device,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if mesh_pos is None or mesh_quat is None:
        return None, None
    mesh_pos_w = mesh_pos.to(device).unsqueeze(0).expand(num_envs, -1, -1).contiguous()
    mesh_quat_w = mesh_quat.to(device).unsqueeze(0).expand(num_envs, -1, -1).contiguous()
    return mesh_pos_w, mesh_quat_w


def default_mesh_pose_tensors(
    num_envs: int,
    mesh_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mesh_pos_w = torch.zeros(num_envs, mesh_count, 3, device=device)
    mesh_quat_w = torch.zeros(num_envs, mesh_count, 4, device=device)
    mesh_quat_w[..., 0] = 1.0
    return mesh_pos_w, mesh_quat_w


def _normalize_warp_device(device):
    if device is None:
        return None
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return "cuda"
        if device.type == "cpu":
            return "cpu"
        return str(device)
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        if device.startswith("cpu"):
            return "cpu"
    return device


def create_raycaster_from_stage(
    raycaster_cls,
    stage,
    paths: Sequence[str],
    device=None,
    simplify_factor=None,
):
    sig = inspect.signature(raycaster_cls.from_prim_paths)
    kwargs = {}
    if "stage" in sig.parameters:
        kwargs["stage"] = stage
    if "paths" in sig.parameters:
        kwargs["paths"] = paths
    elif "prim_paths" in sig.parameters:
        kwargs["prim_paths"] = paths
    elif "prim_path" in sig.parameters and len(paths) == 1:
        kwargs["prim_path"] = paths[0]
    if device is not None and "device" in sig.parameters:
        kwargs["device"] = _normalize_warp_device(device)
    if simplify_factor is not None and "simplify_factor" in sig.parameters:
        kwargs["simplify_factor"] = simplify_factor
    return raycaster_cls.from_prim_paths(**kwargs)


def raycast_fused(
    raycaster,
    ray_starts_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    min_dist: float,
    max_dist: float,
    mesh_pos_w: torch.Tensor | None = None,
    mesh_quat_w: torch.Tensor | None = None,
):
    sig = inspect.signature(raycaster.raycast_fused)
    kwargs = {
        "ray_starts_w": ray_starts_w,
        "ray_dirs_w": ray_dirs_w,
        "min_dist": min_dist,
        "max_dist": max_dist,
    }
    if "mesh_pos_w" in sig.parameters:
        if mesh_pos_w is None or mesh_quat_w is None:
            raise RuntimeError("mesh_pos_w and mesh_quat_w are required for this raycaster.")
        kwargs["mesh_pos_w"] = mesh_pos_w
    if "mesh_quat_w" in sig.parameters:
        if mesh_quat_w is None:
            raise RuntimeError("mesh_quat_w is required for this raycaster.")
        kwargs["mesh_quat_w"] = mesh_quat_w
    return raycaster.raycast_fused(**kwargs)
