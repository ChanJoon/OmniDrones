"""Curriculum learning manager for adaptive terrain difficulty.

Tracks per-environment performance and uses IsaacLab's terrain APIs to move
environments between terrain rows of increasing difficulty.

Row convention: all levels use terrain row indices (e.g. rows 1-5 for a
7-row grid where rows 0 and 6 are guard rows).
"""

import torch
from typing import Tuple, Optional
from tensordict import TensorDict


class CurriculumManager:
    """Manages terrain-based curriculum learning via per-environment performance tracking.

    Each environment independently tracks its success/collision history over a sliding window.
    High performers move to harder terrain rows; struggling agents move to easier rows.

    Row convention:
        ``current_levels`` stores **terrain row indices** (e.g. 1-5), matching
        ``env_terrain_levels`` and ``terrain.terrain_levels`` directly.  Guard rows
        (0 and 6) are excluded via ``min_terrain_row`` / ``max_terrain_row``.

    Args:
        num_envs: Number of parallel environments
        device: Torch device (e.g., "cuda:0")
        window_size: Number of recent episodes to track per environment (default: 20)
        success_threshold_up: Success rate threshold to move to harder terrain (default: 0.75)
        success_threshold_down: Success rate threshold to move to easier terrain (default: 0.25)
        collision_threshold: Collision rate threshold to move down (default: 0.60)
        cooldown_episodes: Minimum episodes between level changes per env (default: 30)
        num_terrain_levels: Number of terrain difficulty levels (default: 5)
        min_terrain_row: Lowest curriculum terrain row index (default: 1)
        max_terrain_row: Highest curriculum terrain row index (default: 5)
        success_metric: Which success signal to use for promotion/demotion.
            One of "hover_success", "success", "goal_success".
    """

    SUCCESS_KEY_MAP = {
        "hover_success": "reset_hover_success",   # hover termination signal
        "success": "success",                       # goal_reached based (latched)
        "goal_success": "reset_goal_success",       # goal termination signal
    }

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        window_size: int = 20,
        success_threshold_up: float = 0.75,
        success_threshold_down: float = 0.25,
        collision_threshold: float = 0.60,
        cooldown_episodes: int = 30,
        num_terrain_levels: int = 5,
        min_terrain_row: int = 1,
        max_terrain_row: int = 5,
        success_metric: str = "hover_success",
    ):
        if success_metric not in self.SUCCESS_KEY_MAP:
            raise ValueError(
                f"Unknown success_metric '{success_metric}'. "
                f"Options: {list(self.SUCCESS_KEY_MAP.keys())}"
            )

        self.num_envs = num_envs
        self.device = device
        self.window_size = window_size
        self.success_threshold_up = success_threshold_up
        self.success_threshold_down = success_threshold_down
        self.collision_threshold = collision_threshold
        self.cooldown_episodes = cooldown_episodes
        self.num_terrain_levels = num_terrain_levels
        self.min_terrain_row = min_terrain_row
        self.max_terrain_row = max_terrain_row
        self.success_metric = success_metric
        self.success_stats_key = self.SUCCESS_KEY_MAP[success_metric]

        # Per-environment sliding window tracking
        # Shape: (num_envs, window_size)
        self.success_history = torch.zeros(num_envs, window_size, dtype=torch.bool, device=device)
        self.collision_history = torch.zeros(num_envs, window_size, dtype=torch.bool, device=device)

        # Current position in sliding window per environment
        self.history_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Track total episodes per environment (monotonic, for cooldown)
        self.episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Track episodes since last window reset (for accurate window_filled detection)
        self.window_episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Track last level change episode per environment
        self.last_level_change = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Current terrain row per environment (uses row indices, e.g. 1-5)
        self.current_levels = torch.full(
            (num_envs,), min_terrain_row, dtype=torch.long, device=device
        )

        # Track whether window is full (for accurate statistics)
        self.window_filled = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update_metrics(self, episode_stats: TensorDict) -> None:
        """Update performance tracking from completed episodes.

        Args:
            episode_stats: TensorDict containing episode statistics. Looks up
                the success key from ``self.success_stats_key`` and collision
                from ``"collision"`` or ``"reset_collision"``.
        """
        # Resolve success signal
        key = self.success_stats_key
        if key in episode_stats.keys():
            success = episode_stats[key].squeeze(-1).bool()
        elif "stats" in episode_stats.keys():
            success = episode_stats["stats"][key].squeeze(-1).bool()
        else:
            raise RuntimeError(
                f"Cannot find success key '{key}' in episode_stats "
                f"(available: {list(episode_stats.keys())})"
            )

        # Resolve collision signal
        if "collision" in episode_stats.keys():
            collision = episode_stats["collision"].squeeze(-1).bool()
        elif "stats" in episode_stats.keys() and "collision" in episode_stats["stats"].keys():
            collision = episode_stats["stats"]["collision"].squeeze(-1).bool()
        elif "stats" in episode_stats.keys() and "reset_collision" in episode_stats["stats"].keys():
            collision = episode_stats["stats"]["reset_collision"].squeeze(-1).bool()
        else:
            collision = torch.zeros(success.shape, dtype=torch.bool, device=self.device)

        # Resolve env_ids
        if "env_id" in episode_stats.keys():
            env_ids = episode_stats["env_id"].squeeze(-1).to(self.device)
        elif "stats" in episode_stats.keys() and "env_id" in episode_stats["stats"].keys():
            env_ids = episode_stats["stats"]["env_id"].squeeze(-1).to(self.device)
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
            if success.numel() != self.num_envs:
                raise RuntimeError(
                    f"Episode stats missing env_id (got {success.numel()} episodes, "
                    f"expected {self.num_envs})."
                )

        env_ids = env_ids.reshape(-1).to(self.device)
        success = success.reshape(-1).to(self.device)
        collision = collision.reshape(-1).to(self.device)
        if env_ids.numel() != success.numel():
            raise RuntimeError(
                f"env_id count ({env_ids.numel()}) does not match "
                f"success count ({success.numel()})"
            )

        # Vectorized sliding window update (safe: each env finishes at most once per step)
        idxs = self.history_idx[env_ids]
        self.success_history[env_ids, idxs] = success
        self.collision_history[env_ids, idxs] = collision
        self.history_idx[env_ids] = (idxs + 1) % self.window_size
        self.episode_count[env_ids] += 1
        self.window_episode_count[env_ids] += 1

        # Mark window as filled after window_size episodes since last reset
        self.window_filled |= (self.window_episode_count >= self.window_size)

    def get_terrain_updates(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determine which environments should move to harder/easier terrain.

        Returns:
            move_up: Boolean tensor (num_envs,) indicating envs promoted
            move_down: Boolean tensor (num_envs,) indicating envs demoted
            new_levels: Long tensor (num_envs,) with authoritative terrain row
                for every env after this update.  Callers should set terrain
                state from ``new_levels`` rather than applying move_up/move_down
                incrementally (supports arbitrary jumps for randomize-at-max).
        """
        # Compute valid episodes using window_episode_count (not episode_count)
        valid_episodes = torch.where(
            self.window_filled,
            torch.full((self.num_envs,), self.window_size, device=self.device),
            self.window_episode_count.clamp(max=self.window_size),
        )

        # Success rate and collision rate
        success_rate = (
            self.success_history.sum(dim=1).float()
            / valid_episodes.clamp(min=1).float()
        )
        collision_rate = (
            self.collision_history.sum(dim=1).float()
            / valid_episodes.clamp(min=1).float()
        )

        # Cooldown check
        cooldown_satisfied = (
            (self.episode_count - self.last_level_change) >= self.cooldown_episodes
        )

        # Sufficient data check
        sufficient_data = valid_episodes >= (self.window_size // 2)

        # Move up: high success, not at max, cooldown ok, enough data
        move_up = (
            (success_rate >= self.success_threshold_up)
            & (self.current_levels < self.max_terrain_row)
            & cooldown_satisfied
            & sufficient_data
        )

        # Move down: low success OR high collision, not at min, cooldown ok, enough data
        move_down = (
            (
                (success_rate <= self.success_threshold_down)
                | (collision_rate >= self.collision_threshold)
            )
            & (self.current_levels > self.min_terrain_row)
            & cooldown_satisfied
            & sufficient_data
        )

        # Apply level changes
        self.current_levels[move_up] += 1
        self.current_levels[move_down] -= 1

        # Randomize-at-max: envs that reached max_terrain_row get randomized back
        at_max = self.current_levels >= self.max_terrain_row
        if at_max.any():
            self.current_levels[at_max] = torch.randint(
                self.min_terrain_row,
                self.max_terrain_row,  # exclusive upper bound → rows [min, max-1]
                (at_max.sum(),),
                device=self.device,
            )

        # Track moved envs (including randomized ones)
        moved_envs = move_up | move_down
        self.last_level_change[moved_envs] = self.episode_count[moved_envs]

        # Window reset for moved envs (clear stale statistics from old level)
        self.success_history[moved_envs] = False
        self.collision_history[moved_envs] = False
        self.history_idx[moved_envs] = 0
        self.window_filled[moved_envs] = False
        self.window_episode_count[moved_envs] = 0
        # Note: episode_count and last_level_change are NOT reset (cooldown stays intact)

        return move_up, move_down, self.current_levels.clone()

    def get_statistics(self) -> dict:
        """Get current curriculum statistics for logging.

        Returns:
            Dictionary with curriculum metrics. Level distribution uses
            0-indexed labels (level_0 = min_terrain_row, etc.) for WandB
            consistency.
        """
        mean_level = self.current_levels.float().mean().item()

        # Level distribution: count envs at each terrain row
        level_counts = []
        for row in range(self.min_terrain_row, self.max_terrain_row + 1):
            level_counts.append((self.current_levels == row).sum().item())

        return {
            "mean_terrain_level": mean_level,
            "level_distribution": level_counts,
        }

    def reset(self) -> None:
        """Reset all tracking state (for new training run)."""
        self.success_history.zero_()
        self.collision_history.zero_()
        self.history_idx.zero_()
        self.episode_count.zero_()
        self.window_episode_count.zero_()
        self.last_level_change.zero_()
        self.current_levels.fill_(self.min_terrain_row)
        self.window_filled.zero_()

    def state_dict(self) -> dict:
        """Get curriculum state for checkpoint saving."""
        return {
            "success_history": self.success_history.cpu(),
            "collision_history": self.collision_history.cpu(),
            "history_idx": self.history_idx.cpu(),
            "episode_count": self.episode_count.cpu(),
            "window_episode_count": self.window_episode_count.cpu(),
            "last_level_change": self.last_level_change.cpu(),
            "current_levels": self.current_levels.cpu(),
            "window_filled": self.window_filled.cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load curriculum state from checkpoint."""
        self.success_history = state_dict["success_history"].to(self.device)
        self.collision_history = state_dict["collision_history"].to(self.device)
        self.history_idx = state_dict["history_idx"].to(self.device)
        self.episode_count = state_dict["episode_count"].to(self.device)
        self.last_level_change = state_dict["last_level_change"].to(self.device)
        self.current_levels = state_dict["current_levels"].to(self.device)
        self.window_filled = state_dict["window_filled"].to(self.device)
        # Backward-compatible: window_episode_count may be missing in old checkpoints
        if "window_episode_count" in state_dict:
            self.window_episode_count = state_dict["window_episode_count"].to(self.device)
        else:
            # Approximate from episode_count (best effort for old checkpoints)
            self.window_episode_count = self.episode_count.clone()
