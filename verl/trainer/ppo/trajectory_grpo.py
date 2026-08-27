# Copyright 2026 The verl-agent team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: UP006, UP045
"""Pure helpers shared by trajectory-aware GRPO integration points.

The helpers in this module deliberately avoid Ray, model, and device state.
They accept plain Python sequences or NumPy arrays and return deterministic
metadata that callers can apply to their own batch containers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import DTypeLike

NATIVE_TRAJECTORY_GRPO_CONFIG = {
    "scheduler": "row",
    "reducer": "token_mean",
    "advantage": "step_row",
    "penalty": "step_local",
    "filter": "off",
}


@dataclass(frozen=True)
class TrajectoryGroup:
    """Rows belonging to one ``(uid, traj_uid)`` trajectory."""

    uid: Hashable
    traj_uid: Hashable
    row_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ZeroWeightPadding:
    """Indices and weights for padding an update without adding loss."""

    indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class TrajectoryUpdateSchedule:
    """Complete-trajectory assignment to dynamic optimizer updates."""

    update_ids: np.ndarray
    update_count: int
    update_row_counts: Tuple[int, ...]


def _as_1d(name: str, values: Sequence[Any], *, dtype: Optional[DTypeLike] = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _require_same_length(**arrays: np.ndarray) -> int:
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"inputs must have equal lengths, got {lengths}")
    return next(iter(lengths.values()), 0)


def _trajectory_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "algorithm" in config:
        algorithm = config["algorithm"]
        if not isinstance(algorithm, Mapping):
            raise ValueError("algorithm config must be a mapping")
        config = algorithm.get("trajectory_grpo", {})
    elif "trajectory_grpo" in config:
        config = config["trajectory_grpo"]
    if not isinstance(config, Mapping):
        raise ValueError("trajectory_grpo config must be a mapping")
    return config


def validate_trajectory_grpo_config(
    config: Mapping[str, Any],
    *,
    actor_strategy: Optional[str] = None,
    ppo_epochs: Optional[int] = None,
    ulysses_sequence_parallel_size: Optional[int] = None,
) -> None:
    """Validate trajectory-GRPO dependencies and phase-I runtime limits.

    ``config`` may be the trajectory-GRPO block, a mapping containing a
    ``trajectory_grpo`` block, or the complete trainer config. Runtime values
    can be supplied explicitly; when omitted they are read from the complete
    trainer config if available.
    """

    trajectory_config = _trajectory_config(config)
    scheduler = trajectory_config.get("scheduler", NATIVE_TRAJECTORY_GRPO_CONFIG["scheduler"])
    reducer = trajectory_config.get("reducer", NATIVE_TRAJECTORY_GRPO_CONFIG["reducer"])
    advantage = trajectory_config.get("advantage", NATIVE_TRAJECTORY_GRPO_CONFIG["advantage"])
    penalty = trajectory_config.get("penalty", NATIVE_TRAJECTORY_GRPO_CONFIG["penalty"])
    filter_mode = trajectory_config.get("filter", NATIVE_TRAJECTORY_GRPO_CONFIG["filter"])

    allowed_values = {
        "scheduler": {"row", "trajectory", "trajectory_packed"},
        "advantage": {"step_row", "trajectory"},
        "penalty": {"step_local", "trajectory"},
        "filter": {"off", "penalty_aware"},
    }
    values = {
        "scheduler": scheduler,
        "advantage": advantage,
        "penalty": penalty,
        "filter": str(filter_mode).replace("-", "_"),
    }
    if reducer not in {"token_mean", "trajectory_mean"}:
        raise ValueError(f"unsupported trajectory_grpo.reducer: {reducer}")
    for name, value in values.items():
        if value not in allowed_values[name]:
            raise ValueError(f"unsupported trajectory_grpo.{name}: {value}")

    if (
        "algorithm" in config
        and values["filter"] == "penalty_aware"
        and bool(config["algorithm"].get("use_kl_in_reward", False))
    ):
        raise ValueError(
            "penalty-aware filter is computed during rollout and is incompatible "
            "with algorithm.use_kl_in_reward=True"
        )

    if scheduler not in {"trajectory", "trajectory_packed"}:
        return

    actor = config.get("actor_rollout_ref", {}).get("actor", {}) if "algorithm" in config else {}
    if actor_strategy is None:
        actor_strategy = actor.get("strategy")
    if ppo_epochs is None:
        ppo_epochs = actor.get("ppo_epochs")
    if ulysses_sequence_parallel_size is None:
        ulysses_sequence_parallel_size = actor.get("ulysses_sequence_parallel_size")

    if actor_strategy is None or not str(actor_strategy).lower().startswith("fsdp"):
        raise ValueError("phase-I trajectory schedulers require an FSDP actor strategy")
    if ppo_epochs != 1:
        raise ValueError("phase-I trajectory schedulers require ppo_epochs=1")
    if ulysses_sequence_parallel_size != 1:
        raise ValueError(
            "phase-I trajectory schedulers require ulysses_sequence_parallel_size=1"
        )


def assign_trajectory_update_ids(
    trajectory_ids: Sequence[int],
    row_weights: Sequence[float],
    *,
    target_rows: int,
) -> TrajectoryUpdateSchedule:
    """Pack complete trajectories into a dynamic number of balanced updates.

    The update count exactly follows the native row scheduler,
    ``ceil(real_rows / target_rows)``. If there are fewer real trajectories
    than requested updates, the two contracts cannot both be satisfied and the
    scheduler fails instead of silently reducing optimizer cadence. Synthetic
    zero-weight rows receive update id ``-1``.
    """

    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    trajectory_array = _as_1d("trajectory_ids", trajectory_ids)
    if trajectory_array.dtype == np.bool_ or not np.issubdtype(
        trajectory_array.dtype, np.integer
    ):
        raise ValueError("trajectory_ids must contain integers")
    weights = _as_1d("row_weights", row_weights, dtype=np.float64)
    _require_same_length(trajectory_ids=trajectory_array, row_weights=weights)
    if np.any(weights < 0):
        raise ValueError("row_weights must be non-negative")

    real_mask = weights > 0
    real_row_count = int(real_mask.sum())
    if real_row_count == 0:
        raise ValueError("trajectory packing requires at least one real row")

    grouped: Dict[int, List[int]] = {}
    first_seen: Dict[int, int] = {}
    for row_index in np.flatnonzero(real_mask):
        trajectory_id = int(trajectory_array[row_index])
        grouped.setdefault(trajectory_id, []).append(int(row_index))
        first_seen.setdefault(trajectory_id, int(row_index))

    update_count = math.ceil(real_row_count / target_rows)
    if update_count > len(grouped):
        raise ValueError(
            "trajectory packing cannot preserve the native optimizer cadence: "
            f"{update_count} updates requested for {len(grouped)} trajectories"
        )
    update_row_counts = [0] * update_count
    trajectory_to_update: Dict[int, int] = {}
    ordered_trajectories = sorted(
        grouped,
        key=lambda trajectory_id: (
            -len(grouped[trajectory_id]),
            first_seen[trajectory_id],
        ),
    )
    for trajectory_id in ordered_trajectories:
        update_id = min(
            range(update_count),
            key=lambda candidate: (update_row_counts[candidate], candidate),
        )
        trajectory_to_update[trajectory_id] = update_id
        update_row_counts[update_id] += len(grouped[trajectory_id])

    update_ids = np.full(len(trajectory_array), -1, dtype=np.int64)
    for trajectory_id, row_indices in grouped.items():
        update_ids[np.asarray(row_indices, dtype=np.int64)] = trajectory_to_update[
            trajectory_id
        ]

    return TrajectoryUpdateSchedule(
        update_ids=update_ids,
        update_count=update_count,
        update_row_counts=tuple(update_row_counts),
    )


def group_rows_by_uid_traj_uid(
    uids: Sequence[Hashable],
    traj_uids: Sequence[Hashable],
) -> Tuple[TrajectoryGroup, ...]:
    """Group rows by ``(uid, traj_uid)`` while preserving first-seen order."""

    uid_array = _as_1d("uids", uids, dtype=object)
    traj_uid_array = _as_1d("traj_uids", traj_uids, dtype=object)
    _require_same_length(uids=uid_array, traj_uids=traj_uid_array)

    grouped = {}
    for row_index, key in enumerate(zip(uid_array.tolist(), traj_uid_array.tolist())):
        grouped.setdefault(key, []).append(row_index)
    return tuple(
        TrajectoryGroup(uid=uid, traj_uid=traj_uid, row_indices=tuple(indices))
        for (uid, traj_uid), indices in grouped.items()
    )


def count_trajectory_invalids(
    uids: Sequence[Hashable],
    traj_uids: Sequence[Hashable],
    row_invalid_counts: Sequence[float],
    *,
    row_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Sum invalid-action counts for every trajectory in first-seen order."""

    invalid_counts = _as_1d("row_invalid_counts", row_invalid_counts, dtype=np.float64)
    groups = group_rows_by_uid_traj_uid(uids, traj_uids)
    if sum(len(group.row_indices) for group in groups) != len(invalid_counts):
        raise ValueError("row_invalid_counts must match uids and traj_uids")
    weights = (
        np.ones(len(invalid_counts), dtype=np.float64)
        if row_weights is None
        else _as_1d("row_weights", row_weights, dtype=np.float64)
    )
    _require_same_length(row_invalid_counts=invalid_counts, row_weights=weights)
    return np.asarray(
        [
            (invalid_counts[group_indices] * weights[group_indices]).sum()
            for group in groups
            for group_indices in [np.asarray(group.row_indices, dtype=np.int64)]
        ],
        dtype=np.float64,
    )


def reduce_trajectory_scores(
    uids: Sequence[Hashable],
    traj_uids: Sequence[Hashable],
    row_scores: Sequence[float],
    *,
    reducer: str = "mean_row",
    row_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Reduce row scores to one score per trajectory."""

    if reducer != "mean_row":
        raise ValueError(f"unsupported trajectory score reducer: {reducer}")
    scores = _as_1d("row_scores", row_scores, dtype=np.float64)
    groups = group_rows_by_uid_traj_uid(uids, traj_uids)
    if sum(len(group.row_indices) for group in groups) != len(scores):
        raise ValueError("row_scores must match uids and traj_uids")
    weights = (
        np.ones(len(scores), dtype=np.float64)
        if row_weights is None
        else _as_1d("row_weights", row_weights, dtype=np.float64)
    )
    _require_same_length(row_scores=scores, row_weights=weights)
    reduced = []
    for group in groups:
        group_indices = np.asarray(group.row_indices, dtype=np.int64)
        group_weights = weights[group_indices]
        denominator = group_weights.sum()
        if denominator <= 0:
            raise ValueError(f"trajectory {group.traj_uid!r} has no real rows")
        reduced.append((scores[group_indices] * group_weights).sum() / denominator)
    return np.asarray(reduced, dtype=np.float64)


def compute_processed_trajectory_rewards(
    uids: Sequence[Hashable],
    traj_uids: Sequence[Hashable],
    row_rewards: Sequence[float],
    *,
    row_invalid_counts: Optional[Sequence[float]] = None,
    invalid_action_penalty_coef: float = 0.0,
    reducer: str = "mean_row",
    row_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return trajectory rewards after optional trajectory-level penalties."""

    rewards = reduce_trajectory_scores(
        uids,
        traj_uids,
        row_rewards,
        reducer=reducer,
        row_weights=row_weights,
    )
    if row_invalid_counts is None:
        return rewards
    invalid_counts = count_trajectory_invalids(
        uids,
        traj_uids,
        row_invalid_counts,
        row_weights=row_weights,
    )
    return rewards - float(invalid_action_penalty_coef) * invalid_counts


def select_penalty_aware_group_indices(
    uids: Sequence[Hashable],
    episode_rewards: Sequence[float],
    trajectory_invalid_counts: Sequence[float],
    *,
    invalid_action_penalty_coef: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep complete uid groups with non-constant processed trajectory rewards."""

    uid_array = _as_1d("uids", uids, dtype=object)
    rewards = _as_1d("episode_rewards", episode_rewards, dtype=np.float64)
    invalid_counts = _as_1d(
        "trajectory_invalid_counts",
        trajectory_invalid_counts,
        dtype=np.float64,
    )
    _require_same_length(
        uids=uid_array,
        episode_rewards=rewards,
        trajectory_invalid_counts=invalid_counts,
    )
    processed_rewards = rewards - float(invalid_action_penalty_coef) * invalid_counts

    grouped = {}
    for index, uid in enumerate(uid_array.tolist()):
        grouped.setdefault(uid, []).append(index)

    keep_indices = []
    for indices in grouped.values():
        group_indices = np.asarray(indices, dtype=np.int64)
        group_rewards = processed_rewards[group_indices]
        if np.ptp(group_rewards) > 0:
            keep_indices.extend(indices)
    return np.asarray(keep_indices, dtype=np.int64), processed_rewards


def require_filter_target_reached(
    collected_trajectories: int,
    target_trajectories: int,
    max_attempts: int,
) -> None:
    """Fail deterministically when explicit filtering cannot fill the target."""

    if collected_trajectories < target_trajectories:
        raise RuntimeError(
            "penalty-aware trajectory filtering exhausted "
            f"{max_attempts} generation batches with "
            f"{collected_trajectories}/{target_trajectories} accepted trajectories"
        )


def take_complete_uid_groups(
    uids: Sequence[Hashable],
    max_trajectories: int,
) -> np.ndarray:
    """Select first-seen complete uid groups without exceeding a target."""

    if max_trajectories < 0:
        raise ValueError("max_trajectories must be non-negative")
    uid_array = _as_1d("uids", uids, dtype=object)
    grouped = {}
    for index, uid in enumerate(uid_array.tolist()):
        grouped.setdefault(uid, []).append(index)

    selected = []
    for indices in grouped.values():
        if len(selected) + len(indices) > max_trajectories:
            break
        selected.extend(indices)
    return np.asarray(selected, dtype=np.int64)


def compute_trajectory_advantages(
    uids: Sequence[Hashable],
    trajectory_scores: Sequence[float],
    *,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute GRPO advantages over trajectory scores grouped by ``uid``.

    Inputs and outputs contain one value per trajectory. A singleton uid group
    receives zero advantage. Sample standard deviation (``ddof=1``) matches
    the native ``torch.std`` normalization used by GRPO.
    """

    uid_array = _as_1d("uids", uids, dtype=object)
    scores = _as_1d("trajectory_scores", trajectory_scores, dtype=np.float64)
    _require_same_length(uids=uid_array, trajectory_scores=scores)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    advantages = np.zeros_like(scores)
    grouped = {}
    for index, uid in enumerate(uid_array.tolist()):
        grouped.setdefault(uid, []).append(index)
    for indices in grouped.values():
        group_indices = np.asarray(indices, dtype=np.int64)
        group_scores = scores[group_indices]
        centered = group_scores - group_scores.mean()
        if normalize_by_std and len(group_indices) > 1:
            centered = centered / (group_scores.std(ddof=1) + epsilon)
        advantages[group_indices] = centered
    return advantages


def broadcast_trajectory_values(
    uids: Sequence[Hashable],
    traj_uids: Sequence[Hashable],
    trajectory_values: Sequence[float],
) -> np.ndarray:
    """Broadcast first-seen trajectory values back to their source rows."""

    groups = group_rows_by_uid_traj_uid(uids, traj_uids)
    values = _as_1d("trajectory_values", trajectory_values)
    if len(groups) != len(values):
        raise ValueError(f"expected {len(groups)} trajectory values, got {len(values)}")
    row_values = np.empty(sum(len(group.row_indices) for group in groups), dtype=values.dtype)
    for group, value in zip(groups, values):
        row_values[np.asarray(group.row_indices, dtype=np.int64)] = value
    return row_values


def make_zero_weight_padding(
    real_indices: Sequence[int],
    target_size: int,
) -> ZeroWeightPadding:
    """Pad indices to ``target_size`` by cycling real rows with zero weights."""

    indices = _as_1d("real_indices", real_indices, dtype=np.int64)
    num_real = len(indices)
    if target_size < num_real:
        raise ValueError(f"target_size={target_size} is smaller than {num_real} real rows")
    if target_size > 0 and num_real == 0:
        raise ValueError("cannot pad an empty update to a non-zero target size")

    num_padding = target_size - num_real
    if num_padding:
        padding_indices = np.resize(indices, num_padding)
        padded_indices = np.concatenate((indices, padding_indices))
    else:
        padded_indices = indices.copy()
    weights = np.concatenate(
        (np.ones(num_real, dtype=np.float64), np.zeros(num_padding, dtype=np.float64))
    )
    return ZeroWeightPadding(
        indices=padded_indices,
        weights=weights,
    )


__all__ = [
    "NATIVE_TRAJECTORY_GRPO_CONFIG",
    "TrajectoryGroup",
    "TrajectoryUpdateSchedule",
    "ZeroWeightPadding",
    "assign_trajectory_update_ids",
    "broadcast_trajectory_values",
    "compute_processed_trajectory_rewards",
    "compute_trajectory_advantages",
    "count_trajectory_invalids",
    "group_rows_by_uid_traj_uid",
    "make_zero_weight_padding",
    "require_filter_target_reached",
    "reduce_trajectory_scores",
    "select_penalty_aware_group_indices",
    "take_complete_uid_groups",
    "validate_trajectory_grpo_config",
]
