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
# pyright: reportArgumentType=false

import numpy as np
import pytest

from verl.trainer.ppo.trajectory_grpo import (
    NATIVE_TRAJECTORY_GRPO_CONFIG,
    assign_trajectory_update_ids,
    broadcast_trajectory_values,
    compute_processed_trajectory_rewards,
    compute_trajectory_advantages,
    count_trajectory_invalids,
    group_rows_by_uid_traj_uid,
    make_zero_weight_padding,
    reduce_trajectory_scores,
    require_filter_target_reached,
    select_penalty_aware_group_indices,
    take_complete_uid_groups,
    validate_trajectory_grpo_config,
)


def test_native_defaults_are_noop_and_valid():
    assert NATIVE_TRAJECTORY_GRPO_CONFIG == {
        "scheduler": "row",
        "reducer": "token_mean",
        "advantage": "step_row",
        "penalty": "step_local",
        "filter": "off",
    }
    validate_trajectory_grpo_config(NATIVE_TRAJECTORY_GRPO_CONFIG)


@pytest.mark.parametrize(
    "config",
    [
        {"scheduler": "row", "reducer": "trajectory_mean"},
        {"scheduler": "row", "advantage": "trajectory"},
        {"scheduler": "row", "penalty": "trajectory"},
        {"scheduler": "row", "filter": "penalty_aware"},
    ],
)
def test_trajectory_controls_are_independent_with_row_scheduler(config):
    validate_trajectory_grpo_config(config)


def test_trajectory_packed_scheduler_is_valid_with_phase_i_runtime():
    validate_trajectory_grpo_config(
        {"scheduler": "trajectory_packed"},
        actor_strategy="fsdp",
        ppo_epochs=1,
        ulysses_sequence_parallel_size=1,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"scheduler": "invalid"}, "unsupported trajectory_grpo.scheduler"),
        ({"reducer": "invalid"}, "unsupported trajectory_grpo.reducer"),
        ({"advantage": "invalid"}, "unsupported trajectory_grpo.advantage"),
        ({"penalty": "invalid"}, "unsupported trajectory_grpo.penalty"),
        ({"filter": "invalid"}, "unsupported trajectory_grpo.filter"),
    ],
)
def test_config_rejects_unsupported_values(config, message):
    with pytest.raises(ValueError, match=message):
        validate_trajectory_grpo_config(config)


@pytest.mark.parametrize(
    "runtime",
    [
        {"actor_strategy": "megatron", "ppo_epochs": 1, "ulysses_sequence_parallel_size": 1},
        {"actor_strategy": "fsdp", "ppo_epochs": 2, "ulysses_sequence_parallel_size": 1},
        {"actor_strategy": "fsdp", "ppo_epochs": 1, "ulysses_sequence_parallel_size": 2},
    ],
)
def test_phase_i_trajectory_scheduler_runtime_limits(runtime):
    for scheduler in ("trajectory", "trajectory_packed"):
        with pytest.raises(ValueError):
            validate_trajectory_grpo_config({"scheduler": scheduler}, **runtime)

    validate_trajectory_grpo_config(
        {"scheduler": "trajectory"},
        actor_strategy="fsdp",
        ppo_epochs=1,
        ulysses_sequence_parallel_size=1,
    )
    validate_trajectory_grpo_config(
        {
            "algorithm": {
                "trajectory_grpo": {
                    "scheduler": "trajectory",
                    "reducer": "trajectory_mean",
                    "advantage": "trajectory",
                    "penalty": "trajectory",
                    "filter": "penalty_aware",
                }
            },
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "ppo_epochs": 1,
                    "ulysses_sequence_parallel_size": 1,
                }
            },
        }
    )


def test_assign_trajectory_update_ids_balances_complete_trajectories():
    trajectory_ids = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 99])
    row_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=float)

    schedule = assign_trajectory_update_ids(
        trajectory_ids,
        row_weights,
        target_rows=5,
    )

    assert schedule.update_count == 2
    assert schedule.update_row_counts == (5, 5)
    assert schedule.update_ids[-1] == -1
    for trajectory_id in (0, 1, 2, 3):
        assigned = np.unique(schedule.update_ids[trajectory_ids == trajectory_id])
        np.testing.assert_array_equal(assigned, [assigned[0]])
        assert assigned[0] >= 0


def test_assign_trajectory_update_ids_rejects_impossible_native_cadence():
    with pytest.raises(
        ValueError,
        match="cannot preserve the native optimizer cadence",
    ):
        assign_trajectory_update_ids(
            trajectory_ids=[0, 0, 0, 1, 1],
            row_weights=[1, 1, 1, 1, 1],
            target_rows=1,
        )


def test_penalty_aware_filter_rejects_reward_kl_only_for_full_config():
    full_config = {
        "algorithm": {
            "use_kl_in_reward": True,
            "trajectory_grpo": {
                "scheduler": "row",
                "reducer": "token_mean",
                "advantage": "step_row",
                "penalty": "step_local",
                "filter": "penalty_aware",
            },
        },
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp",
                "ppo_epochs": 1,
                "ulysses_sequence_parallel_size": 1,
            }
        },
    }

    with pytest.raises(ValueError, match="incompatible.*use_kl_in_reward=True"):
        validate_trajectory_grpo_config(full_config)

    full_config["algorithm"]["trajectory_grpo"]["filter"] = "off"
    validate_trajectory_grpo_config(full_config)


def test_trajectory_grouping_reduction_penalty_and_broadcast():
    uids = np.array(["a", "a", "a", "a", "b"], dtype=object)
    traj_uids = np.array(["t1", "t1", "t2", "t2", "t3"], dtype=object)
    groups = group_rows_by_uid_traj_uid(uids, traj_uids)

    assert [(group.uid, group.traj_uid, group.row_indices) for group in groups] == [
        ("a", "t1", (0, 1)),
        ("a", "t2", (2, 3)),
        ("b", "t3", (4,)),
    ]
    np.testing.assert_array_equal(
        count_trajectory_invalids(uids, traj_uids, [1, 2, 0, 1, 3]),
        [3, 1, 3],
    )
    np.testing.assert_allclose(
        reduce_trajectory_scores(uids, traj_uids, [1, 3, 4, 8, 10]),
        [2, 6, 10],
    )
    np.testing.assert_allclose(
        compute_processed_trajectory_rewards(
            uids,
            traj_uids,
            [1, 3, 4, 8, 10],
            row_invalid_counts=[1, 2, 0, 1, 3],
            invalid_action_penalty_coef=0.5,
        ),
        [0.5, 5.5, 8.5],
    )
    np.testing.assert_array_equal(
        broadcast_trajectory_values(uids, traj_uids, [20, 60, 100]),
        [20, 20, 60, 60, 100],
    )


def test_trajectory_penalty_and_fanout_ignore_zero_weight_padding():
    uids = ["a", "a", "a", "a", "a"]
    traj_uids = ["t1", "t1", "t2", "t2", "t1"]
    row_weights = [1, 1, 1, 1, 0]
    processed = compute_processed_trajectory_rewards(
        uids,
        traj_uids,
        [1, 3, 4, 8, 1000],
        row_invalid_counts=[1, 0, 0, 1, 1000],
        invalid_action_penalty_coef=0.5,
        row_weights=row_weights,
    )
    np.testing.assert_allclose(processed, [1.5, 5.5])

    trajectory_advantages = compute_trajectory_advantages(
        ["a", "a"],
        processed,
        normalize_by_std=False,
    )
    np.testing.assert_allclose(trajectory_advantages, [-2.0, 2.0])
    np.testing.assert_allclose(
        broadcast_trajectory_values(uids, traj_uids, trajectory_advantages),
        [-2.0, -2.0, 2.0, 2.0, -2.0],
    )


def test_trajectory_advantage_is_grouped_by_uid():
    advantages = compute_trajectory_advantages(
        ["a", "a", "b"],
        [2.0, 6.0, 10.0],
        normalize_by_std=False,
    )
    np.testing.assert_allclose(advantages, [-2.0, 2.0, 0.0])

    normalized = compute_trajectory_advantages(["a", "a"], [2.0, 6.0])
    np.testing.assert_allclose(
        normalized,
        [-0.70710678, 0.70710678],
        atol=1e-5,
    )


def test_trajectory_advantage_matches_three_success_five_failure_reference():
    normalized = compute_trajectory_advantages(
        ["a"] * 8,
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        normalized[:3],
        [1.207612] * 3,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        normalized[3:],
        [-0.724567] * 5,
        atol=1e-5,
    )


def test_zero_weight_padding():
    padding = make_zero_weight_padding([4], 4)
    np.testing.assert_array_equal(padding.indices, [4, 4, 4, 4])
    np.testing.assert_array_equal(padding.weights, [1.0, 0.0, 0.0, 0.0])


def test_penalty_aware_filter_keeps_complete_nonconstant_uid_groups():
    keep_indices, processed_rewards = select_penalty_aware_group_indices(
        ["a", "a", "b", "b"],
        [1.0, 1.0, 0.0, 1.0],
        [0, 0, 0, 2],
        invalid_action_penalty_coef=0.25,
    )
    np.testing.assert_array_equal(keep_indices, [2, 3])
    np.testing.assert_allclose(processed_rewards, [1.0, 1.0, 0.0, 0.5])


def test_penalty_aware_filter_replenishment_failure_is_explicit():
    np.testing.assert_array_equal(
        take_complete_uid_groups(["a", "a", "b", "b"], 2),
        [0, 1],
    )
    require_filter_target_reached(4, 4, 3)
    with pytest.raises(
        RuntimeError,
        match=r"exhausted 3 generation batches with 2/4 accepted trajectories",
    ):
        require_filter_target_reached(2, 4, 3)
