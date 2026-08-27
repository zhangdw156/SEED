# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import math
from collections.abc import Hashable, Sequence
from typing import Dict, List, Literal, cast, overload

import numpy as np
import torch
from PIL import Image

from verl import DataProto
from verl.trainer.ppo.trajectory_grpo import (
    group_rows_by_uid_traj_uid,
    make_zero_weight_padding,
    select_penalty_aware_group_indices,
)


def _trajectory_grpo_value(config, name, default):
    trajectory_config = config.algorithm.get("trajectory_grpo", {})
    return trajectory_config.get(name, default)


def _needs_trajectory_row_metadata(config) -> bool:
    return any(
        (
            str(config.algorithm.get("adv_estimator", "")).lower()
            == "cpad",
            _trajectory_grpo_value(config, "scheduler", "row")
            in {"trajectory", "trajectory_packed"},
            _trajectory_grpo_value(config, "reducer", "token_mean") == "trajectory_mean",
            _trajectory_grpo_value(config, "advantage", "step_row") == "trajectory",
            _trajectory_grpo_value(config, "penalty", "step_local") == "trajectory",
        )
    )


def _trajectory_invalid_counts(batch_list: List[List[Dict]]) -> np.ndarray:
    counts = []
    for trajectory in batch_list:
        counts.append(
            sum(
                not bool(row.get("is_action_valid", True))
                for row in trajectory
                if bool(row.get("active_masks", True))
            )
        )
    return np.asarray(counts, dtype=np.float64)

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(
    config,
    data: DataProto,
    mode="copy",
    canonical_loss_plan_required: bool = False,
    track_source_indices: bool = False,
) -> DataProto:
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    distillation = config.get("distillation")
    pure_distillation = bool(
        distillation is not None and distillation.get("enabled", False)
    )
    actor_config = config.actor_rollout_ref.actor
    actor_micro_batch_size = actor_config.get(
        "ppo_micro_batch_size_per_gpu",
        None,
    )
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = actor_config.ppo_mini_batch_size
    elif pure_distillation and (
        actor_config.get("use_dynamic_bsz", False)
        or actor_micro_batch_size is None
    ):
        size_divisor_actor = world_size
    else:
        size_divisor_actor = (
            actor_micro_batch_size * world_size
        )

    if pure_distillation:
        size_divisor = size_divisor_actor
        if actor_config.use_kl_loss:
            size_divisor_ref = (
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu
                * world_size
            )
            size_divisor = np.lcm(
                size_divisor_actor,
                size_divisor_ref,
            ).item()
    else:
        size_divisor_rollout = (
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
            * world_size
        )
        if config.algorithm.use_kl_in_reward or actor_config.use_kl_loss:
            size_divisor_ref = (
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu
                * world_size
            )
        else:
            size_divisor_ref = size_divisor_rollout
        size_divisor = np.lcm.reduce(
            np.array(
                [
                    size_divisor_ref,
                    size_divisor_rollout,
                    size_divisor_actor,
                ],
            )
        ).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    if (
        canonical_loss_plan_required
        and _trajectory_grpo_value(config, "scheduler", "row") == "row"
    ):
        # Once the adjusted batch reaches the configured global PPO mini-batch
        # size, row scheduling requires both divisors to align. Smaller batches
        # keep the trainer's existing clamp-to-batch-size behavior.
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        base_adjusted_size = (
            (bs + size_divisor - 1) // size_divisor
        ) * size_divisor
        if base_adjusted_size >= ppo_mini_batch_size:
            size_divisor = np.lcm(size_divisor, ppo_mini_batch_size).item()
    remainder = bs % size_divisor
    base_source_indices = data.non_tensor_batch.get("_batch_source_idx")
    if base_source_indices is None:
        base_source_indices = np.arange(bs, dtype=np.int64)
    else:
        base_source_indices = np.asarray(
            base_source_indices,
            dtype=np.int64,
        )
    if (
        pure_distillation
        or canonical_loss_plan_required
        or _needs_trajectory_row_metadata(config)
    ):
        if mode != "copy":
            raise ValueError(
                "trajectory-aware row processing only supports deterministic zero-weight padding"
            )
        if "uid" not in data.non_tensor_batch or "traj_uid" not in data.non_tensor_batch:
            raise ValueError("trajectory-aware row processing requires uid and traj_uid metadata")

        device = data.batch["input_ids"].device
        groups = group_rows_by_uid_traj_uid(
            data.non_tensor_batch["uid"],
            data.non_tensor_batch["traj_uid"],
        )
        trajectory_ids = np.empty(bs, dtype=np.int64)
        for trajectory_id, group in enumerate(groups):
            trajectory_ids[np.asarray(group.row_indices, dtype=np.int64)] = trajectory_id

        data.batch["row_weights"] = torch.ones(bs, dtype=torch.float32, device=device)
        data.batch["trajectory_id"] = torch.as_tensor(
            trajectory_ids,
            dtype=torch.int64,
            device=device,
        )

        if remainder == 0:
            if track_source_indices:
                data.non_tensor_batch["_batch_source_idx"] = (
                    base_source_indices
                )
            return data

        to_add = size_divisor - remainder
        padding = make_zero_weight_padding(
            np.arange(bs, dtype=np.int64).tolist(),
            bs + to_add,
        )
        adjusted_batch = data.select_idxs(padding.indices)
        adjusted_batch.batch["row_weights"] = torch.as_tensor(
            padding.weights,
            dtype=torch.float32,
            device=device,
        )
        if "loss_mask" in adjusted_batch.batch:
            adjusted_batch.batch["loss_mask"][bs:] = 0
        if track_source_indices:
            adjusted_batch.non_tensor_batch["_batch_source_idx"] = (
                base_source_indices[padding.indices]
            )
        return adjusted_batch

    if remainder == 0:
        if track_source_indices:
            data.non_tensor_batch["_batch_source_idx"] = base_source_indices
        return data
    
    dup_indices = np.empty(0, dtype=np.int64)
    keep_mask = np.ones(bs, dtype=bool)
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if track_source_indices:
        if mode == "copy":
            source_indices = np.concatenate(
                (
                    base_source_indices,
                    base_source_indices[
                        np.asarray(dup_indices, dtype=np.int64)
                    ],
                )
            )
        else:
            source_indices = base_source_indices[keep_mask]
        adjusted_batch.non_tensor_batch["_batch_source_idx"] = source_indices

    return adjusted_batch


FilterGroupResult = tuple[
    List[List[Dict]],
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]
FilterGroupWithIndicesResult = tuple[
    List[List[Dict]],
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@overload
def filter_group_data(
    batch_list: List[List[Dict]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: Dict[str, np.ndarray],
    traj_uid: np.ndarray,
    tool_callings: np.ndarray,
    config,
    last_try: bool = False,
    *,
    return_keep_indices: Literal[False] = False,
) -> FilterGroupResult: ...


@overload
def filter_group_data(
    batch_list: List[List[Dict]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: Dict[str, np.ndarray],
    traj_uid: np.ndarray,
    tool_callings: np.ndarray,
    config,
    last_try: bool = False,
    *,
    return_keep_indices: Literal[True],
) -> FilterGroupWithIndicesResult: ...


def filter_group_data(
    batch_list: List[List[Dict]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: Dict[str, np.ndarray],
    traj_uid: np.ndarray,
    tool_callings: np.ndarray,
    config,
    last_try: bool = False,
    *,
    return_keep_indices: bool = False,
) -> FilterGroupResult | FilterGroupWithIndicesResult:
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    filter_mode = str(_trajectory_grpo_value(config, "filter", "off")).replace("-", "_")
    penalty_aware = filter_mode == "penalty_aware"
    if last_try:
        if penalty_aware:
            print(
                "Warning: penalty-aware trajectory filtering exhausted its "
                "resampling budget; using the final unfiltered generation "
                "batch so training can continue."
            )
        result = cast(
            FilterGroupResult,
            (
                batch_list,
                episode_rewards,
                episode_lengths,
                success,
                traj_uid,
                tool_callings,
            ),
        )
        if return_keep_indices:
            return cast(
                FilterGroupWithIndicesResult,
                (*result, np.arange(len(batch_list), dtype=np.int64)),
            )
        return result

    if penalty_aware:
        trajectory_uids = np.asarray(
            [trajectory[0]["uid"] for trajectory in batch_list],
            dtype=object,
        )
        invalid_counts = _trajectory_invalid_counts(batch_list)
        keep_indices, _processed_rewards = select_penalty_aware_group_indices(
            cast(Sequence[Hashable], trajectory_uids),
            cast(Sequence[float], episode_rewards),
            cast(Sequence[float], invalid_counts),
            invalid_action_penalty_coef=config.actor_rollout_ref.actor.invalid_action_penalty_coef,
        )
    else:
        batch_size = config.data.train_batch_size
        group_n = config.env.rollout.n
        if group_n <= 1:
            print("Warning: group_n <= 1, no need to adopt dynamic sampling")

        # Handle each group
        keep_indices = np.array([], dtype=np.int64)
        for i in range(batch_size):
            # Get the indices of the current group
            group_indices = np.arange(i * group_n, (i + 1) * group_n)
            group_rewards = episode_rewards[group_indices]

            # check if all group_traj_uid are the same
            for index in group_indices:
                assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

            # Check if all rewards in the group are the same
            if not np.all(group_rewards == group_rewards[0]):
                # If so, keep the entire group, otherwise, remove it
                keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    result = cast(
        FilterGroupResult,
        (
            batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        ),
    )
    if return_keep_indices:
        return cast(
            FilterGroupWithIndicesResult,
            (*result, keep_indices),
        )
    return result
