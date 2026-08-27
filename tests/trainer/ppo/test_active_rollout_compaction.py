import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.__spec__ = ModuleSpec("datasets", loader=None)
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub

if "gymnasium" not in sys.modules:
    gymnasium_stub = types.ModuleType("gymnasium")
    gymnasium_stub.__spec__ = ModuleSpec("gymnasium", loader=None)
    gymnasium_stub.Env = object
    gymnasium_stub.spaces = types.ModuleType("gymnasium.spaces")
    sys.modules["gymnasium"] = gymnasium_stub
    sys.modules["gymnasium.spaces"] = gymnasium_stub.spaces

if "gym" not in sys.modules:
    gym_stub = types.ModuleType("gym")
    gym_stub.__spec__ = ModuleSpec("gym", loader=None)
    gym_stub.Env = object
    sys.modules["gym"] = gym_stub

if "torchvision" not in sys.modules:
    transforms_stub = types.ModuleType("torchvision.transforms")
    transforms_stub.Compose = lambda transforms: transforms
    transforms_stub.ToTensor = object
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.__spec__ = ModuleSpec("torchvision", loader=None)
    torchvision_stub.transforms = transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub

alfworld_vendor_stub = types.ModuleType(
    "agent_system.environments.env_package.alfworld.alfworld"
)
alfworld_agents_stub = types.ModuleType(
    "agent_system.environments.env_package.alfworld.alfworld.agents"
)
alfworld_environment_stub = types.ModuleType(
    "agent_system.environments.env_package.alfworld.alfworld.agents.environment"
)
alfworld_environment_stub.get_environment = lambda env_type: None
sys.modules.setdefault(alfworld_vendor_stub.__name__, alfworld_vendor_stub)
sys.modules.setdefault(alfworld_agents_stub.__name__, alfworld_agents_stub)
sys.modules.setdefault(alfworld_environment_stub.__name__, alfworld_environment_stub)

from agent_system.environments.env_manager import (  # noqa: E402
    AlfWorldEnvironmentManager,
    WebshopEnvironmentManager,
)
from agent_system.environments.env_package.alfworld.envs import (  # noqa: E402
    AlfworldEnvs,
)
from agent_system.environments.env_package.webshop.envs import (  # noqa: E402
    WebshopMultiProcessEnv,
)
from agent_system.multi_turn_rollout.rollout_loop import (  # noqa: E402
    TrajectoryCollector,
)
from seed.analysis import build_traj_step_indices  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.trainer.ppo import ray_trainer  # noqa: E402


def _config(max_steps=3, collect_env_aux_data=False):
    return OmegaConf.create(
        {
            "env": {
                "max_steps": max_steps,
                "history_length": 10,
                "rollout": {"n": 1},
            },
            "actor_rollout_ref": {
                "actor": {
                    "collect_env_aux_data": collect_env_aux_data,
                    "sp_coef": 0.0,
                    "id_coef": 0.0,
                }
            },
            "algorithm": {
                "filter_groups": {"enable": False},
                "trajectory_grpo": {"filter": "off"},
            },
            "data": {"train_batch_size": 3},
        }
    )


def _gen_batch(size=3):
    return DataProto.from_dict(
        tensors={"input_ids": torch.arange(size).reshape(-1, 1)},
        non_tensors={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "prompt"}]] * size,
                dtype=object,
            ),
            "data_source": np.array(["alfworld"] * size, dtype=object),
        },
    )


def _fake_preprocess(gen_batch, obs):
    size = len(gen_batch)
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(size).reshape(-1, 1),
            "attention_mask": torch.ones((size, 1), dtype=torch.long),
            "position_ids": torch.arange(size).reshape(-1, 1),
        },
        non_tensors={
            "raw_prompt_ids": np.array(
                [[index] for index in range(size)],
                dtype=object,
            ),
            "anchor_obs": np.array(obs["anchor"], dtype=object),
            "obs_text": np.array(obs["text"], dtype=object),
            "index": np.arange(size, dtype=object),
        },
    )


class FakeTokenizer:
    def batch_decode(self, responses, skip_special_tokens=True):
        return [f"action-{int(response[0])}" for response in responses]


class FakeActorRollout:
    world_size = 1

    def __init__(self):
        self.generation_batch_sizes = []
        self.begin_calls = 0
        self.end_calls = 0
        self.begin_error = None
        self.end_error = None
        self.generation_error = None

    def begin_rollout_session(self):
        self.begin_calls += 1
        if self.begin_error is not None:
            raise self.begin_error

    def end_rollout_session(self):
        self.end_calls += 1
        if self.end_error is not None:
            raise self.end_error

    def generate_sequences(self, batch):
        self.generation_batch_sizes.append(len(batch))
        if self.generation_error is not None:
            raise self.generation_error
        tensors = {key: value.clone() for key, value in batch.batch.items()}
        tensors["responses"] = torch.arange(len(batch)).reshape(-1, 1)
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={
                key: value.copy()
                for key, value in batch.non_tensor_batch.items()
            },
        )


class SelectedStepManager:
    def __init__(self):
        self.step_indices = []
        self.step_count = 0

    def reset(self, kwargs):
        return (
            {
                "text": ["obs-0", "obs-1", "obs-2"],
                "text_base": ["base-0", "base-1", "base-2"],
                "image": None,
                "anchor": ["anchor-0", "anchor-1", "anchor-2"],
            },
            [
                {"admissible_commands": [f"cmd-{index}"]}
                for index in range(3)
            ],
        )

    def step_selected(self, text_actions, indices):
        self.step_indices.append(list(indices))
        self.step_count += 1
        dones = np.array(
            [index == self.step_count - 1 for index in indices],
            dtype=bool,
        )
        return (
            {
                "text": [
                    f"obs-{index}-step-{self.step_count}"
                    for index in indices
                ],
                "text_base": [
                    f"base-{index}-step-{self.step_count}"
                    for index in indices
                ],
                "image": None,
                "anchor": [
                    f"anchor-{index}-step-{self.step_count}"
                    for index in indices
                ],
            },
            np.ones(len(indices), dtype=np.float32),
            dones,
            [
                {
                    "won": float(done),
                    "admissible_commands": [f"next-{index}"],
                }
                for index, done in zip(indices, dones)
            ],
        )

    def success_evaluator(self, **kwargs):
        return {
            "success_rate": np.array(
                [infos[-1]["won"] for infos in kwargs["total_infos"]],
                dtype=np.float32,
            )
        }


class LegacyStepManager:
    def __init__(self):
        self.step_batch_sizes = []
        self.step_count = 0

    def reset(self, kwargs):
        return (
            {
                "text": ["obs-0", "obs-1", "obs-2"],
                "text_base": ["base-0", "base-1", "base-2"],
                "image": None,
                "anchor": ["anchor-0", "anchor-1", "anchor-2"],
            },
            [
                {"admissible_commands": [f"cmd-{index}"]}
                for index in range(3)
            ],
        )

    def step(self, text_actions):
        self.step_batch_sizes.append(len(text_actions))
        self.step_count += 1
        dones = np.array(
            [
                self.step_count >= 1,
                self.step_count >= 2,
                self.step_count >= 3,
            ],
            dtype=bool,
        )
        return (
            {
                "text": [
                    f"obs-{index}-step-{self.step_count}"
                    for index in range(3)
                ],
                "text_base": [
                    f"base-{index}-step-{self.step_count}"
                    for index in range(3)
                ],
                "image": None,
                "anchor": [
                    f"anchor-{index}-step-{self.step_count}"
                    for index in range(3)
                ],
            },
            np.ones(3, dtype=np.float32),
            dones,
            [
                {
                    "won": float(done),
                    "admissible_commands": [f"next-{index}"],
                }
                for index, done in enumerate(dones)
            ],
        )

    def success_evaluator(self, **kwargs):
        return {
            "success_rate": np.array(
                [infos[-1]["won"] for infos in kwargs["total_infos"]],
                dtype=np.float32,
            )
        }


def _collector(max_steps=3, collect_env_aux_data=False):
    collector = TrajectoryCollector(
        _config(
            max_steps=max_steps,
            collect_env_aux_data=collect_env_aux_data,
        ),
        FakeTokenizer(),
    )
    collector.preprocess_batch = _fake_preprocess
    return collector


def test_staggered_termination_compacts_generation_and_preserves_seed_sidecars():
    actor = FakeActorRollout()
    envs = SelectedStepManager()
    collector = _collector(collect_env_aux_data=True)

    trajectories, rewards, lengths, success, traj_uid, tool_callings = (
        collector.vanilla_multi_turn_loop(_gen_batch(), actor, envs)
    )

    assert actor.generation_batch_sizes == [3, 2, 1]
    assert envs.step_indices == [[0, 1, 2], [1, 2], [2]]
    assert [len(trajectory) for trajectory in trajectories] == [1, 2, 3]
    assert [[row["index"] for row in trajectory] for trajectory in trajectories] == [
        [0],
        [1, 1],
        [2, 2, 2],
    ]
    assert [[row["step_num"] for row in trajectory] for trajectory in trajectories] == [
        [0],
        [0, 1],
        [0, 1, 2],
    ]
    assert [row["step_id"] for row in trajectories[2]] == ["2_0_0", "2_0_1", "2_0_2"]
    assert [row["traj_uid"] for row in trajectories[2]] == [traj_uid[2]] * 3
    assert trajectories[2][2]["history"] == [
        {"text_obs": "anchor-2", "action": "action-2"},
        {"text_obs": "anchor-2-step-1", "action": "action-1"},
    ]
    assert trajectories[2][1]["admissibles"] == ["next-2"]
    assert trajectories[2][2]["next_obs"] == "anchor-2-step-3"
    np.testing.assert_array_equal(rewards, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lengths, [1.0, 2.0, 3.0])

    gathered = collector.gather_rollout_data(
        trajectories,
        rewards,
        lengths,
        success,
        traj_uid,
        tool_callings,
    )
    assert len(gathered) == 6
    assert list(gathered.non_tensor_batch["traj_uid"]).count(traj_uid[2]) == 3
    np.testing.assert_array_equal(
        build_traj_step_indices(gathered.non_tensor_batch["traj_uid"]),
        [0, 0, 1, 0, 1, 2],
    )

    trainer = cast(Any, object.__new__(ray_trainer.RayPPOTrainer))
    snapshot = trainer._build_seed_teacher_signal_snapshot(gathered)
    np.testing.assert_array_equal(
        snapshot.non_tensor_batch["traj_uid"],
        gathered.non_tensor_batch["traj_uid"],
    )
    source_indices = np.asarray([5, 0, 2, 2], dtype=np.int64)
    padded = gathered[source_indices]
    padded.non_tensor_batch["_batch_source_idx"] = source_indices
    row_values = torch.arange(len(gathered), dtype=torch.float32).unsqueeze(1)
    row_values = row_values.expand_as(gathered.batch["responses"])
    teacher_signal_batch = DataProto.from_dict(
        tensors={
            "teacher_log_prob": row_values,
            "episode_teacher_log_prob": row_values + 10,
            "step_teacher_log_prob": row_values + 20,
            "critical_step_mask": torch.tensor(
                [False, True, False, True, False, True]
            ),
            "step_skill_mask": torch.tensor(
                [True, False, True, False, True, False]
            ),
            "teacher_signal_mask": torch.tensor(
                [True, True, False, True, False, True]
            ),
        },
        meta_info={},
    )
    trainer._build_seed_skill_gen_payload = lambda **_kwargs: None
    merged = trainer._merge_async_seed_teacher_signals(
        batch=padded,
        teacher_signal_batch=teacher_signal_batch,
    )
    torch.testing.assert_close(
        merged.batch["teacher_log_prob"][:, 0],
        torch.tensor([5.0, 0.0, 2.0, 2.0]),
    )
    assert "_batch_source_idx" not in merged.non_tensor_batch


def test_compacted_and_legacy_paths_preserve_active_transition_rewards():
    compact_actor = FakeActorRollout()
    compact = _collector()
    compact_result = compact.vanilla_multi_turn_loop(
        _gen_batch(),
        compact_actor,
        SelectedStepManager(),
    )

    legacy_actor = FakeActorRollout()
    legacy = _collector()
    legacy_env = LegacyStepManager()
    legacy_result = legacy.vanilla_multi_turn_loop(
        _gen_batch(),
        legacy_actor,
        legacy_env,
    )

    compact_trajectories, compact_rewards, compact_lengths = compact_result[:3]
    legacy_trajectories, legacy_rewards, legacy_lengths = legacy_result[:3]
    assert compact_actor.generation_batch_sizes == [3, 2, 1]
    assert legacy_actor.generation_batch_sizes == [3, 3, 3]
    assert legacy_env.step_batch_sizes == [3, 3, 3]
    np.testing.assert_array_equal(compact_rewards, legacy_rewards)
    np.testing.assert_array_equal(compact_lengths, legacy_lengths)
    assert [
        [row["rewards"] for row in trajectory]
        for trajectory in compact_trajectories
    ] == [
        [
            row["rewards"]
            for row in trajectory
            if row["active_masks"]
        ]
        for trajectory in legacy_trajectories
    ]


def test_rollout_session_cleanup_and_capability_fallback():
    collector = _collector()
    actor = FakeActorRollout()
    actor.generation_error = RuntimeError("generation failed")

    try:
        collector.vanilla_multi_turn_loop(_gen_batch(), actor, SelectedStepManager())
    except RuntimeError as error:
        assert str(error) == "generation failed"
    else:
        raise AssertionError("generation failure did not propagate")
    assert actor.begin_calls == 1
    assert actor.end_calls == 1

    actor = FakeActorRollout()
    actor.generation_error = RuntimeError("generation failed")
    actor.end_error = ValueError("cleanup failed")
    try:
        collector.vanilla_multi_turn_loop(_gen_batch(), actor, SelectedStepManager())
    except RuntimeError as error:
        assert str(error) == "generation failed"
        assert error.rollout_session_cleanup_error is actor.end_error
    else:
        raise AssertionError("primary rollout failure did not propagate")

    class ActorWithoutSession:
        world_size = 1

        def __init__(self):
            self.delegate = FakeActorRollout()

        def generate_sequences(self, batch):
            return self.delegate.generate_sequences(batch)

    fallback_actor = ActorWithoutSession()
    trajectories = collector.vanilla_multi_turn_loop(
        _gen_batch(),
        fallback_actor,
        SelectedStepManager(),
    )[0]
    assert fallback_actor.delegate.generation_batch_sizes == [3, 2, 1]
    assert [len(trajectory) for trajectory in trajectories] == [1, 2, 3]


class FakeAlfworldVectorEnv:
    def __init__(self):
        self.get_admissible_commands = [["a0"], ["a1"], ["a2"]]
        self.selected_calls = []

    def reset(self):
        return (
            [
                "room 0. Your task is to: task zero",
                "room 1. Your task is to: task one",
                "room 2. Your task is to: task two",
            ],
            None,
            [{"extra.gamefile": f"game-{index}"} for index in range(3)],
        )

    def step_selected(self, actions, indices):
        self.selected_calls.append((list(actions), list(indices)))
        text_obs = []
        infos = []
        for action, index in zip(actions, indices):
            self.get_admissible_commands[index] = [f"next-{index}"]
            text_obs.append(f"result-{index}-{action}")
            infos.append({"extra.gamefile": None, "won": 0})
        return (
            text_obs,
            None,
            np.zeros(len(indices), dtype=np.float32),
            np.zeros(len(indices), dtype=bool),
            infos,
        )


def test_alfworld_manager_preserves_original_slot_gamefile_history_and_observation():
    envs = FakeAlfworldVectorEnv()
    manager = AlfWorldEnvironmentManager(
        envs,
        lambda actions, action_pools: (list(actions), [True] * len(actions)),
        _config(),
    )
    manager.reset(kwargs=None)

    next_obs, _, _, infos = manager.step_selected(["act-2", "act-0"], [2, 0])
    later_obs, _, _, _ = manager.step_selected(["act-2b"], [2])

    assert [len(manager.memory[index]) for index in range(3)] == [1, 0, 2]
    assert manager.memory[2][0]["text_obs"].startswith("room 2.")
    assert manager.pre_text_obs == [
        "result-0-act-0",
        "room 1. Your task is to: task one",
        "result-2-act-2b",
    ]
    assert "task two" in next_obs["text"][0]
    assert "task zero" in next_obs["text"][1]
    assert "act-2" in later_obs["text"][0]
    assert infos[0]["extra.gamefile"] == "game-2"
    assert infos[1]["extra.gamefile"] == "game-0"


def _webshop_info(index, won=False):
    return {
        "available_actions": {
            "has_search_bar": True,
            "clickables": [f"item-{index}"],
        },
        "won": won,
        "task_score": float(won),
    }


def _webshop_observation(index, suffix):
    return (
        f"WebShop [SEP] Instruction: [SEP] task {index} "
        f"[SEP] page-{index}-{suffix}"
    )


class FakeWebshopVectorEnv:
    def __init__(self, terminate=False):
        self.selected_calls = []
        self.step_count = 0
        self.terminate = terminate

    def reset(self):
        self.step_count = 0
        return (
            [
                _webshop_observation(index, "reset")
                for index in range(3)
            ],
            [_webshop_info(index) for index in range(3)],
        )

    def step_selected(self, actions, indices):
        self.selected_calls.append((list(actions), list(indices)))
        self.step_count += 1
        dones = np.array(
            [
                self.terminate and index == self.step_count - 1
                for index in indices
            ],
            dtype=bool,
        )
        return (
            [
                _webshop_observation(index, action)
                for action, index in zip(actions, indices)
            ],
            np.ones(len(indices), dtype=np.float32),
            dones,
            [
                _webshop_info(index, won=bool(done))
                for index, done in zip(indices, dones)
            ],
        )


def test_webshop_manager_and_collector_compact_on_original_slots():
    envs = FakeWebshopVectorEnv(terminate=True)
    manager = WebshopEnvironmentManager(
        envs,
        lambda actions: (list(actions), [True] * len(actions)),
        _config(),
    )
    actor = FakeActorRollout()
    collector = _collector()

    trajectories, rewards, lengths = collector.vanilla_multi_turn_loop(
        _gen_batch(),
        actor,
        manager,
    )[:3]

    assert actor.generation_batch_sizes == [3, 2, 1]
    assert [indices for _, indices in envs.selected_calls] == [
        [0, 1, 2],
        [1, 2],
        [2],
    ]
    assert [len(trajectory) for trajectory in trajectories] == [1, 2, 3]
    assert [len(manager.memory[index]) for index in range(3)] == [1, 2, 3]
    assert "page-0-reset" in manager.memory[0][0]["text_obs"]
    assert "page-2-reset" in manager.memory[2][0]["text_obs"]
    np.testing.assert_array_equal(rewards, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lengths, [1.0, 2.0, 3.0])


class FakeRemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, action=None):
        self.calls.append(action)
        return self.result


def test_vector_envs_step_only_selected_workers(monkeypatch):
    alfworld = object.__new__(AlfworldEnvs)
    alfworld.num_processes = 3
    alfworld.multi_modal = False
    alfworld.prev_admissible_commands = [["old-0"], ["old-1"], ["old-2"]]
    alfworld.workers = []
    for index in range(3):
        alfworld.workers.append(
            types.SimpleNamespace(
                step=FakeRemoteMethod(
                    (
                        [f"obs-{index}"],
                        [0],
                        [False],
                        {
                            "won": [0],
                            "goal_condition_success_rate": [0.0],
                            "admissible_commands": [[f"new-{index}"]],
                        },
                    )
                )
            )
        )

    monkeypatch.setattr(
        "agent_system.environments.env_package.alfworld.envs.ray.get",
        lambda futures: futures,
    )
    text_obs = alfworld.step_selected(["action-2", "action-0"], [2, 0])[0]
    assert text_obs == ["obs-2", "obs-0"]
    assert alfworld.workers[1].step.calls == []
    assert alfworld.prev_admissible_commands == [
        ["new-0"],
        ["old-1"],
        ["new-2"],
    ]

    webshop = object.__new__(WebshopMultiProcessEnv)
    webshop.num_processes = 3
    webshop._closed = True
    webshop._workers = [
        types.SimpleNamespace(
            step=FakeRemoteMethod(
                (f"obs-{index}", float(index), False, _webshop_info(index))
            )
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        "agent_system.environments.env_package.webshop.envs.ray.get",
        lambda futures: futures,
    )
    obs, rewards, _, _ = webshop.step_selected(
        ["action-2", "action-0"],
        [2, 0],
    )
    assert obs == ["obs-2", "obs-0"]
    assert rewards == [2.0, 0.0]
    assert webshop._workers[1].step.calls == []
