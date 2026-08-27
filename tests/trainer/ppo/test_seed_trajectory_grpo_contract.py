import os
import subprocess
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from verl import DataProto
from verl.trainer.ppo import ray_trainer

ROOT = Path(__file__).parents[3]
CONFIG_DIR = ROOT / "verl/trainer/config"
LAUNCHERS = tuple(
    ROOT / "examples" / f"seed_trainer_{size}" / f"run_{benchmark}.sh"
    for size in ("1.5b", "3b", "7b")
    for benchmark in ("alfworld", "webshop")
)


def _launcher_overrides(path):
    result = subprocess.run(
        ["bash", str(path)],
        cwd=ROOT,
        env={**os.environ, "LAUNCHER_DRY_RUN": "true"},
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if "=" in line]


def _compose_launcher(path):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        version_base=None,
    ):
        return compose(
            config_name="ppo_trainer",
            overrides=_launcher_overrides(path),
        )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_seed_launcher_real_hydra_compose_and_validation(launcher):
    config = _compose_launcher(launcher)
    assert config.data.seed == 0
    assert dict(config.algorithm.trajectory_grpo) == {
        "scheduler": "row",
        "reducer": "token_mean",
        "advantage": "step_row",
        "penalty": "step_local",
        "filter": "off",
    }

    trainer = cast(
        Any,
        object.__new__(ray_trainer.RayPPOTrainer),
    )
    trainer.config = config
    trainer.use_reference_policy = True
    trainer.use_critic = False
    trainer._validate_config()


class _StopAfterActorUpdate(Exception):
    pass


class _ActorRolloutStub:
    def compute_log_prob(self, batch):
        return DataProto.from_dict(
            tensors={
                "entropys": torch.zeros_like(
                    batch.batch["responses"],
                    dtype=torch.float32,
                ),
                "old_log_probs": torch.zeros_like(
                    batch.batch["responses"],
                    dtype=torch.float32,
                ),
            }
        )

    def update_actor(self, batch):
        assert batch.meta_info["multi_turn"] is False
        assert torch.count_nonzero(batch.batch["advantages"]).item() > 0
        raise _StopAfterActorUpdate


def _rollout_batch():
    prompts = torch.tensor([[1, 2], [3, 4]])
    responses = torch.tensor([[11, 12], [21, 22]])
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
        },
        non_tensors={
            "uid": np.asarray(["group", "group"], dtype=object),
            "traj_uid": np.asarray(["traj-0", "traj-1"], dtype=object),
        },
    )


def test_fit_consumes_native_step_row_before_actor_update(monkeypatch):
    config = _compose_launcher(LAUNCHERS[0])
    config.algorithm.adv_estimator = "grpo"
    config.actor_rollout_ref.actor.use_invalid_action_penalty = False
    config.trainer.val_before_train = False
    config.trainer.total_epochs = 1
    config.trainer.total_training_steps = 1
    config.trainer.balance_batch = False
    config.trainer.test_freq = 0
    config.trainer.save_freq = 0
    config.trainer.rollout_data_dir = None

    trainer = cast(
        Any,
        object.__new__(ray_trainer.RayPPOTrainer),
    )
    trainer.config = config
    trainer.val_reward_fn = None
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.arange(2).unsqueeze(0),
            "raw_prompt_ids": np.asarray([[1, 2]], dtype=object),
            "data_source": np.asarray(["alfworld"], dtype=object),
        }
    ]
    trainer.traj_collector = types.SimpleNamespace(
        multi_turn_loop=lambda **_kwargs: _rollout_batch()
    )
    trainer.actor_rollout_wg = _ActorRolloutStub()
    trainer.envs = object()
    trainer.reward_fn = object()
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.ref_in_actor = False
    trainer._load_checkpoint = types.MethodType(
        lambda _self: None,
        trainer,
    )

    observed = []
    original_trajectory_config = ray_trainer._trajectory_config

    def recording_trajectory_config(value):
        result = original_trajectory_config(value)
        observed.append(result)
        return result

    tracking_stub = types.ModuleType("verl.utils.tracking")
    tracking_stub_any = cast(Any, tracking_stub)
    tracking_stub_any.Tracking = type(
        "Tracking",
        (),
        {"__init__": lambda self, **_kwargs: None},
    )
    monkeypatch.setattr(
        ray_trainer,
        "_trajectory_config",
        recording_trajectory_config,
    )
    monkeypatch.setattr(
        ray_trainer,
        "adjust_batch",
        lambda _config, batch, **_kwargs: batch,
    )
    monkeypatch.setattr(
        ray_trainer,
        "compute_reward",
        lambda _batch, _reward_fn: (
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            {},
        ),
    )
    monkeypatch.setattr(
        ray_trainer,
        "tqdm",
        lambda **_kwargs: types.SimpleNamespace(),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "verl.utils.tracking",
        tracking_stub,
    )

    with pytest.raises(_StopAfterActorUpdate):
        trainer.fit()

    assert observed
    assert observed[-1] == {
        "scheduler": "row",
        "reducer": "token_mean",
        "advantage": "step_row",
        "penalty": "step_local",
        "filter": "off",
    }
