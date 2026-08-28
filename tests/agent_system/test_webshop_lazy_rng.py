import sys
import types
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments import env_manager


def _make_webshop_config(*, fairness=False):
    return OmegaConf.create(
        {
            "algorithm": {"lhop": {"enable": False}},
            "actor_rollout_ref": {
                "model": {"path": "Qwen2.5-1.5B-Instruct"}
            },
            "data": {
                "train_batch_size": 2,
                "val_batch_size": 3,
                "apply_chat_template_kwargs": {},
            },
            "env": {
                "env_name": "Webshop",
                "fairness": fairness,
                "seed": 17,
                "rollout": {"n": 2},
                "resources_per_worker": {
                    "num_cpus": 0.1,
                    "num_gpus": 0,
                },
                "webshop": {
                    "use_small": True,
                    "human_goals": False,
                },
            },
            "trainer": {"val_only": False},
        }
    )


def _install_fake_webshop(monkeypatch):
    builds = []
    closes = []

    class FakeRawEnvs:
        def __init__(self, phase, sample):
            self.phase = phase
            self.sample = sample

        def close(self):
            closes.append(self.phase)

    class FakeManager:
        def __init__(self, raw_envs, projection, _config):
            self.raw_envs = raw_envs
            self.projection = projection

        def sample(self):
            return self.raw_envs.sample

        def close(self):
            self.raw_envs.close()

    def build_webshop_envs(*, is_train, rng=None, **kwargs):
        assert rng is not None
        phase = "train" if is_train else "validation"
        sample = int(rng.choice(np.arange(10_000)))
        builds.append(
            {
                "phase": phase,
                "rng": rng,
                "sample": sample,
                "kwargs": kwargs,
            }
        )
        return FakeRawEnvs(phase, sample)

    webshop_module = types.ModuleType(
        "agent_system.environments.env_package.webshop"
    )
    webshop_module_any = cast(Any, webshop_module)
    webshop_module_any.build_webshop_envs = build_webshop_envs
    webshop_module_any.webshop_projection = (
        lambda actions, **_kwargs: actions
    )
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.webshop",
        webshop_module,
    )
    monkeypatch.setattr(
        env_manager,
        "WebshopEnvironmentManager",
        FakeManager,
    )
    return builds, closes


def test_webshop_normal_path_does_not_eagerly_build_pools(monkeypatch):
    builds, _ = _install_fake_webshop(monkeypatch)

    train_envs, val_envs = env_manager.make_envs(
        _make_webshop_config()
    )

    assert builds == []
    train_envs.close()
    val_envs.close()


def test_webshop_normal_path_train_and_validation_are_phase_exclusive(
    monkeypatch,
):
    builds, closes = _install_fake_webshop(monkeypatch)
    train_envs, val_envs = env_manager.make_envs(
        _make_webshop_config()
    )

    train_envs.sample()
    assert [build["phase"] for build in builds] == ["train"]

    val_envs.sample()
    assert closes == ["train"]
    assert [build["phase"] for build in builds] == [
        "train",
        "validation",
    ]

    train_envs.sample()
    assert closes == ["train", "validation"]
    assert [build["phase"] for build in builds] == [
        "train",
        "validation",
        "train",
    ]

    train_envs.close()
    val_envs.close()


def test_webshop_normal_path_reuses_each_phase_rng_after_reacquire(
    monkeypatch,
):
    builds, _ = _install_fake_webshop(monkeypatch)
    train_envs, val_envs = env_manager.make_envs(
        _make_webshop_config()
    )

    train_samples = [train_envs.sample()]
    validation_samples = [val_envs.sample()]
    train_samples.append(train_envs.sample())
    validation_samples.append(val_envs.sample())

    train_builds = [
        build for build in builds if build["phase"] == "train"
    ]
    validation_builds = [
        build for build in builds if build["phase"] == "validation"
    ]
    train_control = np.random.RandomState(17)
    validation_control = np.random.RandomState(1017)

    assert train_samples == [
        int(train_control.choice(np.arange(10_000))),
        int(train_control.choice(np.arange(10_000))),
    ]
    assert validation_samples == [
        int(validation_control.choice(np.arange(10_000))),
        int(validation_control.choice(np.arange(10_000))),
    ]
    assert train_builds[0]["rng"] is train_builds[1]["rng"]
    assert validation_builds[0]["rng"] is validation_builds[1]["rng"]

    train_envs.close()
    val_envs.close()


def test_webshop_fairness_path_keeps_canonical_validation_and_training_rng(
    monkeypatch,
):
    builds, _ = _install_fake_webshop(monkeypatch)
    train_envs, val_envs = env_manager.make_envs(
        _make_webshop_config(fairness=True)
    )

    first = train_envs.sample()
    train_envs.release()
    second = train_envs.sample()
    train_builds = [
        build for build in builds if build["phase"] == "train"
    ]
    control = np.random.RandomState(17)

    assert isinstance(
        val_envs,
        env_manager.CanonicalValidationEnvironments,
    )
    assert [first, second] == [
        int(control.choice(np.arange(10_000))),
        int(control.choice(np.arange(10_000))),
    ]
    assert train_builds[0]["rng"] is train_builds[1]["rng"]

    train_envs.close()
    val_envs.close()
