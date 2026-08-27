import sys
import types
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments import env_manager


def test_webshop_lazy_release_reacquire_continues_training_rng(
    monkeypatch,
):
    samples = []
    rng_objects = []

    class FakeRawEnvs:
        def __init__(self, sample):
            self.sample = sample

        def close(self):
            pass

    class FakeManager:
        def __init__(self, raw_envs, _projection, _config):
            self.raw_envs = raw_envs

        def sample(self):
            return self.raw_envs.sample

        def close(self):
            self.raw_envs.close()

    def build_webshop_envs(*, is_train, rng=None, **_kwargs):
        assert is_train
        assert rng is not None
        rng_objects.append(rng)
        sample = int(rng.choice(np.arange(10_000)))
        samples.append(sample)
        return FakeRawEnvs(sample)

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

    config = OmegaConf.create(
        {
            "algorithm": {"lhop": {"enable": False}},
            "actor_rollout_ref": {
                "model": {"path": "Qwen2.5-1.5B-Instruct"}
            },
            "data": {
                "train_batch_size": 2,
                "val_batch_size": 128,
                "apply_chat_template_kwargs": {},
            },
            "env": {
                "env_name": "Webshop",
                "fairness": True,
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

    train_envs, _ = env_manager.make_envs(config)
    train_envs = cast(Any, train_envs)
    first = train_envs.sample()
    train_envs.release()
    second = train_envs.sample()
    train_envs.close()

    control = np.random.RandomState(17)
    assert [first, second] == [
        int(control.choice(np.arange(10_000))),
        int(control.choice(np.arange(10_000))),
    ]
    assert len(rng_objects) == 2
    assert rng_objects[0] is rng_objects[1]
    assert samples == [first, second]
