from pathlib import Path

from verl.workers.rollout.vllm_config import (
    build_vllm_sampling_params_kwargs,
    resolve_vllm_engine_seed,
)


class _FakeSamplingParams:
    def __init__(self):
        self.seed = None
        self.temperature = 1.0
        self.top_p = 1.0


def test_rollout_seed_is_engine_only():
    kwargs = build_vllm_sampling_params_kwargs(
        {
            "seed": 7,
            "temperature": 0.4,
            "top_p": 0.9,
            "unknown": "ignored",
        },
        _FakeSamplingParams,
        n=1,
        max_tokens=512,
        seed=99,
    )

    assert kwargs == {
        "n": 1,
        "max_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.9,
    }


def test_top_level_rollout_seed_wins_and_duplicate_engine_seed_is_removed():
    engine_kwargs = {
        "seed": 41,
        "swap_space": 8,
    }

    seed = resolve_vllm_engine_seed(
        {"seed": 7},
        engine_kwargs,
    )

    assert seed == 7
    assert engine_kwargs == {"swap_space": 8}


def test_legacy_engine_seed_remains_a_fallback():
    engine_kwargs = {"seed": 41}

    seed = resolve_vllm_engine_seed(
        {},
        engine_kwargs,
        offset=2,
    )

    assert seed == 43
    assert engine_kwargs == {}


def test_all_vllm_rollout_paths_use_shared_sampling_config_builder():
    repo_root = Path(__file__).parents[3]
    rollout_dir = repo_root / "verl/workers/rollout/vllm_rollout"
    paths = [
        rollout_dir / "vllm_rollout_spmd.py",
        rollout_dir / "vllm_rollout.py",
        rollout_dir / "vllm_async_server.py",
        rollout_dir / "fire_vllm_rollout.py",
    ]

    for path in paths:
        source = path.read_text()
        assert "build_vllm_sampling_params_kwargs(" in source
        assert "if hasattr(SamplingParams(), str(k))" not in source


def test_rollout_seed_still_controls_vllm_engine_rng():
    repo_root = Path(__file__).parents[3]
    rollout_dir = repo_root / "verl/workers/rollout/vllm_rollout"

    spmd_source = (rollout_dir / "vllm_rollout_spmd.py").read_text()
    assert "engine_seed = resolve_vllm_engine_seed(" in spmd_source
    assert "seed=engine_seed" in spmd_source

    legacy_source = (rollout_dir / "vllm_rollout.py").read_text()
    assert "engine_seed = resolve_vllm_engine_seed(" in legacy_source
    assert "seed=engine_seed" in legacy_source

    async_source = (rollout_dir / "vllm_async_server.py").read_text()
    assert "seed=resolve_vllm_engine_seed(" in async_source
    assert "offset=self.vllm_dp_rank" in async_source
