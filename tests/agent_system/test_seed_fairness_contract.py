import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from agent_system.environments.fairness import (
    VALIDATION_CONCURRENCY,
    alfworld_gamefiles,
    alfworld_worker_gamefiles,
    canonical_validation_chunks,
    canonical_validation_splits,
    webshop_goal_fingerprint,
    webshop_goal_indices,
    webshop_reset_goal_indices,
)

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples"
SIZES = ("1.5b", "3b", "7b")
BENCHMARKS = ("alfworld", "webshop")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_canonical_fairness_manifests_are_ordered_and_chunked(
    tmp_path,
    monkeypatch,
):
    alfworld_dir = tmp_path / "alfworld"
    webshop_dir = tmp_path / "webshop"
    alfworld_dir.mkdir()
    webshop_dir.mkdir()
    for split, count in (
        ("evaluation_seen", 140),
        ("evaluation_unseen", 134),
    ):
        _write_jsonl(
            alfworld_dir / f"{split}.jsonl",
            [
                {"metadata": {"gamefile": f"$ALFWORLD_DATA/{split}-{i}"}}
                for i in range(count)
            ],
        )
    _write_jsonl(
        webshop_dir / "evaluation.jsonl",
        [{"metadata": {"goal_idx": i}} for i in range(500)],
    )
    monkeypatch.setenv("ALFWORLD_FAIRNESS_DIR", str(alfworld_dir))
    monkeypatch.setenv("WEBSHOP_FAIRNESS_DIR", str(webshop_dir))

    assert canonical_validation_splits("alfworld") == (
        "evaluation_seen",
        "evaluation_unseen",
    )
    assert canonical_validation_splits("webshop") == ("evaluation",)
    assert [len(x) for x in canonical_validation_chunks(
        "alfworld", "evaluation_seen"
    )] == [128, 12]
    assert [len(x) for x in canonical_validation_chunks(
        "alfworld", "evaluation_unseen"
    )] == [128, 6]
    assert [len(x) for x in canonical_validation_chunks(
        "webshop", "evaluation"
    )] == [128, 128, 128, 116]
    assert len(alfworld_gamefiles("evaluation_seen")) == 140
    assert webshop_goal_indices("evaluation") == list(range(500))

    with pytest.raises(ValueError, match="validation concurrency"):
        canonical_validation_chunks(
            "webshop",
            "evaluation",
            concurrency=VALIDATION_CONCURRENCY + 1,
        )


def test_fixed_assignment_and_webshop_goal_identity():
    games = [f"game-{i}" for i in range(12)]
    assert alfworld_worker_gamefiles(
        fixed_assignment=True,
        canonical_gamefiles=games,
        num_processes=12,
    ) == games
    goals = [{"instruction_text": "blue shirt", "goal_idx": 0}]
    assert webshop_goal_fingerprint(goals) == webshop_goal_fingerprint(
        [dict(goals[0])]
    )
    validation = list(range(500))
    assert webshop_reset_goal_indices(
        is_train=False,
        goal_indices=validation,
        env_num=500,
        group_n=1,
        rng=np.random.RandomState(0),
    ) == validation


def test_examples_expose_exactly_six_standalone_launchers():
    allowed_files = {
        "README.md",
        "__init__.py",
        *{
            f"seed_trainer_{size}/run_{benchmark}.sh"
            for size in SIZES
            for benchmark in BENCHMARKS
        },
    }
    visible_files = {
        str(path.relative_to(EXAMPLES))
        for path in EXAMPLES.rglob("*")
        if path.is_file()
        and "data_preprocess" not in path.parts
        and "__pycache__" not in path.parts
    }
    assert visible_files == allowed_files
    assert {
        path.name
        for path in (EXAMPLES / "data_preprocess").iterdir()
        if path.is_file()
    } == {"__init__.py", "prepare.py"}
    assert not list(EXAMPLES.rglob("__pycache__"))

    for size in SIZES:
        for benchmark in BENCHMARKS:
            launcher = (
                EXAMPLES
                / f"seed_trainer_{size}"
                / f"run_{benchmark}.sh"
            )
            text = launcher.read_text(encoding="utf-8")
            assert "Qwen2.5-${MODEL_LABEL}-Instruct" in text
            assert "SFT" not in text
            assert "conda activate" not in text
            assert "mamba activate" not in text
            assert "algorithm.adv_estimator=seed" in text
            assert "algorithm.seed.enable_analysis=True" in text
            assert "actor_rollout_ref.actor.opd_loss_coef=0.01" in text
            assert "env.fairness=true" in text
            assert "trainer.n_gpus_per_node=4" in text
            assert "trainer.total_training_steps=150" in text
            assert "trainer.test_freq=5" in text
            assert "trainer.save_freq=10" in text
            assert "trainer.max_actor_ckpt_to_keep=2" in text
            assert 'exec bash "${SCRIPT_DIR}/../' not in text
            assert '"$@"' in text


@pytest.mark.parametrize(
    ("size", "expected_model", "expected_tp"),
    (
        ("1.5b", "Qwen2.5-1.5B-Instruct", "1"),
        ("3b", "Qwen2.5-3B-Instruct", "2"),
        ("7b", "Qwen2.5-7B-Instruct", "4"),
    ),
)
@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_launcher_dry_run_resolves_size_and_keeps_cli_override_last(
    size,
    expected_model,
    expected_tp,
    benchmark,
):
    launcher = (
        EXAMPLES
        / f"seed_trainer_{size}"
        / f"run_{benchmark}.sh"
    )
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "trainer.experiment_name=cli_override",
        ],
        cwd=ROOT,
        env={**os.environ, "LAUNCHER_DRY_RUN": "true"},
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert f"actor_rollout_ref.model.path=/data/zhangdw12/models/{expected_model}" in lines
    assert f"actor_rollout_ref.rollout.tensor_model_parallel_size={expected_tp}" in lines
    assert lines[-1] == "trainer.experiment_name=cli_override"
