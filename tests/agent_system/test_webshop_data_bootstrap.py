import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BOOTSTRAP = ROOT / "scripts/bootstrap_webshop_data.sh"
WEBSHOP_ROOT = Path("agent_system/environments/env_package/webshop/webshop")
ASSETS = (
    Path("data/items_shuffle_1000.json"),
    Path("data/items_ins_v2_1000.json"),
    Path("data/items_human_ins.json"),
    Path("search_engine/indexes"),
)
LAUNCHERS = tuple(
    ROOT / "examples" / f"seed_trainer_{size}" / "run_webshop.sh"
    for size in ("1.5b", "3b", "7b")
)


def _prepare_shared_root(parent: Path) -> Path:
    shared_root = parent / "verl-agent" / WEBSHOP_ROOT
    for relative_path in ASSETS[:-1]:
        path = shared_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path.name, encoding="utf-8")
    indexes = shared_root / ASSETS[-1]
    indexes.mkdir(parents=True)
    (indexes / "segments_1").write_text("ready", encoding="utf-8")
    return shared_root


def _run_bootstrap(repo_root: Path):
    return subprocess.run(
        ["bash", str(BOOTSTRAP)],
        cwd=ROOT,
        env={
            **os.environ,
            "REPO_ROOT": str(repo_root),
            "PYTHON_BIN": sys.executable,
        },
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_webshop_launchers_bootstrap_after_dry_run(launcher: Path):
    text = launcher.read_text(encoding="utf-8")
    dry_run_offset = text.index('if [[ "${LAUNCHER_DRY_RUN:-false}" == true ]]')
    bootstrap_offset = text.index(
        'bash "${REPO_ROOT}/scripts/bootstrap_webshop_data.sh"'
    )
    preprocess_offset = text.index(
        '"${PYTHON_BIN}" -m examples.data_preprocess.prepare'
    )
    assert dry_run_offset < bootstrap_offset < preprocess_offset


def test_bootstrap_creates_only_relative_links(tmp_path: Path):
    shared_root = _prepare_shared_root(tmp_path)
    repo_root = tmp_path / ROOT.name

    _run_bootstrap(repo_root)
    _run_bootstrap(repo_root)

    for relative_path in ASSETS:
        local_path = repo_root / WEBSHOP_ROOT / relative_path
        assert local_path.is_symlink()
        assert not Path(os.readlink(local_path)).is_absolute()
        assert local_path.resolve() == (shared_root / relative_path).resolve()


def test_bootstrap_converts_matching_absolute_link(tmp_path: Path):
    shared_root = _prepare_shared_root(tmp_path)
    repo_root = tmp_path / ROOT.name
    local_path = repo_root / WEBSHOP_ROOT / ASSETS[0]
    local_path.parent.mkdir(parents=True)
    local_path.symlink_to(shared_root / ASSETS[0])
    assert Path(os.readlink(local_path)).is_absolute()

    _run_bootstrap(repo_root)

    assert local_path.is_symlink()
    assert not Path(os.readlink(local_path)).is_absolute()
    assert local_path.resolve() == (shared_root / ASSETS[0]).resolve()
