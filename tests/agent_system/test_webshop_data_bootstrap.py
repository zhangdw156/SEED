import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BOOTSTRAP = ROOT / "scripts/bootstrap_webshop_data.sh"
WEBSHOP_RELATIVE_ROOT = Path(
    "agent_system/environments/env_package/webshop/webshop"
)
ASSETS = (
    Path("data/items_shuffle_1000.json"),
    Path("data/items_ins_v2_1000.json"),
    Path("data/items_human_ins.json"),
    Path("search_engine/indexes"),
)
WEBSHOP_LAUNCHERS = tuple(
    ROOT / "examples" / f"seed_trainer_{size}" / "run_webshop.sh"
    for size in ("1.5b", "3b", "7b")
)


def _prepare_shared_root(shared_root):
    for relative_path in ASSETS[:-1]:
        path = shared_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path.name, encoding="utf-8")
    indexes = shared_root / ASSETS[-1]
    indexes.mkdir(parents=True)
    (indexes / "segments_1").write_text("ready", encoding="utf-8")


def _run_bootstrap(repo_root, shared_root, *, check=True):
    return subprocess.run(
        ["bash", str(BOOTSTRAP)],
        cwd=ROOT,
        env={
            **os.environ,
            "REPO_ROOT": str(repo_root),
            "WEBSHOP_SHARED_ROOT": str(shared_root),
            "WEBSHOP_LOCAL_ROOT": str(repo_root / WEBSHOP_RELATIVE_ROOT),
        },
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("launcher", WEBSHOP_LAUNCHERS)
def test_webshop_launcher_keeps_python3_and_bootstrap_order(launcher):
    text = launcher.read_text(encoding="utf-8")

    assert 'export PYTHON_BIN="${PYTHON_BIN:-python3}"' in text
    dry_run_offset = text.index('if [[ "${LAUNCHER_DRY_RUN:-false}" == true ]]')
    bootstrap_offset = text.index(
        'bash "${REPO_ROOT}/scripts/bootstrap_webshop_data.sh"'
    )
    preprocess_offset = text.index(
        '"${PYTHON_BIN}" -m examples.data_preprocess.prepare'
    )
    assert dry_run_offset < bootstrap_offset < preprocess_offset


def test_bootstrap_links_missing_assets_and_is_idempotent(tmp_path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    _prepare_shared_root(shared_root)

    local_root = repo_root / WEBSHOP_RELATIVE_ROOT
    preserved = local_root / ASSETS[0]
    preserved.parent.mkdir(parents=True)
    preserved.write_text("local", encoding="utf-8")

    _run_bootstrap(repo_root, shared_root)
    first_links = {}
    for relative_path in ASSETS:
        local_path = local_root / relative_path
        if relative_path == ASSETS[0]:
            assert local_path.read_text(encoding="utf-8") == "local"
            assert not local_path.is_symlink()
        else:
            assert local_path.is_symlink()
            first_links[relative_path] = os.readlink(local_path)
            assert local_path.resolve() == (shared_root / relative_path).resolve()

    _run_bootstrap(repo_root, shared_root)
    assert preserved.read_text(encoding="utf-8") == "local"
    assert {
        relative_path: os.readlink(local_root / relative_path)
        for relative_path in first_links
    } == first_links


def test_bootstrap_missing_source_fails_before_creating_links(tmp_path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    _prepare_shared_root(shared_root)
    missing_source = shared_root / ASSETS[2]
    missing_source.unlink()

    result = _run_bootstrap(repo_root, shared_root, check=False)

    assert result.returncode != 0
    assert str(missing_source) in result.stderr
    assert "setup.sh -d all" in result.stderr
    assert not (repo_root / WEBSHOP_RELATIVE_ROOT).exists()


@pytest.mark.parametrize("launcher", WEBSHOP_LAUNCHERS)
def test_launcher_dry_run_never_bootstraps_webshop_assets(
    launcher,
    tmp_path,
):
    repo_root = tmp_path / "clean-clone"
    repo_root.mkdir()

    subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env={
            **os.environ,
            "LAUNCHER_DRY_RUN": "true",
            "REPO_ROOT": str(repo_root),
            "WEBSHOP_SHARED_ROOT": str(tmp_path / "missing-shared-root"),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert list(repo_root.iterdir()) == []


def test_bootstrap_serializes_concurrent_launchers(tmp_path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    local_root = repo_root / WEBSHOP_RELATIVE_ROOT
    _prepare_shared_root(shared_root)
    environment = {
        **os.environ,
        "REPO_ROOT": str(repo_root),
        "WEBSHOP_SHARED_ROOT": str(shared_root),
        "WEBSHOP_LOCAL_ROOT": str(local_root),
    }

    processes = [
        subprocess.Popen(
            ["bash", str(BOOTSTRAP)],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(16)
    ]
    results = [process.communicate(timeout=20) + (process.returncode,) for process in processes]

    assert all(returncode == 0 for _, _, returncode in results), results
    for relative_path in ASSETS:
        local_path = local_root / relative_path
        assert local_path.is_symlink()
        assert local_path.resolve() == (shared_root / relative_path).resolve()


def test_bootstrap_rolls_back_links_after_failure(tmp_path):
    repo_root = tmp_path / "clone"
    shared_root = tmp_path / "shared"
    local_root = repo_root / WEBSHOP_RELATIVE_ROOT
    bin_dir = tmp_path / "bin"
    counter = tmp_path / "ln-count"
    _prepare_shared_root(shared_root)
    bin_dir.mkdir()
    ln_stub = bin_dir / "ln"
    real_ln = shutil.which("ln")
    assert real_ln is not None
    ln_stub.write_text(
        "#!/usr/bin/env bash\n"
        'count="$(cat "${LN_COUNTER}" 2>/dev/null || printf 0)"\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "${count}" > "${LN_COUNTER}"\n'
        'if (( count == 2 )); then exit 42; fi\n'
        'exec "${REAL_LN}" "$@"\n',
        encoding="utf-8",
    )
    ln_stub.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(BOOTSTRAP)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LN_COUNTER": str(counter),
            "REAL_LN": real_ln,
            "REPO_ROOT": str(repo_root),
            "WEBSHOP_SHARED_ROOT": str(shared_root),
            "WEBSHOP_LOCAL_ROOT": str(local_root),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 42
    assert not any((local_root / relative_path).is_symlink() for relative_path in ASSETS)
