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

"""Canonical ALFWorld and WebShop task pools used by fair experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_DATASETS = {
    "alfworld": "zhangdw/alfworld",
    "webshop": "zhangdw/webshop",
}
_FILES = {
    "alfworld": {
        "train": "train.jsonl",
        "evaluation_seen": "evaluation_seen.jsonl",
        "evaluation_unseen": "evaluation_unseen.jsonl",
    },
    "webshop": {
        "train": "train.jsonl",
        "evaluation": "evaluation.jsonl",
    },
}
_EXPECTED_COUNTS = {
    ("alfworld", "train"): 3553,
    ("alfworld", "evaluation_seen"): 140,
    ("alfworld", "evaluation_unseen"): 134,
    ("webshop", "train"): 6410,
    ("webshop", "evaluation"): 500,
}
VALIDATION_CONCURRENCY = 128


def _environment_name(environment: str) -> str:
    normalized = environment.strip().lower()
    if normalized not in _DATASETS:
        supported = ", ".join(sorted(_DATASETS))
        raise ValueError(
            f"Unsupported fairness environment {environment!r}; "
            f"expected one of: {supported}"
        )
    return normalized


def _fairness_dir(environment: str) -> Path:
    environment = _environment_name(environment)
    override = os.environ.get(f"{environment.upper()}_FAIRNESS_DIR")
    if override:
        return Path(override).expanduser()
    cache_root = Path(
        os.environ.get(
            "VERL_AGENT_FAIRNESS_CACHE",
            "~/.cache/verl-agent/fairness",
        )
    ).expanduser()
    return cache_root / environment


def _candidate_endpoints() -> list[str]:
    endpoints = []
    if configured := os.environ.get("HF_ENDPOINT"):
        endpoints.append(configured.rstrip("/"))
    endpoints.extend(("https://huggingface.co", "https://hf-mirror.com"))
    return list(dict.fromkeys(endpoints))


def _download_fairness_file(
    environment: str,
    filename: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for endpoint in _candidate_endpoints():
        url = (
            f"{endpoint}/datasets/{_DATASETS[environment]}/resolve/main/"
            f"fairness/{filename}"
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except (OSError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as writer:
            writer.write(payload)
            temporary_path = Path(writer.name)
        temporary_path.replace(destination)
        return
    raise RuntimeError(
        f"Unable to download {environment} fairness/{filename}.\n"
        + "\n".join(errors)
    )


def fairness_file(environment: str, split: str) -> Path:
    environment = _environment_name(environment)
    try:
        filename = _FILES[environment][split]
    except KeyError as exc:
        supported = ", ".join(sorted(_FILES[environment]))
        raise ValueError(
            f"Unsupported {environment} fairness split {split!r}; "
            f"expected one of: {supported}"
        ) from exc
    path = _fairness_dir(environment) / filename
    if not path.is_file():
        _download_fairness_file(environment, filename, path)
    return path


def load_fairness_rows(
    environment: str,
    split: str,
) -> list[dict[str, Any]]:
    environment = _environment_name(environment)
    path = fairness_file(environment, split)
    rows = []
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path}:{line_number}, got "
                    f"{type(row).__name__}"
                )
            rows.append(row)
    expected_count = _EXPECTED_COUNTS[(environment, split)]
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rows in {path}, found {len(rows)}"
        )
    identity_name = (
        "metadata.gamefile"
        if environment == "alfworld"
        else "metadata.goal_idx"
    )
    identity_key = "gamefile" if environment == "alfworld" else "goal_idx"
    identities = [
        row.get("metadata", {}).get(identity_key)
        for row in rows
    ]
    if any(identity is None for identity in identities):
        raise ValueError(f"Missing {identity_name} in {path}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate {identity_name} values in {path}")
    return rows


def alfworld_gamefiles(split: str) -> list[str]:
    return [
        str(row["metadata"]["gamefile"])
        for row in load_fairness_rows("alfworld", split)
    ]


def alfworld_fairness_split(
    *,
    is_train: bool,
    eval_dataset: str,
    requested_split: str | None = None,
) -> str:
    if is_train:
        return "train"
    if requested_split is not None:
        return requested_split
    if eval_dataset == "eval_out_of_distribution":
        return "evaluation_unseen"
    return "evaluation_seen"


def alfworld_worker_gamefiles(
    *,
    fixed_assignment: bool,
    canonical_gamefiles: list[str],
    num_processes: int,
) -> list[str | None]:
    if not fixed_assignment:
        return [None] * num_processes
    if num_processes != len(canonical_gamefiles):
        raise ValueError(
            "ALFWorld fairness chunk requires exactly "
            f"{len(canonical_gamefiles)} workers, got {num_processes}"
        )
    return list(canonical_gamefiles)


def webshop_goal_indices(split: str) -> list[int]:
    return [
        int(row["metadata"]["goal_idx"])
        for row in load_fairness_rows("webshop", split)
    ]


def webshop_goal_fingerprint(goals: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        goals,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def webshop_reset_goal_indices(
    *,
    is_train: bool,
    fairness_enabled: bool = True,
    goal_indices: list[int],
    env_num: int,
    group_n: int,
    rng: Any,
) -> list[int]:
    if is_train or not fairness_enabled:
        selected = [
            int(value)
            for value in rng.choice(
                goal_indices,
                size=env_num,
                replace=False,
            )
        ]
    else:
        if len(goal_indices) != env_num:
            raise ValueError(
                "WebShop fairness validation requires exactly "
                f"{len(goal_indices)} workers, got {env_num}"
            )
        selected = list(goal_indices)
    return [
        goal_idx
        for goal_idx in selected
        for _ in range(group_n)
    ]


def canonical_validation_splits(environment: str) -> tuple[str, ...]:
    environment = _environment_name(environment)
    if environment == "alfworld":
        return ("evaluation_seen", "evaluation_unseen")
    return ("evaluation",)


def canonical_validation_chunks(
    environment: str,
    split: str,
    *,
    concurrency: int = VALIDATION_CONCURRENCY,
) -> tuple[tuple[Any, ...], ...]:
    environment = _environment_name(environment)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency <= 0
        or concurrency > VALIDATION_CONCURRENCY
    ):
        raise ValueError(
            f"validation concurrency must be in [1, {VALIDATION_CONCURRENCY}]"
        )
    if split not in canonical_validation_splits(environment):
        supported = ", ".join(canonical_validation_splits(environment))
        raise ValueError(
            f"Unsupported {environment} validation split {split!r}; "
            f"expected one of: {supported}"
        )
    values = (
        alfworld_gamefiles(split)
        if environment == "alfworld"
        else webshop_goal_indices(split)
    )
    return tuple(
        tuple(values[start : start + concurrency])
        for start in range(0, len(values), concurrency)
    )
