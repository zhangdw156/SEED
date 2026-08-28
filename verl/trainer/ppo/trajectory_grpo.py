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
"""Native SEED trajectory-GRPO configuration contract."""

from collections.abc import Mapping
from typing import Any

NATIVE_TRAJECTORY_GRPO_CONFIG = {
    "scheduler": "row",
    "reducer": "token_mean",
    "advantage": "step_row",
    "penalty": "step_local",
    "filter": "off",
}


def _trajectory_config_block(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "algorithm" in config:
        algorithm = config["algorithm"]
        if not isinstance(algorithm, Mapping):
            raise ValueError("algorithm config must be a mapping")
        config = algorithm.get("trajectory_grpo", {})
    elif "trajectory_grpo" in config:
        config = config["trajectory_grpo"]
    if not isinstance(config, Mapping):
        raise ValueError("trajectory_grpo config must be a mapping")
    return config


def resolve_trajectory_grpo_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve omitted native fields and reject unsupported configuration keys."""

    trajectory_config = _trajectory_config_block(config)
    unknown_fields = set(trajectory_config) - set(NATIVE_TRAJECTORY_GRPO_CONFIG)
    if unknown_fields:
        fields = ", ".join(sorted(map(str, unknown_fields)))
        raise ValueError(f"unsupported trajectory_grpo fields: {fields}")
    return {
        **NATIVE_TRAJECTORY_GRPO_CONFIG,
        **dict(trajectory_config),
    }


def validate_trajectory_grpo_config(config: Mapping[str, Any]) -> None:
    """Require the official native SEED row/token/step/off behavior."""

    resolved = resolve_trajectory_grpo_config(config)
    for field, native_value in NATIVE_TRAJECTORY_GRPO_CONFIG.items():
        configured_value = resolved[field]
        if configured_value != native_value:
            raise ValueError(
                f"trajectory_grpo.{field} must be {native_value!r}, "
                f"got {configured_value!r}"
            )


__all__ = [
    "NATIVE_TRAJECTORY_GRPO_CONFIG",
    "resolve_trajectory_grpo_config",
    "validate_trajectory_grpo_config",
]
