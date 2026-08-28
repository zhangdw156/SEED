# Copyright 2026 The SEED team.
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
"""Shared vLLM rollout configuration helpers."""

_ENGINE_ONLY_CONFIG_KEYS = frozenset({"seed"})


def resolve_vllm_engine_seed(
    config,
    engine_kwargs=None,
    *,
    default=0,
    offset=0,
):
    """Resolve one engine seed and remove duplicate engine kwargs.

    The top-level rollout seed is canonical. A legacy
    ``engine_kwargs.vllm.seed`` remains a fallback when the top-level key is
    absent.
    """

    engine_kwargs_seed = None
    if engine_kwargs is not None:
        engine_kwargs_seed = engine_kwargs.pop("seed", None)

    if "seed" in config and config.get("seed") is not None:
        seed = config.get("seed")
    elif engine_kwargs_seed is not None:
        seed = engine_kwargs_seed
    else:
        seed = default
    return int(seed) + int(offset)


def build_vllm_sampling_params_kwargs(
    config,
    sampling_params_cls,
    **defaults,
):
    """Build request sampling kwargs without reusing engine-only settings.

    ``rollout.seed`` initializes the vLLM engine RNG. Forwarding the same value
    to ``SamplingParams.seed`` would reinitialize every request with an
    identical random stream and collapse grouped rollout diversity.
    """

    kwargs = {
        str(key): value
        for key, value in defaults.items()
        if str(key) not in _ENGINE_ONLY_CONFIG_KEYS
    }
    sampling_params = sampling_params_cls()
    for raw_key in config.keys():
        key = str(raw_key)
        if key in _ENGINE_ONLY_CONFIG_KEYS:
            continue
        if hasattr(sampling_params, key):
            kwargs[key] = config.get(raw_key)
    return kwargs
