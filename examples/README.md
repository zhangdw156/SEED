# ICLR SEED Fairness Experiments (without SFT)

This snapshot exposes exactly six standalone Stage-2 launchers:

- `seed_trainer_1.5b/{run_alfworld.sh,run_webshop.sh}`
- `seed_trainer_3b/{run_alfworld.sh,run_webshop.sh}`
- `seed_trainer_7b/{run_alfworld.sh,run_webshop.sh}`

All launchers start directly from the original
`Qwen2.5-{1.5B,3B,7B}-Instruct` checkpoints. They do not activate an SFT
environment or reference an SFT checkpoint. SEED still performs its official
online hindsight analysis, policy-teacher rescoring, and OPD actor update during
Stage 2.

Set `LAUNCHER_DRY_RUN=true` to print the resolved Hydra arguments without
preprocessing data or starting training. Extra CLI arguments are appended last,
so explicit overrides win.

## Rollout performance behavior

The ALFWorld and WebShop launchers share two runtime-only optimizations:

- synchronous FSDP-vLLM keeps weights, KV-cache residency, and generation RNG
  state live for one bounded environment rollout, then restores training state;
- after trajectories terminate, later generation and environment steps run only
  on the remaining `active_indices` (for example `3 -> 2 -> 1`).

Compaction preserves each original environment slot, task/gamefile, observation,
history, reward, `uid`, `traj_uid`, and SEED step metadata. Therefore online
analysis, policy-teacher rescoring, OPD, fairness assignment, persistent RNG,
and `trajectory_grpo.advantage=step_row` retain their existing semantics.
Backends without rollout-session support continue to use the legacy per-call
context path. Session cleanup failures fail closed before actor updates.

These are code-path optimizations only; this snapshot does not claim a measured
end-to-end speedup.

ALFWorld defaults to:

- Python: `/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/bin/python3`
- raw data: `${HOME}/.cache/alfworld`
- fairness manifests: `${VERL_AGENT_FAIRNESS_CACHE:-~/.cache/verl-agent/fairness}`
- optional glibc shim under the same runtime root, loaded only when present

WebShop uses `python3` from the current shell and never activates conda/mamba.
Prepare and activate the intended mamba environment before launching. WebShop
data and indexes remain under the bundled repository-local environment package.
When any required local asset is missing, the launcher links it from
`${WEBSHOP_SHARED_ROOT:-/data/zhangdw12/work/verl-agent/agent_system/environments/env_package/webshop/webshop}`.
The bootstrap leaves existing local files, directories, and links untouched,
serializes concurrent launchers, rolls back links from failed attempts, and
validates non-empty data files plus a Lucene `segments_*` file. It fails before
preprocessing when a required shared source is invalid or missing, and is
skipped entirely by `LAUNCHER_DRY_RUN=true`.
