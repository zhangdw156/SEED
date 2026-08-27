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

ALFWorld defaults to:

- Python: `/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/bin/python3`
- raw data: `${HOME}/.cache/alfworld`
- fairness manifests: `${VERL_AGENT_FAIRNESS_CACHE:-~/.cache/verl-agent/fairness}`
- optional glibc shim under the same runtime root, loaded only when present

WebShop uses `python3` from the current shell and never activates conda/mamba.
Prepare and activate the intended mamba environment before launching. WebShop
data and indexes remain under the bundled repository-local environment package.
