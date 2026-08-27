#!/usr/bin/env bash
set -euo pipefail

export MODEL_ROOT="${MODEL_ROOT:-/data/zhangdw12/models}"
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/data/verl-agent}"
MODEL_SIZE="${MODEL_SIZE:-$(basename "$(dirname "${BASH_SOURCE[0]}")" | sed 's/^seed_trainer_//')}"
case "${MODEL_SIZE}" in
  1.5b) MODEL_LABEL=1.5B; TP=1; ACTOR_MICRO=16; LOGPROB_MICRO=32; OPTIMIZER_OFFLOAD=False; GPU_UTIL=0.6 ;;
  3b) MODEL_LABEL=3B; TP=2; ACTOR_MICRO=8; LOGPROB_MICRO=16; OPTIMIZER_OFFLOAD=False; GPU_UTIL=0.6 ;;
  7b) MODEL_LABEL=7B; TP=4; ACTOR_MICRO=8; LOGPROB_MICRO=8; OPTIMIZER_OFFLOAD=True; GPU_UTIL=0.45 ;;
  *) echo "Unsupported model size: ${MODEL_SIZE}" >&2; exit 2 ;;
esac
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-${MODEL_LABEL}-Instruct}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
VALIDATION_CONCURRENCY="${VALIDATION_CONCURRENCY:-128}"
cd "${REPO_ROOT}"

args=(
  algorithm.adv_estimator=seed algorithm.use_kl_in_reward=False algorithm.gamma=0.95
  algorithm.trajectory_grpo.scheduler=row algorithm.trajectory_grpo.reducer=token_mean
  algorithm.trajectory_grpo.advantage=step_row algorithm.trajectory_grpo.penalty=step_local
  algorithm.trajectory_grpo.filter=off
  algorithm.seed.mode=mean_norm algorithm.seed.skill_mode=episode_only
  algorithm.seed.skill_teacher_mode=step_priority algorithm.seed.step_advantage_w=0.0
  algorithm.seed.episode_skill_teacher_advantage_w=0.0 algorithm.seed.step_skill_teacher_advantage_w=0.0
  algorithm.seed.opd_start_after_steps=null algorithm.seed.opd_stop_after_steps=null
  algorithm.seed.failed_only=False algorithm.seed.failed_only_after_steps=null
  algorithm.seed.failure_success_threshold=1.0 algorithm.seed.enable_analysis=True
  algorithm.seed.selector=llm algorithm.seed.analysis_backend=policy_vllm
  algorithm.seed.analysis_num_workers=1 algorithm.seed.analysis_context_length=16384
  algorithm.seed.analysis_max_completion_tokens=4096
  algorithm.seed.analysis_prompt_version=seed algorithm.seed.analysis_include_episode_summary=True
  algorithm.seed.normalize_teacher_adv=False
  "data.train_files=${DATA_ROOT}/text/train.parquet" "data.val_files=${DATA_ROOT}/text/test.parquet"
  data.seed=0 data.train_batch_size=16 data.val_batch_size=128 data.max_prompt_length=4096
  data.max_response_length=512 data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True
  "actor_rollout_ref.model.path=${MODEL_PATH}" actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.ppo_mini_batch_size=64 "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO}"
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.shuffle=False
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.opd_loss_coef=0.01 actor_rollout_ref.actor.opd_gate_beta=5.0
  actor_rollout_ref.actor.use_invalid_action_penalty=True actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
  actor_rollout_ref.actor.fsdp_config.param_offload=False "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.rollout.name=vllm ++actor_rollout_ref.rollout.seed=0
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP}" "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL}" actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.max_model_len=20480 actor_rollout_ref.rollout.max_num_batched_tokens=20480
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=1 "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True reward_model.enable=False reward_model.reward_manager=episode
  env.env_name=Webshop env.fairness=true env.seed=0 env.history_length=2 env.max_steps=15 env.rollout.n=8
  env.resources_per_worker.num_cpus=0.1 "env.validation_concurrency=${VALIDATION_CONCURRENCY}"
  ++env.webshop.observation_mode=text
  trainer.critic_warmup=0 "trainer.logger=['console','swanlab']" trainer.project_name=iclr27_webshop
  "trainer.experiment_name=seed_qwen2.5_${MODEL_SIZE}_wo_sft" trainer.n_gpus_per_node=4 trainer.nnodes=1
  trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep=2 trainer.max_critic_ckpt_to_keep=2
  trainer.save_freq=10 trainer.test_freq=5 trainer.total_epochs=150 trainer.total_training_steps=150
  trainer.val_before_train=true
)
if [[ "${LAUNCHER_DRY_RUN:-false}" == true ]]; then printf '%s\n' "${args[@]}" "$@"; exit 0; fi
bash "${REPO_ROOT}/scripts/bootstrap_webshop_data.sh"
"${PYTHON_BIN}" -m examples.data_preprocess.prepare --mode text --local_dir "${DATA_ROOT}" --train_data_size 16 --val_data_size 128
exec "${PYTHON_BIN}" -m verl.trainer.main_ppo "${args[@]}" "$@"
