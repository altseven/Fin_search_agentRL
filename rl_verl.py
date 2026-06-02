from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rl_common import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_PATH,
    ensure_dirs,
    log,
)
from rl_config import MVPConfig
from rl_dataset import export_verl_dataset


def is_local_model_path(model_path: str) -> bool:
    return model_path.startswith(("/", "./", "../", "model/")) or Path(model_path).exists()


def resolve_model_path_for_command(model_path: str) -> str:
    if is_local_model_path(model_path):
        return str(Path(model_path).expanduser().resolve())
    return model_path


def print_download_hints(model_path: str = DEFAULT_MODEL_PATH, model_dir: str = DEFAULT_MODEL_DIR) -> None:
    local_dir = Path(model_path) if is_local_model_path(model_path) else Path(model_dir) / "Qwen3-4B"
    local_dir = local_dir.expanduser()
    msg = f"""
Model download hints for China mainland networks:

  pip install -U modelscope
  modelscope download --model {DEFAULT_HF_MODEL_ID} --local_dir {local_dir}

Then run this script or the verl command with:
  --model-path {local_dir}
"""
    print(msg.strip())


def ensure_model_available(cfg: MVPConfig) -> None:
    if not cfg.auto_download_model:
        return
    if not is_local_model_path(cfg.model_path):
        return
    model_dir = Path(cfg.model_path).expanduser()
    if model_dir.exists() and any(model_dir.iterdir()):
        log(f"Model directory exists: {model_dir}")
        return
    log(f"Model directory missing or empty, downloading {DEFAULT_HF_MODEL_ID} to {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "modelscope"], check=True)
    subprocess.run(
        [
            "modelscope",
            "download",
            "--model",
            DEFAULT_HF_MODEL_ID,
            "--local_dir",
            str(model_dir),
        ],
        check=True,
    )


def make_verl_command(cfg: MVPConfig, train_path: Path, valid_path: Path) -> str:
    project_root = Path(__file__).resolve().parent
    tool_path = project_root / "rl_tools.py"
    reward_path = project_root / "rl_reward.py"
    repo = Path(cfg.verl_dir).expanduser().resolve()
    data_root = Path(cfg.data_dir).expanduser().resolve()
    run_root = Path(cfg.run_dir).expanduser().resolve() if cfg.run_dir else train_path.parent.parent.resolve()
    verl_file_logger_path = run_root / "verl" / "training_metrics.jsonl"
    hf_datasets_cache = run_root / "verl" / "hf_datasets_cache"
    checkpoint_dir = run_root / "verl" / "checkpoints"
    model_path = resolve_model_path_for_command(cfg.model_path)
    python_bin = Path(sys.executable).resolve()
    total_seq_len = int(cfg.max_prompt_length) + int(cfg.max_response_length)
    rollout_max_model_len = cfg.rollout_max_model_len or total_seq_len
    rollout_max_num_batched_tokens = cfg.rollout_max_num_batched_tokens or max(8192, total_seq_len * 2)
    actor_ppo_max_token_len_per_gpu = cfg.actor_ppo_max_token_len_per_gpu or total_seq_len * max(
        1, int(cfg.ppo_micro_batch_size_per_gpu)
    )
    ref_log_prob_max_token_len_per_gpu = cfg.ref_log_prob_max_token_len_per_gpu or total_seq_len * 3
    rollout_log_prob_max_token_len_per_gpu = cfg.rollout_log_prob_max_token_len_per_gpu or total_seq_len * 3
    max_assistant_turns = max(2, int(cfg.max_tool_calls) + 1)
    lora_overrides = ""
    if cfg.lora_rank > 0:
        lora_overrides = f"""  actor_rollout_ref.model.lora_rank={cfg.lora_rank} \\
  actor_rollout_ref.model.lora_alpha={cfg.lora_alpha} \\
  actor_rollout_ref.model.target_modules=all-linear \\
"""
    command = f"""#!/usr/bin/env bash
set -xeuo pipefail

export PYTHONPATH="{project_root}:{repo}:${{PYTHONPATH:-}}"
export STOCK_AGENT_DATA_DIR="{data_root}"
export HYDRA_FULL_ERROR=1
export VERL_FILE_LOGGER_PATH="{verl_file_logger_path}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export HF_HUB_DISABLE_TELEMETRY=1

cd "{repo}"
"{python_bin}" -m ray.scripts.scripts stop --force || true

"{python_bin}" -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  algorithm.use_kl_in_reward=False \\
  data.train_files="{train_path}" \\
  data.val_files="{valid_path}" \\
  data.train_batch_size={cfg.train_batch_size} \\
  data.max_prompt_length={cfg.max_prompt_length} \\
  data.max_response_length={cfg.max_response_length} \\
  data.filter_overlong_prompts=False \\
  data.truncation=left \\
  data.return_raw_chat=True \\
  +data.apply_chat_template_kwargs.enable_thinking=False \\
  actor_rollout_ref.model.path="{model_path}" \\
  +actor_rollout_ref.model.override_config.attn_implementation={cfg.attn_implementation} \\
  actor_rollout_ref.model.use_remove_padding={str(cfg.use_remove_padding).lower()} \\
  actor_rollout_ref.model.enable_gradient_checkpointing=True \\
{lora_overrides}  actor_rollout_ref.actor.optim.lr={cfg.actor_lr} \\
  actor_rollout_ref.actor.optim.weight_decay={cfg.actor_weight_decay} \\
  actor_rollout_ref.actor.optim.lr_warmup_steps={cfg.actor_lr_warmup_steps} \\
  actor_rollout_ref.actor.ppo_mini_batch_size={cfg.ppo_mini_batch_size} \\
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg.ppo_micro_batch_size_per_gpu} \\
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_ppo_max_token_len_per_gpu} \\
  actor_rollout_ref.actor.use_kl_loss=True \\
  actor_rollout_ref.actor.kl_loss_coef=0.001 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.actor.entropy_coeff=0 \\
  actor_rollout_ref.actor.fsdp_config.param_offload={str(cfg.actor_fsdp_param_offload).lower()} \\
  actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(cfg.actor_fsdp_optimizer_offload).lower()} \\
  actor_rollout_ref.actor.fsdp_config.fsdp_size={cfg.actor_fsdp_size} \\
  actor_rollout_ref.actor.ulysses_sequence_parallel_size={cfg.actor_ulysses_sequence_parallel_size} \\
  actor_rollout_ref.actor.use_dynamic_bsz=True \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.rollout_tp} \\
  actor_rollout_ref.rollout.gpu_memory_utilization={cfg.rollout_gpu_memory_utilization} \\
  actor_rollout_ref.rollout.max_model_len={rollout_max_model_len} \\
  actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens} \\
  actor_rollout_ref.rollout.enable_chunked_prefill=True \\
  actor_rollout_ref.rollout.enable_prefix_caching=True \\
  actor_rollout_ref.rollout.n={cfg.rollout_n} \\
  actor_rollout_ref.rollout.load_format=safetensors \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg.log_prob_micro_batch_size_per_gpu} \\
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={rollout_log_prob_max_token_len_per_gpu} \\
  actor_rollout_ref.rollout.multi_turn.enable=True \\
  actor_rollout_ref.rollout.multi_turn.format=hermes \\
  actor_rollout_ref.rollout.multi_turn.function_tool_path="{tool_path}" \\
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns={max_assistant_turns} \\
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=768 \\
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \\
  actor_rollout_ref.rollout.agent.num_workers={cfg.rollout_agent_num_workers} \\
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \\
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={cfg.log_prob_micro_batch_size_per_gpu} \\
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={ref_log_prob_max_token_len_per_gpu} \\
  actor_rollout_ref.ref.fsdp_config.param_offload={str(cfg.ref_fsdp_param_offload).lower()} \\
  actor_rollout_ref.ref.fsdp_config.fsdp_size={cfg.ref_fsdp_size} \\
  actor_rollout_ref.ref.ulysses_sequence_parallel_size={cfg.ref_ulysses_sequence_parallel_size} \\
  reward.custom_reward_function.path="{reward_path}" \\
  reward.custom_reward_function.name=compute_score \\
  reward.reward_manager.name=naive \\
  reward.num_workers={cfg.reward_num_workers} \\
  trainer.critic_warmup=0 \\
  trainer.logger='["console","file"]' \\
  trainer.project_name=stock_agent_rl_mvp \\
  trainer.experiment_name=qwen3_4b_sse50_grpo_full \\
  trainer.resume_mode=disable \\
  trainer.default_local_dir="{checkpoint_dir}" \\
  trainer.n_gpus_per_node={cfg.n_gpus_per_node} \\
  trainer.nnodes=1 \\
  trainer.save_freq=5 \\
  trainer.test_freq=1 \\
  trainer.total_epochs={cfg.total_epochs} \\
  "$@"
"""
    return command


def write_verl_command_script(cfg: MVPConfig) -> Path:
    dirs = ensure_dirs(cfg)
    train_path = dirs["verl"] / "train.parquet"
    valid_path = dirs["verl"] / "valid.parquet"
    if not train_path.exists() or not valid_path.exists():
        train_path, valid_path = export_verl_dataset(cfg)
    command = make_verl_command(cfg, train_path.resolve(), valid_path.resolve())
    out_path = Path(cfg.command_file).expanduser()
    out = out_path.resolve() if out_path.is_absolute() else (dirs["run"] / out_path).resolve()
    out.write_text(command, encoding="utf-8")
    out.chmod(0o755)
    log(f"Wrote verl command script: {out}")
    return out


def run_verl_command_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"verl command script not found: {script_path}")
    log(f"Launching verl training: {script_path}")
    env = os.environ.copy()
    subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"], env=env, check=False)
    subprocess.run(["bash", str(script_path)], env=env, check=True)


def find_latest_verl_command(result_dir: str, command_file: str) -> Path:
    base = Path(result_dir).expanduser().resolve()
    candidates: list[Path] = []
    if base.exists():
        for run_dir in base.glob("*_result*"):
            script = run_dir / command_file
            if script.exists():
                candidates.append(script)
    if not candidates:
        raise FileNotFoundError(f"No {command_file} found under {base}/*_result*/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
