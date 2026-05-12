#!/usr/bin/env python3
"""Launch the SWE-bench Pro RL Ray job with explicit, testable config."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def env_flag(env: dict[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value not in {"0", "false", "False", "no", "No"}


def env_int(env: dict[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_str(env: dict[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or value == "" else value


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"swepro_{stamp}"


def ray_submission_id(run_id: str) -> str:
    submission_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-")
    return submission_id or default_run_id()


def detect_nvlink() -> str:
    try:
        topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "0"
    return "1" if len(re.findall(r"NV[0-9][0-9]*", topo)) > 0 else "0"


def load_model_args(repo_root: Path, model_script: str) -> list[str]:
    script = repo_root / model_script
    if not script.exists():
        raise FileNotFoundError(f"model args script not found: {script}")
    bash = f"source {shlex.quote(str(script))}; printf '%s\\0' \"${{MODEL_ARGS[@]}}\""
    output = subprocess.check_output(["bash", "-lc", bash])
    return [part.decode() for part in output.split(b"\0") if part]


def apply_profile_defaults(env: dict[str, str], profile: str, repo_root: Path, *, write_state: bool) -> None:
    if profile == "env":
        return
    if profile != "speedscope-current":
        raise ValueError(f"unknown profile: {profile}")

    run_id = env.setdefault(
        "SWEPRO_RUN_ID",
        f"gcp02_speedscope_8gpu_tp1_pp1_cp8_ep8_131k_16kpg_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )

    trace_dir = repo_root / ".traces"
    if write_state:
        trace_dir.mkdir(parents=True, exist_ok=True)
        for stale_trace in trace_dir.glob("swepro-current.spans*.jsonl"):
            stale_trace.unlink()
        (trace_dir / "current-run-id.txt").write_text(f"{run_id}\n")

    defaults = {
        "SLIME_SPEEDSCOPE_TRACE_PATH": str(trace_dir / "swepro-current.spans.jsonl"),
        "SWEPRO_SPEEDSCOPE_TRACE_PATH": str(trace_dir / "swepro-current.spans.jsonl"),
        "SWEPRO_MODEL_TRACE_PATH": f"/data/swebench-pro/traces/{run_id}.model.jsonl",
        "SWEPRO_PROMPT_DATA": "/data/swebench-pro/swebench_pro_train_cached_images.jsonl",
        "SWEPRO_REF_LOAD": "/data/swebench-pro/checkpoints/GLM-4.7-Flash_torch_dist_tp1_pp1_cp8_ep8_shared_v4",
        "SWEPRO_ROLLOUT_FUNCTION_PATH": "fully_async_rollout.generate_rollout_fully_async",
        "SWEPRO_DISABLE_SAVE": "1",
        "SWEPRO_NUM_ROLLOUT": "10",
        "SWEPRO_ROLLOUT_BATCH_SIZE": "16",
        "SWEPRO_OVER_SAMPLING_BATCH_SIZE": "16",
        "SWEPRO_N_SAMPLES_PER_PROMPT": "4",
        "SWEPRO_GLOBAL_BATCH_SIZE": "64",
        "SWEPRO_ASYNC_MAX_INFLIGHT": "16",
        "SWEPRO_ASYNC_GROUP_MAX_ATTEMPTS": "2",
        "SWEPRO_ACTOR_NUM_NODES": "2",
        "SWEPRO_ACTOR_NUM_GPUS_PER_NODE": "4",
        "SWEPRO_TP": "1",
        "SWEPRO_PP": "1",
        "SWEPRO_CP": "8",
        "SWEPRO_EP": "8",
        "SWEPRO_ETP": "1",
        "SWEPRO_QKV_FORMAT": "bshd",
        "SWEPRO_USE_DYNAMIC_BATCH_SIZE": "0",
        "SWEPRO_MICRO_BATCH_SIZE": "1",
        "SWEPRO_MOE_TOKEN_DISPATCHER_TYPE": "allgather",
        "SWEPRO_SEQ_LENGTH": "131072",
        "SWEPRO_MAX_CONTEXT_LEN": "131072",
        "SWEPRO_MAX_RESPONSE_LEN": "131072",
        "SWEPRO_ROLLOUT_NUM_GPUS": "1",
        "SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
        "SWEPRO_MAX_TOKENS_PER_GPU": "16384",
        "SWEPRO_LOG_PROBS_CHUNK_SIZE": "512",
        "SWEPRO_UPDATE_WEIGHTS_INTERVAL": "2",
        "SWEPRO_MAX_TOOL_CALLS": "0",
        "SWEPRO_EPISODE_WALL_TIMEOUT": "0",
        "SWEPRO_TURN_MAX_TOKENS": "8192",
        "SWEPRO_MODEL_CALL_TIMEOUT": "1800",
        "SWEPRO_REQUEST_TIMEOUT": "1800",
        "SWEPRO_REQUEST_RETRIES": "5",
        "SWEPRO_SESSION_START_TIMEOUT": "1200",
        "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT": "300",
        "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT": "300",
        "SWEPRO_SESSION_CLOSE_TIMEOUT": "120",
        "SWEPRO_SESSION_HEALTH_TIMEOUT": "30",
        "SWEPRO_SESSION_ROLLOUT_RETRIES": "1",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)


@dataclass(frozen=True)
class DerivedConfig:
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    tp: int
    pp: int
    cp: int
    ep: int
    etp: int
    max_tokens_per_gpu: int
    dynamic_batch_token_limit: int
    requested_context_len: int
    rollout_max_context_len: int
    requested_response_len: int
    rollout_max_response_len: int
    seq_length: int
    log_probs_chunk_size: int


@dataclass(frozen=True)
class DurableLogConfig:
    enabled: bool
    run_id: str
    submission_id: str
    log_dir: Path
    poll_seconds: int


def derive_config(env: dict[str, str]) -> DerivedConfig:
    actor_num_nodes = env_int(env, "SWEPRO_ACTOR_NUM_NODES", 1)
    actor_num_gpus_per_node = env_int(env, "SWEPRO_ACTOR_NUM_GPUS_PER_NODE", 4)
    tp = env_int(env, "SWEPRO_TP", 4)
    pp = env_int(env, "SWEPRO_PP", 1)
    cp = env_int(env, "SWEPRO_CP", 1)
    ep = env_int(env, "SWEPRO_EP", 4)
    etp = env_int(env, "SWEPRO_ETP", 1)
    max_tokens_per_gpu = env_int(env, "SWEPRO_MAX_TOKENS_PER_GPU", 65536)
    dynamic_batch_token_limit = max_tokens_per_gpu * cp
    requested_context_len = env_int(env, "SWEPRO_MAX_CONTEXT_LEN", dynamic_batch_token_limit)
    rollout_max_context_len = min(requested_context_len, dynamic_batch_token_limit)
    requested_response_len = env_int(env, "SWEPRO_MAX_RESPONSE_LEN", rollout_max_context_len)
    rollout_max_response_len = min(requested_response_len, rollout_max_context_len)
    seq_length = env_int(env, "SWEPRO_SEQ_LENGTH", rollout_max_context_len)
    log_probs_chunk_size = env_int(env, "SWEPRO_LOG_PROBS_CHUNK_SIZE", 1024)
    return DerivedConfig(
        actor_num_nodes=actor_num_nodes,
        actor_num_gpus_per_node=actor_num_gpus_per_node,
        tp=tp,
        pp=pp,
        cp=cp,
        ep=ep,
        etp=etp,
        max_tokens_per_gpu=max_tokens_per_gpu,
        dynamic_batch_token_limit=dynamic_batch_token_limit,
        requested_context_len=requested_context_len,
        rollout_max_context_len=rollout_max_context_len,
        requested_response_len=requested_response_len,
        rollout_max_response_len=rollout_max_response_len,
        seq_length=seq_length,
        log_probs_chunk_size=log_probs_chunk_size,
    )


def durable_log_config(env: dict[str, str]) -> DurableLogConfig:
    run_id = env.setdefault("SWEPRO_RUN_ID", default_run_id())
    submission_id = env_str(env, "SWEPRO_RAY_SUBMISSION_ID", ray_submission_id(run_id))
    log_dir = Path(env_str(env, "SWEPRO_DURABLE_LOG_DIR", f"/data/swebench-pro/runs/{submission_id}"))
    return DurableLogConfig(
        enabled=env_flag(env, "SWEPRO_DURABLE_LOGS", True),
        run_id=run_id,
        submission_id=submission_id,
        log_dir=log_dir,
        poll_seconds=env_int(env, "SWEPRO_DURABLE_LOG_POLL_SECONDS", 30),
    )


def runtime_env(env: dict[str, str], repo_root: Path, has_nvlink: str) -> dict[str, dict[str, str]]:
    swepro_dir = str(repo_root / "examples/swebench-pro")
    fully_async_dir = str(repo_root / "examples/fully_async")
    trace_path = env.get("SLIME_SPEEDSCOPE_TRACE_PATH", "")
    swepro_trace_path = env.get("SWEPRO_SPEEDSCOPE_TRACE_PATH", trace_path)
    env_vars = {
        "PYTHONPATH": f"/root/src/Megatron-LM/:{swepro_dir}:{fully_async_dir}:{repo_root}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": env_str(env, "NCCL_NVLS_ENABLE", has_nvlink),
        "NCCL_MNNVL_ENABLE": env_str(env, "NCCL_MNNVL_ENABLE", "0"),
        "NCCL_CUMEM_ENABLE": env_str(env, "NCCL_CUMEM_ENABLE", "1"),
        "NCCL_CUMEM_HOST_ENABLE": env_str(env, "NCCL_CUMEM_HOST_ENABLE", "1"),
        "MC_FORCE_MNNVL": env_str(env, "MC_FORCE_MNNVL", ""),
        "NCCL_IB_DISABLE": env_str(env, "NCCL_IB_DISABLE", ""),
        "NCCL_SHM_DISABLE": env_str(env, "NCCL_SHM_DISABLE", ""),
        "NCCL_P2P_DISABLE": env_str(env, "NCCL_P2P_DISABLE", ""),
        "NCCL_ALGO": env_str(env, "NCCL_ALGO", ""),
        "NCCL_PROTO": env_str(env, "NCCL_PROTO", ""),
        "NCCL_STORE_TIMEOUT": env_str(env, "NCCL_STORE_TIMEOUT", "7200"),
        "NCCL_GRAPH_MIXING_SUPPORT": env_str(env, "NCCL_GRAPH_MIXING_SUPPORT", ""),
        "NCCL_IB_GID_INDEX": env_str(env, "NCCL_IB_GID_INDEX", ""),
        "NCCL_SOCKET_IFNAME": env_str(env, "NCCL_SOCKET_IFNAME", "eth0"),
        "GLOO_SOCKET_IFNAME": env_str(env, "GLOO_SOCKET_IFNAME", "eth0"),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": env_str(env, "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
        "NCCL_DEBUG": env_str(env, "NCCL_DEBUG", ""),
        "NVIDIA_GDRCOPY": env_str(env, "NVIDIA_GDRCOPY", "1"),
        "UCX_TLS": env_str(env, "UCX_TLS", "cuda_ipc,cuda_copy,rc"),
        "UCX_IB_GID_INDEX": env_str(env, "UCX_IB_GID_INDEX", "3"),
        "UCX_RC_TIMEOUT": env_str(env, "UCX_RC_TIMEOUT", "600s"),
        "UCX_KEEPALIVE_INTERVAL": env_str(env, "UCX_KEEPALIVE_INTERVAL", "300s"),
        "NIXL_LOG_LEVEL": env_str(env, "NIXL_LOG_LEVEL", ""),
        "NIXL_TELEMETRY_ENABLE": env_str(env, "NIXL_TELEMETRY_ENABLE", ""),
        "NIXL_TELEMETRY_EXPORTER": env_str(env, "NIXL_TELEMETRY_EXPORTER", ""),
        "NIXL_TELEMETRY_PROMETHEUS_PORT": env_str(env, "NIXL_TELEMETRY_PROMETHEUS_PORT", ""),
        "RAY_DEDUP_LOGS": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SLIME_ASYNC_ALLOW_STALE_ROLLOUTS": env_str(env, "SLIME_ASYNC_ALLOW_STALE_ROLLOUTS", "1"),
        "SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET": env_str(env, "SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET", "1"),
        "SLIME_WEIGHT_UPDATE_POST_PROCESS_WEIGHTS": env_str(env, "SLIME_WEIGHT_UPDATE_POST_PROCESS_WEIGHTS", ""),
        "SLIME_WEIGHT_UPDATE_SINGLE_TENSOR_BUCKETS": env_str(env, "SLIME_WEIGHT_UPDATE_SINGLE_TENSOR_BUCKETS", ""),
        "SLIME_WEIGHT_UPDATE_DEBUG_MANIFEST": env_str(env, "SLIME_WEIGHT_UPDATE_DEBUG_MANIFEST", ""),
        "SLIME_WEIGHT_UPDATE_VALIDATE_EACH_GROUP": env_str(env, "SLIME_WEIGHT_UPDATE_VALIDATE_EACH_GROUP", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE", "0"),
        "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL": env_str(env, "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_NET": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_NET", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_DEBUG", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS", ""),
        "SLIME_SPEEDSCOPE_TRACE_PATH": trace_path,
        "SWEPRO_SPEEDSCOPE_TRACE_PATH": swepro_trace_path,
        "SWEPRO_NATS_URL": env_str(env, "SWEPRO_NATS_URL", "nats://warnold-swepro-nats:4222"),
        "SWEPRO_AGENT_MODE": env_str(env, "SWEPRO_AGENT_MODE", "sweagent_session"),
        "SWEPRO_MAX_TOOL_CALLS": env_str(env, "SWEPRO_MAX_TOOL_CALLS", "20"),
        "SWEPRO_EPISODE_WALL_TIMEOUT": env_str(env, "SWEPRO_EPISODE_WALL_TIMEOUT", "0"),
        "SWEPRO_TURN_MAX_TOKENS": env_str(env, "SWEPRO_TURN_MAX_TOKENS", "8192"),
        "SWEPRO_MODEL": env_str(env, "SWEPRO_MODEL", "/data/glm-4.7-30b-a3b"),
        "SWEPRO_MODEL_TRACE_PATH": env.get("SWEPRO_MODEL_TRACE_PATH", ""),
        "SWEPRO_MODEL_CALL_TIMEOUT": env_str(env, "SWEPRO_MODEL_CALL_TIMEOUT", "600"),
        "SWEPRO_REQUEST_TIMEOUT": env_str(env, "SWEPRO_REQUEST_TIMEOUT", "540"),
        "SWEPRO_REQUEST_RETRIES": env_str(env, "SWEPRO_REQUEST_RETRIES", "2"),
        "SWEPRO_SESSION_START_TIMEOUT": env_str(env, "SWEPRO_SESSION_START_TIMEOUT", "900"),
        "SWEPRO_SESSION_START_CALL_TIMEOUT": env_str(
            env,
            "SWEPRO_SESSION_START_CALL_TIMEOUT",
            env_str(env, "SWEPRO_SESSION_START_TIMEOUT", "900"),
        ),
        "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT": env_str(env, "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT", "180"),
        "SWEPRO_SESSION_STEP_CALL_TIMEOUT": env_str(
            env,
            "SWEPRO_SESSION_STEP_CALL_TIMEOUT",
            env_str(env, "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT", "180"),
        ),
        "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT": env_str(env, "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT", "300"),
        "SWEPRO_SESSION_SUBMIT_CALL_TIMEOUT": env_str(
            env,
            "SWEPRO_SESSION_SUBMIT_CALL_TIMEOUT",
            env_str(env, "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT", "300"),
        ),
        "SWEPRO_SESSION_CLOSE_TIMEOUT": env_str(env, "SWEPRO_SESSION_CLOSE_TIMEOUT", "60"),
        "SWEPRO_SESSION_CLOSE_CALL_TIMEOUT": env_str(
            env,
            "SWEPRO_SESSION_CLOSE_CALL_TIMEOUT",
            env_str(env, "SWEPRO_SESSION_CLOSE_TIMEOUT", "60"),
        ),
        "SWEPRO_SESSION_HEALTH_TIMEOUT": env_str(env, "SWEPRO_SESSION_HEALTH_TIMEOUT", "30"),
        "SWEPRO_SESSION_HEALTH_CALL_TIMEOUT": env_str(
            env,
            "SWEPRO_SESSION_HEALTH_CALL_TIMEOUT",
            env_str(env, "SWEPRO_SESSION_HEALTH_TIMEOUT", "30"),
        ),
        "SWEPRO_SESSION_CAPACITY_RETRY_DELAY": env_str(env, "SWEPRO_SESSION_CAPACITY_RETRY_DELAY", "10"),
        "SWEPRO_SESSION_ROLLOUT_RETRIES": env_str(env, "SWEPRO_SESSION_ROLLOUT_RETRIES", "0"),
        "SWEPRO_ASYNC_MAX_INFLIGHT": env.get("SWEPRO_ASYNC_MAX_INFLIGHT", ""),
        "SWEPRO_ASYNC_GROUP_MAX_ATTEMPTS": env_str(env, "SWEPRO_ASYNC_GROUP_MAX_ATTEMPTS", "1"),
    }
    return {
        "env_vars": {key: value for key, value in env_vars.items() if value != ""}
    }


def build_command(
    env: dict[str, str],
    repo_root: Path,
    derived: DerivedConfig,
    durable_logs: DurableLogConfig,
    extra_train_args: list[str],
    ray_address: str,
    *,
    create_dirs: bool,
) -> list[str]:
    model_args = load_model_args(repo_root, env_str(env, "SWEPRO_MODEL_ARGS_SCRIPT", "scripts/models/glm4.7-30B-A3B.sh"))

    save_args: list[str] = []
    if not env_flag(env, "SWEPRO_DISABLE_SAVE", False):
        save_dir = env_str(env, "SWEPRO_SAVE_DIR", "/data/checkpoints/glm-4.7-30b-a3b-swepro")
        if create_dirs:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_args = ["--save", save_dir, "--save-interval", env_str(env, "SWEPRO_SAVE_INTERVAL", "10000")]

    kl_args: list[str] = []
    if env_flag(env, "SWEPRO_ENABLE_KL_LOSS", False):
        kl_args = [
            "--use-kl-loss",
            "--kl-loss-coef",
            env_str(env, "SWEPRO_KL_LOSS_COEF", "0.0"),
            "--kl-loss-type",
            env_str(env, "SWEPRO_KL_LOSS_TYPE", "low_var_kl"),
        ]

    weight_backuper_args = ["--disable-weights-backuper"] if env_flag(env, "SWEPRO_DISABLE_WEIGHTS_BACKUPER", True) else []
    ray_job_submit_args = ["--no-wait"] if env_flag(env, "SWEPRO_RAY_JOB_NO_WAIT", True) else []
    debug_args: list[str] = []
    if dump_details := env.get("SWEPRO_DUMP_DETAILS"):
        debug_args.extend(["--dump-details", dump_details])
    if save_debug_rollout_data := env.get("SWEPRO_SAVE_DEBUG_ROLLOUT_DATA"):
        debug_args.extend(["--save-debug-rollout-data", save_debug_rollout_data])
    if load_debug_rollout_data := env.get("SWEPRO_LOAD_DEBUG_ROLLOUT_DATA"):
        debug_args.extend(["--load-debug-rollout-data", load_debug_rollout_data])
        if env_flag(env, "SWEPRO_LOAD_DEBUG_ROLLOUT_DATA_WITH_UPDATES", False):
            debug_args.append("--load-debug-rollout-data-with-updates")
    if load_debug_subsample := env.get("SWEPRO_LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE"):
        debug_args.extend(["--load-debug-rollout-data-subsample", load_debug_subsample])
    train_env_args: list[str] = []
    if train_env_vars := env.get("SWEPRO_TRAIN_ENV_VARS"):
        train_env_args.extend(["--train-env-vars", train_env_vars])

    defer_fp32_logits_args = ["--defer-fp32-logits"] if env_flag(env, "SWEPRO_DEFER_FP32_LOGITS", False) else []

    model_parallel_args = [
        "--tensor-model-parallel-size",
        str(derived.tp),
        "--expert-model-parallel-size",
        str(derived.ep),
        "--expert-tensor-parallel-size",
        str(derived.etp),
        "--pipeline-model-parallel-size",
        str(derived.pp),
        "--context-parallel-size",
        str(derived.cp),
    ]
    if env_flag(env, "SWEPRO_SEQUENCE_PARALLEL", True):
        model_parallel_args.insert(2, "--sequence-parallel")
    decoder_last_pipeline_layers = env.get("SWEPRO_DECODER_LAST_PIPELINE_NUM_LAYERS")
    if decoder_last_pipeline_layers:
        model_parallel_args.extend(["--decoder-last-pipeline-num-layers", decoder_last_pipeline_layers])

    qkv_args = ["--qkv-format", env_str(env, "SWEPRO_QKV_FORMAT", "thd")]
    if env_flag(env, "SWEPRO_USE_DYNAMIC_BATCH_SIZE", True):
        batch_size_args = [
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu",
            str(derived.max_tokens_per_gpu),
        ]
    else:
        batch_size_args = [
            "--micro-batch-size",
            env_str(env, "SWEPRO_MICRO_BATCH_SIZE", "1"),
        ]

    optimizer_args = [
        "--optimizer",
        "adam",
        "--lr",
        env_str(env, "SWEPRO_LR", "1e-6"),
        "--lr-decay-style",
        env_str(env, "SWEPRO_LR_DECAY_STYLE", "constant"),
        "--min-lr",
        env_str(env, "SWEPRO_MIN_LR", env_str(env, "SWEPRO_LR", "1e-6")),
        "--weight-decay",
        env_str(env, "SWEPRO_WEIGHT_DECAY", "0.1"),
        "--adam-beta1",
        env_str(env, "SWEPRO_ADAM_BETA1", "0.9"),
        "--adam-beta2",
        env_str(env, "SWEPRO_ADAM_BETA2", "0.98"),
    ]
    optimizer_cpu_offload = env_flag(env, "SWEPRO_OPTIMIZER_CPU_OFFLOAD", False)
    if env_flag(env, "SWEPRO_USE_PRECISION_AWARE_OPTIMIZER", False) or optimizer_cpu_offload:
        optimizer_args.append("--use-precision-aware-optimizer")
    if optimizer_cpu_offload:
        optimizer_args.append("--optimizer-cpu-offload")
    if env_flag(env, "SWEPRO_USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD", False):
        optimizer_args.append("--use-torch-optimizer-for-cpu-offload")
    if env_flag(env, "SWEPRO_OVERLAP_CPU_OPTIMIZER_D2H_H2D", False):
        optimizer_args.append("--overlap-cpu-optimizer-d2h-h2d")
    if exp_avg_dtype := env.get("SWEPRO_EXP_AVG_DTYPE"):
        optimizer_args.extend(["--exp-avg-dtype", exp_avg_dtype])
    if exp_avg_sq_dtype := env.get("SWEPRO_EXP_AVG_SQ_DTYPE"):
        optimizer_args.extend(["--exp-avg-sq-dtype", exp_avg_sq_dtype])

    train_args = [
        "python3",
        "train_async.py",
        *qkv_args,
        "--actor-num-nodes",
        str(derived.actor_num_nodes),
        "--actor-num-gpus-per-node",
        str(derived.actor_num_gpus_per_node),
        "--distributed-backend",
        env_str(env, "SWEPRO_DISTRIBUTED_BACKEND", "cpu:gloo,cuda:nccl"),
        *train_env_args,
        *model_args,
        "--seq-length",
        str(derived.seq_length),
        "--rollout-backend",
        "dynamo",
        "--rollout-function-path",
        env_str(env, "SWEPRO_ROLLOUT_FUNCTION_PATH", "slime.rollout.sglang_rollout.generate_rollout"),
        "--dynamo-frontend-url",
        env_str(env, "DYNAMO_FRONTEND_URL", "http://warnold-swepro-frontend:3000"),
        "--dynamo-worker-system-port",
        "30001",
        "--dynamo-frontend-wait-timeout",
        "300",
        "--rollout-num-gpus",
        env_str(env, "SWEPRO_ROLLOUT_NUM_GPUS", "4"),
        "--rollout-num-gpus-per-engine",
        env_str(env, "SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE", "4"),
        "--sglang-server-concurrency",
        env_str(env, "SWEPRO_SGLANG_SERVER_CONCURRENCY", "64"),
        "--hf-checkpoint",
        env_str(env, "SWEPRO_HF_CHECKPOINT", "/data/glm-4.7-30b-a3b"),
        "--ref-load",
        env_str(env, "SWEPRO_REF_LOAD", "/data/glm-4.7-30b-a3b_torch_dist"),
        *save_args,
        "--prompt-data",
        env_str(env, "SWEPRO_PROMPT_DATA", "/data/swebench-pro/swebench_pro_train.jsonl"),
        "--input-key",
        env_str(env, "SWEPRO_INPUT_KEY", "prompt"),
        "--label-key",
        env_str(env, "SWEPRO_LABEL_KEY", "instance_id"),
        "--rollout-shuffle",
        "--num-rollout",
        env_str(env, "SWEPRO_NUM_ROLLOUT", "1"),
        "--rollout-batch-size",
        env_str(env, "SWEPRO_ROLLOUT_BATCH_SIZE", "4"),
        "--over-sampling-batch-size",
        env_str(env, "SWEPRO_OVER_SAMPLING_BATCH_SIZE", "4"),
        "--n-samples-per-prompt",
        env_str(env, "SWEPRO_N_SAMPLES_PER_PROMPT", "1"),
        "--rollout-max-response-len",
        str(derived.rollout_max_response_len),
        "--rollout-max-context-len",
        str(derived.rollout_max_context_len),
        "--rollout-temperature",
        env_str(env, "SWEPRO_TEMPERATURE", "1"),
        "--global-batch-size",
        env_str(env, "SWEPRO_GLOBAL_BATCH_SIZE", "1"),
        "--balance-data",
        *model_parallel_args,
        "--attention-backend",
        env_str(env, "SWEPRO_ATTENTION_BACKEND", "flash"),
        "--moe-token-dispatcher-type",
        env_str(env, "SWEPRO_MOE_TOKEN_DISPATCHER_TYPE", "allgather"),
        *batch_size_args,
        "--log-probs-chunk-size",
        str(derived.log_probs_chunk_size),
        *defer_fp32_logits_args,
        *kl_args,
        *weight_backuper_args,
        "--advantage-estimator",
        "grpo",
        "--disable-rewards-normalization",
        "--update-weights-interval",
        env_str(env, "SWEPRO_UPDATE_WEIGHTS_INTERVAL", "2"),
        "--update-weight-buffer-size",
        env_str(env, "SWEPRO_UPDATE_WEIGHT_BUFFER_SIZE", str(512 * 1024 * 1024)),
        "--distributed-timeout-minutes",
        env_str(env, "SWEPRO_DISTRIBUTED_TIMEOUT_MINUTES", "30"),
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        *optimizer_args,
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--custom-generate-function-path",
        "generate_with_swebench_pro.generate",
        "--custom-rm-path",
        "generate_with_swebench_pro.reward_func",
        *debug_args,
        *extra_train_args,
    ]

    has_nvlink = detect_nvlink()
    ray_env = runtime_env(env, repo_root, has_nvlink)
    return [
        "ray",
        "job",
        "submit",
        f"--address={ray_address}",
        f"--submission-id={durable_logs.submission_id}",
        *ray_job_submit_args,
        f"--working-dir={repo_root}",
        f"--metadata-json={json.dumps({'swepro_run_id': durable_logs.run_id, 'swepro_log_dir': str(durable_logs.log_dir)}, separators=(',', ':'))}",
        f"--runtime-env-json={json.dumps(ray_env, separators=(',', ':'))}",
        "--",
        *train_args,
    ]


def tee_command(command: list[str], log_path: Path, *, env: dict[str, str], cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
        log.write(f"===== exit_code={return_code} {datetime.now(timezone.utc).isoformat()} =====\n")
        return return_code


def ray_log_follower_script(durable_logs: DurableLogConfig, ray_address: str) -> str:
    return f"""
set -u
log_dir={shlex.quote(str(durable_logs.log_dir))}
job_id={shlex.quote(durable_logs.submission_id)}
address={shlex.quote(ray_address)}
poll_seconds={durable_logs.poll_seconds}
mkdir -p "$log_dir/ray-logs"
sentinel="$log_dir/.ray-log-follower-start"
touch "$sentinel"
echo "===== $(date -Is) starting Ray log follower for $job_id ====="

(
  ray job logs --address="$address" --log-style=record --log-color=false --follow "$job_id"
) >> "$log_dir/ray-driver.log" 2>> "$log_dir/ray-log-follower.log" &
logs_pid=$!

while true; do
  {{
    echo "===== $(date -Is) ====="
    ray job status --address="$address" --log-style=record --log-color=false "$job_id"
  }} >> "$log_dir/ray-status.log" 2>&1

  if tail -40 "$log_dir/ray-status.log" | grep -Eq "SUCCEEDED|FAILED|STOPPED"; then
    break
  fi
  if ! kill -0 "$logs_pid" >/dev/null 2>&1; then
    break
  fi
  sleep "$poll_seconds"
done

wait "$logs_pid" || true
{{
  echo "===== final $(date -Is) ====="
  ray job status --address="$address" --log-style=record --log-color=false "$job_id" || true
  ray job list --address="$address" --log-style=record --log-color=false || true
}} >> "$log_dir/ray-status.log" 2>&1

if [ -d /tmp/ray/session_latest/logs ]; then
  find /tmp/ray/session_latest/logs -maxdepth 1 -type f \\
    \\( -name "job-driver-$job_id.log" -o -name "worker-*.err" -o -name "worker-*.out" -o -name "ray_process_exit.log" \\) \\
    -newer "$sentinel" -print0 | while IFS= read -r -d '' file; do
      cp -n "$file" "$log_dir/ray-logs/$(basename "$file")" 2>/dev/null || true
    done
fi
echo "===== $(date -Is) Ray log follower complete for $job_id ====="
"""


def start_ray_log_follower(durable_logs: DurableLogConfig, ray_address: str) -> None:
    if not durable_logs.enabled:
        return

    durable_logs.log_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = durable_logs.log_dir / "ray-log-follower.log"
    script = ray_log_follower_script(durable_logs, ray_address)
    with monitor_log.open("a", encoding="utf-8") as monitor:
        subprocess.Popen(
            ["bash", "-lc", script],
            stdout=monitor,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["env", "speedscope-current"], default="env")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--ray-address", default="http://127.0.0.1:8265")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-env", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def main() -> int:
    args, extra_train_args = parse_args()
    repo_root = Path(args.repo_root or os.getcwd()).resolve()
    env = dict(os.environ)
    apply_profile_defaults(env, args.profile, repo_root, write_state=not args.dry_run)
    derived = derive_config(env)
    durable_logs = durable_log_config(env)

    print(
        "SWEPRO trainer: "
        f"actor_nodes={derived.actor_num_nodes}, "
        f"gpus_per_node={derived.actor_num_gpus_per_node}, "
        f"tp={derived.tp}, pp={derived.pp}, cp={derived.cp}, ep={derived.ep}"
    )
    print(
        "SWEPRO rollout cap: "
        f"max_tokens_per_gpu={derived.max_tokens_per_gpu}, "
        f"dynamic_batch_token_limit={derived.dynamic_batch_token_limit}, "
        f"requested_context_len={derived.requested_context_len}, "
        f"rollout_max_context_len={derived.rollout_max_context_len}, "
        f"requested_response_len={derived.requested_response_len}, "
        f"rollout_max_response_len={derived.rollout_max_response_len}, "
        f"seq_length={derived.seq_length}, "
        f"log_probs_chunk_size={derived.log_probs_chunk_size}"
    )
    if args.profile != "env":
        print(f"SWEPRO run_id: {env['SWEPRO_RUN_ID']}")
    if durable_logs.enabled:
        print(f"SWEPRO durable logs: {durable_logs.log_dir}")
        print(f"SWEPRO Ray submission_id: {durable_logs.submission_id}")

    command = build_command(
        env,
        repo_root,
        derived,
        durable_logs,
        extra_train_args,
        args.ray_address,
        create_dirs=not args.dry_run,
    )
    if args.print_env:
        print(json.dumps({k: env[k] for k in sorted(env) if k.startswith(("SWEPRO_", "SLIME_"))}, indent=2))
    if args.dry_run:
        print(shlex.join(command))
        return 0
    if durable_logs.enabled:
        durable_logs.log_dir.mkdir(parents=True, exist_ok=True)
        (durable_logs.log_dir / "run_id.txt").write_text(f"{durable_logs.run_id}\n", encoding="utf-8")
        (durable_logs.log_dir / "submission_id.txt").write_text(f"{durable_logs.submission_id}\n", encoding="utf-8")
        (durable_logs.log_dir / "entrypoint.sh").write_text(f"{shlex.join(command)}\n", encoding="utf-8")
        debug_env = {
            key: env[key]
            for key in sorted(env)
            if key.startswith(
                (
                    "DYNAMO_",
                    "NCCL_",
                    "NIXL_",
                    "NVIDIA_",
                    "RAY_",
                    "SLIME_",
                    "SWEPRO_",
                    "TORCH_",
                    "UCX_",
                )
            )
        }
        (durable_logs.log_dir / "env.json").write_text(json.dumps(debug_env, indent=2) + "\n", encoding="utf-8")

    if durable_logs.enabled:
        return_code = tee_command(command, durable_logs.log_dir / "ray-submit.log", env=env, cwd=repo_root)
    else:
        return_code = subprocess.run(command, env=env, cwd=repo_root).returncode
    if return_code != 0:
        return return_code
    if durable_logs.enabled:
        start_ray_log_follower(durable_logs, args.ray_address)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
