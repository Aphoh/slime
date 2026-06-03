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

    env.setdefault(
        "SWEPRO_RUN_ID",
        f"gcp02_dynamo_trace_8gpu_tp1_pp1_cp8_ep8_131k_16kpg_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )

    defaults = {
        "DYNAMO_FRONTEND_URL": "http://warnold-swepro-frontend:3000",
        "SWEPRO_PROMPT_DATA": "/data/swebench-pro/swebench_pro_train_cached_images.jsonl",
        "SWEPRO_REF_LOAD": "/data/swebench-pro/checkpoints/GLM-4.7-Flash_torch_dist_tp1_pp1_cp8_ep8_shared_v4",
        "SWEPRO_ROLLOUT_FUNCTION_PATH": "slime.rollout.fully_async_rollout.generate_rollout_fully_async",
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
        "SWEPRO_QKV_FORMAT": "thd",
        "SWEPRO_USE_DYNAMIC_BATCH_SIZE": "0",
        "SWEPRO_MICRO_BATCH_SIZE": "1",
        "SWEPRO_MOE_TOKEN_DISPATCHER_TYPE": "alltoall",
        "SWEPRO_SEQ_LENGTH": "131072",
        "SWEPRO_MAX_CONTEXT_LEN": "131072",
        "SWEPRO_MAX_RESPONSE_LEN": "131072",
        "SWEPRO_ROLLOUT_NUM_GPUS": "1",
        "SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
        "SWEPRO_MAX_TOKENS_PER_GPU": "16384",
        "SWEPRO_LOG_PROBS_CHUNK_SIZE": "512",
        "SWEPRO_OPTIMIZER_CPU_OFFLOAD": "1",
        "SWEPRO_OVERLAP_CPU_OPTIMIZER_D2H_H2D": "1",
        "SWEPRO_UPDATE_WEIGHTS_INTERVAL": "2",
        "SWEPRO_MAX_TOOL_CALLS": "0",
        "SWEPRO_EPISODE_WALL_TIMEOUT": "0",
        "SWEPRO_TURN_MAX_TOKENS": "0",
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


@dataclass(frozen=True)
class TachometerConfig:
    enabled: bool
    binary: str
    frequency: float
    rows_per_parquet: int
    save_interval_secs: int
    sync_interval_secs: int
    discovery_timeout: int
    dyn_system_port: int
    nixl_port: int
    expected_engines: int


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
    log_probs_chunk_size = env_int(env, "SWEPRO_LOG_PROBS_CHUNK_SIZE", 512)
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


def tachometer_config(env: dict[str, str]) -> TachometerConfig:
    rollout_gpus = env_int(env, "SWEPRO_ROLLOUT_NUM_GPUS", 4)
    gpus_per_engine = max(1, env_int(env, "SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE", 4))
    expected_engines = max(1, rollout_gpus // gpus_per_engine)
    return TachometerConfig(
        enabled=env_flag(env, "SWEPRO_TACHOMETER_ENABLED", True),
        binary=env_str(env, "SWEPRO_TACHOMETER_BIN", "tachometer-scraper"),
        frequency=float(env_str(env, "SWEPRO_TACHOMETER_FREQ", "0.5")),
        rows_per_parquet=env_int(env, "SWEPRO_TACHOMETER_ROWS_PER_PARQUET", 1_000_000),
        save_interval_secs=env_int(env, "SWEPRO_TACHOMETER_SAVE_INTERVAL_SECS", 5),
        sync_interval_secs=env_int(env, "SWEPRO_TACHOMETER_SYNC_INTERVAL_SECS", 0),
        discovery_timeout=env_int(env, "SWEPRO_TACHOMETER_DISCOVERY_TIMEOUT", 600),
        dyn_system_port=env_int(env, "SWEPRO_DYNAMO_WORKER_SYSTEM_PORT", 30001),
        nixl_port=env_int(env, "SWEPRO_TACHOMETER_NIXL_PORT", 19090),
        expected_engines=expected_engines,
    )


def runtime_env(env: dict[str, str], repo_root: Path, has_nvlink: str) -> dict[str, dict[str, str]]:
    swepro_dir = str(repo_root / "examples/swebench-pro")
    fully_async_dir = str(repo_root / "examples/fully_async")
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
        "NCCL_IB_HCA": env_str(env, "NCCL_IB_HCA", ""),
        "NCCL_CROSS_NIC": env_str(env, "NCCL_CROSS_NIC", ""),
        "NCCL_IB_MERGE_NICS": env_str(env, "NCCL_IB_MERGE_NICS", ""),
        "NCCL_P2P_PXN_LEVEL": env_str(env, "NCCL_P2P_PXN_LEVEL", ""),
        "NCCL_SOCKET_IFNAME": env_str(env, "NCCL_SOCKET_IFNAME", "eth0"),
        "GLOO_SOCKET_IFNAME": env_str(env, "GLOO_SOCKET_IFNAME", "eth0"),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": env_str(env, "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
        "NCCL_DEBUG": env_str(env, "NCCL_DEBUG", ""),
        "NCCL_DEBUG_SUBSYS": env_str(env, "NCCL_DEBUG_SUBSYS", ""),
        "NCCL_DEBUG_FILE": env_str(env, "NCCL_DEBUG_FILE", ""),
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
        "SLIME_SKIP_WEIGHT_UPDATES": env_str(env, "SLIME_SKIP_WEIGHT_UPDATES", ""),
        "SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET": env_str(env, "SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET", "1"),
        "SLIME_WEIGHT_UPDATE_POST_PROCESS_WEIGHTS": env_str(env, "SLIME_WEIGHT_UPDATE_POST_PROCESS_WEIGHTS", ""),
        "SLIME_WEIGHT_UPDATE_SINGLE_TENSOR_BUCKETS": env_str(env, "SLIME_WEIGHT_UPDATE_SINGLE_TENSOR_BUCKETS", ""),
        "SLIME_WEIGHT_UPDATE_DEBUG_MANIFEST": env_str(env, "SLIME_WEIGHT_UPDATE_DEBUG_MANIFEST", ""),
        "SLIME_WEIGHT_UPDATE_SLOW_MANIFEST_SECONDS": env_str(
            env, "SLIME_WEIGHT_UPDATE_SLOW_MANIFEST_SECONDS", ""
        ),
        "SLIME_WEIGHT_UPDATE_SLOW_MANIFEST_LIMIT": env_str(env, "SLIME_WEIGHT_UPDATE_SLOW_MANIFEST_LIMIT", ""),
        "SLIME_WEIGHT_UPDATE_VALIDATE_EACH_GROUP": env_str(env, "SLIME_WEIGHT_UPDATE_VALIDATE_EACH_GROUP", ""),
        "SLIME_WEIGHT_UPDATE_DIRECT_EXPERTS": env_str(env, "SLIME_WEIGHT_UPDATE_DIRECT_EXPERTS", ""),
        "SLIME_WEIGHT_UPDATE_EXPERT_PRE_BROADCAST_BARRIER": env_str(
            env, "SLIME_WEIGHT_UPDATE_EXPERT_PRE_BROADCAST_BARRIER", ""
        ),
        "SLIME_WEIGHT_UPDATE_PER_ENGINE_GROUPS": env_str(env, "SLIME_WEIGHT_UPDATE_PER_ENGINE_GROUPS", "1"),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_P2P_PXN_LEVEL": env_str(
            env, "SLIME_WEIGHT_UPDATE_NCCL_P2P_PXN_LEVEL", ""
        ),
        "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_ENABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_ENABLE", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_HOST_ENABLE": env_str(
            env, "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_HOST_ENABLE", ""
        ),
        "SLIME_WEIGHT_UPDATE_NCCL_GRAPH_MIXING_SUPPORT": env_str(
            env, "SLIME_WEIGHT_UPDATE_NCCL_GRAPH_MIXING_SUPPORT", ""
        ),
        "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE", "0"),
        "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL": env_str(env, "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_NET": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_NET", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_DEBUG", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS", ""),
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_FILE": env_str(env, "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_FILE", ""),
        "SLIME_DYNAMO_ENGINE_ROUTE_LOGGING": env_str(env, "SLIME_DYNAMO_ENGINE_ROUTE_LOGGING", ""),
        "SWEPRO_RUN_ID": env_str(env, "SWEPRO_RUN_ID", default_run_id()),
        "DYNAMO_FRONTEND_URL": env_str(env, "DYNAMO_FRONTEND_URL", "http://warnold-swepro-frontend:3000"),
        "SWEPRO_DYNAMO_FRONTEND_URL": env.get("SWEPRO_DYNAMO_FRONTEND_URL", ""),
        "SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT": env.get("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT", ""),
        "SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT": env_str(env, "SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT", "20390"),
        "SWEPRO_NATS_URL": env_str(env, "SWEPRO_NATS_URL", "nats://warnold-swepro-nats:4222"),
        "SWEPRO_AGENT_MODE": env_str(env, "SWEPRO_AGENT_MODE", "sweagent_session"),
        "SWEPRO_MAX_TOOL_CALLS": env_str(env, "SWEPRO_MAX_TOOL_CALLS", "0"),
        "SWEPRO_EPISODE_WALL_TIMEOUT": env_str(env, "SWEPRO_EPISODE_WALL_TIMEOUT", "0"),
        "SWEPRO_TURN_MAX_TOKENS": env_str(env, "SWEPRO_TURN_MAX_TOKENS", "0"),
        "SWEPRO_MODEL": env_str(env, "SWEPRO_MODEL", "/data/glm-4.7-30b-a3b"),
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
        "SWEPRO_ROLLOUT_WITH_MOCK_TRAINER": env.get("SWEPRO_ROLLOUT_WITH_MOCK_TRAINER", ""),
        "SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND": env.get("SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND", ""),
        "SWEPRO_MOCK_WEIGHT_UPDATE_SECONDS": env.get("SWEPRO_MOCK_WEIGHT_UPDATE_SECONDS", ""),
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
    mock_trainer_enabled = env_flag(env, "SWEPRO_ROLLOUT_WITH_MOCK_TRAINER", False)
    if mock_trainer_enabled:
        debug_args.append("--debug-rollout-only")
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
    optimizer_cpu_offload = env_flag(env, "SWEPRO_OPTIMIZER_CPU_OFFLOAD", True)
    if env_flag(env, "SWEPRO_USE_PRECISION_AWARE_OPTIMIZER", False) or optimizer_cpu_offload:
        optimizer_args.append("--use-precision-aware-optimizer")
    if optimizer_cpu_offload:
        optimizer_args.append("--optimizer-cpu-offload")
    if env_flag(env, "SWEPRO_USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD", False):
        optimizer_args.append("--use-torch-optimizer-for-cpu-offload")
    if env_flag(env, "SWEPRO_OVERLAP_CPU_OPTIMIZER_D2H_H2D", optimizer_cpu_offload):
        optimizer_args.append("--overlap-cpu-optimizer-d2h-h2d")
    if exp_avg_dtype := env.get("SWEPRO_EXP_AVG_DTYPE"):
        optimizer_args.extend(["--exp-avg-dtype", exp_avg_dtype])
    if exp_avg_sq_dtype := env.get("SWEPRO_EXP_AVG_SQ_DTYPE"):
        optimizer_args.extend(["--exp-avg-sq-dtype", exp_avg_sq_dtype])

    mock_trainer_args: list[str] = []
    if mock_trainer_tps := env.get("SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND"):
        mock_trainer_args.extend(["--mock-trainer-tokens-per-second", mock_trainer_tps])
    if mock_weight_update_seconds := env.get("SWEPRO_MOCK_WEIGHT_UPDATE_SECONDS"):
        mock_trainer_args.extend(["--mock-weight-update-seconds", mock_weight_update_seconds])

    rollout_function_path = env_str(
        env,
        "SWEPRO_ROLLOUT_FUNCTION_PATH",
        "slime.rollout.fully_async_rollout.rollout_with_mock_trainer"
        if mock_trainer_enabled
        else "slime.rollout.sglang_rollout.generate_rollout",
    )

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
        rollout_function_path,
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
        env_str(env, "SWEPRO_MOE_TOKEN_DISPATCHER_TYPE", "alltoall"),
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
        *mock_trainer_args,
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
if [ -d /data/swebench-pro/pod-logs ]; then
  mkdir -p "$log_dir/pod-logs"
  cp -a /data/swebench-pro/pod-logs/. "$log_dir/pod-logs/" 2>/dev/null || true
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


def tachometer_collector_script(
    durable_logs: DurableLogConfig,
    tachometer: TachometerConfig,
    ray_address: str,
    frontend_url: str,
) -> str:
    script = r"""
set -u
log_dir=@@LOG_DIR@@
job_id=@@JOB_ID@@
run_id=@@RUN_ID@@
address=@@RAY_ADDRESS@@
frontend_url=@@FRONTEND_URL@@
tachometer_bin=@@TACHOMETER_BIN@@
expected_engines=@@EXPECTED_ENGINES@@
dyn_system_port=@@DYN_SYSTEM_PORT@@
nixl_port=@@NIXL_PORT@@
frequency=@@FREQUENCY@@
rows_per_parquet=@@ROWS_PER_PARQUET@@
save_interval_secs=@@SAVE_INTERVAL_SECS@@
sync_interval_secs=@@SYNC_INTERVAL_SECS@@
discovery_timeout=@@DISCOVERY_TIMEOUT@@
metrics_dir="$log_dir/metrics"
config_path="$metrics_dir/tachometer-scraper.toml"
storage_path="$metrics_dir/tachometer-data"
local_dir="$metrics_dir/tachometer-local"
scraper_log="$metrics_dir/tachometer-scraper.log"
discovery_json="$metrics_dir/dynamo-workers.json"
mkdir -p "$metrics_dir"
echo "===== $(date -Is) starting tachometer collector for $job_id =====" >> "$scraper_log"

if ! command -v "$tachometer_bin" >/dev/null 2>&1; then
  echo "tachometer binary not found in PATH: $tachometer_bin" >> "$scraper_log"
  exit 0
fi

rm -rf "$storage_path" "$local_dir"

RUN_ID="$run_id" \
FRONTEND_URL="$frontend_url" \
EXPECTED_ENGINES="$expected_engines" \
DYN_SYSTEM_PORT="$dyn_system_port" \
NIXL_PORT="$nixl_port" \
FREQUENCY="$frequency" \
ROWS_PER_PARQUET="$rows_per_parquet" \
SAVE_INTERVAL_SECS="$save_interval_secs" \
DISCOVERY_TIMEOUT="$discovery_timeout" \
CONFIG_PATH="$config_path" \
STORAGE_PATH="$storage_path" \
DISCOVERY_JSON="$discovery_json" \
python3 - <<'PY' >> "$scraper_log" 2>&1
import json
import os
import time
import urllib.error
import urllib.request


def quote(value):
    return json.dumps(str(value))


def parse_tcp_host(transport_tcp):
    return transport_tcp.split("/", 1)[0].rsplit(":", 1)[0]


run_id = os.environ["RUN_ID"]
frontend_url = os.environ["FRONTEND_URL"].rstrip("/")
expected = int(os.environ["EXPECTED_ENGINES"])
dyn_system_port = int(os.environ["DYN_SYSTEM_PORT"])
nixl_port = int(os.environ["NIXL_PORT"])
frequency = float(os.environ["FREQUENCY"])
rows_per_parquet = int(os.environ["ROWS_PER_PARQUET"])
save_interval_secs = int(os.environ["SAVE_INTERVAL_SECS"])
deadline = time.time() + int(os.environ["DISCOVERY_TIMEOUT"])

workers = {}
last_current = {}
last_seen = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{frontend_url}/health", timeout=5) as resp:
            health = json.load(resp)
    except Exception as exc:
        print(f"waiting for frontend health: {exc}")
        time.sleep(2)
        continue

    current = {}
    for inst in health.get("instances", []):
        if inst.get("endpoint") != "generate":
            continue
        instance_id = inst.get("instance_id")
        tcp = (inst.get("transport") or {}).get("tcp")
        if instance_id is None or not tcp:
            continue
        current[str(instance_id)] = parse_tcp_host(tcp)
    last_current = current

    if len(current) != last_seen:
        print(f"frontend reports {len(current)} generate workers, want {expected}: {current}")
        last_seen = len(current)
    if len(current) >= expected:
        workers = dict(sorted(current.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))
        break
    time.sleep(2)

if not workers:
    workers = dict(sorted(last_current.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))
    if workers:
        print(f"warning: only discovered {len(workers)} of {expected} expected Dynamo workers before timeout; collecting partial worker metrics")
    else:
        print("warning: no Dynamo workers discovered before timeout; collecting frontend metrics only")

with open(os.environ["DISCOVERY_JSON"], "w") as f:
    json.dump({"frontend_url": frontend_url, "workers": workers, "dyn_system_port": dyn_system_port, "nixl_port": nixl_port}, f, indent=2)

with open(os.environ["CONFIG_PATH"], "w") as f:
    f.write(f"storage = {quote(os.environ['STORAGE_PATH'])}\n")
    f.write(f"rows_per_parquet = {rows_per_parquet}\n")
    f.write(f"save_interval_secs = {save_interval_secs}\n\n")

    def endpoint(name, url, filter_name, metadata):
        f.write("[[endpoints]]\n")
        f.write(f"name = {quote(name)}\n")
        f.write(f"url = {quote(url)}\n")
        f.write(f"frequency = {frequency}\n")
        f.write(f"filter = {quote(filter_name)}\n\n")
        f.write("[endpoints.node_metadata]\n")
        for key, value in sorted(metadata.items()):
            f.write(f"{quote(key)} = {quote(value)}\n")
        f.write("\n")

    endpoint(
        "dynamo_frontend",
        f"{frontend_url}/metrics",
        "frontend",
        {"run_id": run_id, "node": "warnold-swepro-frontend", "index": "0", "component": "frontend"},
    )
    endpoint(
        "ray_head",
        "http://warnold-swepro-trainer:8265/metrics",
        "backend",
        {"run_id": run_id, "node": "warnold-swepro-trainer", "endpoint": "trainer_ray", "endpoint_index": "0", "component": "trainer"},
    )
    for idx, (instance_id, host) in enumerate(workers.items()):
        endpoint(
            f"dynamo_engine_{idx}",
            f"http://{host}:{dyn_system_port}/metrics",
            "backend",
            {"run_id": run_id, "node": host, "endpoint": "dynamo_engine", "endpoint_index": str(idx), "instance_id": instance_id, "component": "inference"},
        )
        if nixl_port > 0:
            endpoint(
                f"dynamo_engine_{idx}_nixl",
                f"http://{host}:{nixl_port}/metrics",
                "backend",
                {"run_id": run_id, "node": host, "endpoint": "nixl", "endpoint_index": str(idx), "instance_id": instance_id, "component": "inference"},
            )

print(f"wrote tachometer config to {os.environ['CONFIG_PATH']} with {1 + 1 + len(workers) * (2 if nixl_port > 0 else 1)} endpoints")
PY

"$tachometer_bin" --config "$config_path" --local-dir "$local_dir" --sync-interval "$sync_interval_secs" >> "$scraper_log" 2>&1 &
scraper_pid=$!
echo "$scraper_pid" > "$metrics_dir/tachometer-scraper.pid"

while true; do
  sleep 15
  {
    echo "===== $(date -Is) ====="
    ray job status --address="$address" --log-style=record --log-color=false "$job_id"
  } >> "$metrics_dir/ray-status-for-metrics.log" 2>&1 || true
  if tail -40 "$metrics_dir/ray-status-for-metrics.log" | grep -Eq "SUCCEEDED|FAILED|STOPPED"; then
    break
  fi
  if ! kill -0 "$scraper_pid" >/dev/null 2>&1; then
    echo "tachometer scraper exited before Ray job reached a terminal state" >> "$scraper_log"
    exit 0
  fi
done

echo "===== $(date -Is) stopping tachometer collector for $job_id =====" >> "$scraper_log"
kill -TERM "$scraper_pid" >/dev/null 2>&1 || true
wait "$scraper_pid" || true
echo "===== $(date -Is) tachometer collector complete for $job_id =====" >> "$scraper_log"
"""
    replacements = {
        "@@LOG_DIR@@": shlex.quote(str(durable_logs.log_dir)),
        "@@JOB_ID@@": shlex.quote(durable_logs.submission_id),
        "@@RUN_ID@@": shlex.quote(durable_logs.run_id),
        "@@RAY_ADDRESS@@": shlex.quote(ray_address),
        "@@FRONTEND_URL@@": shlex.quote(frontend_url),
        "@@TACHOMETER_BIN@@": shlex.quote(tachometer.binary),
        "@@EXPECTED_ENGINES@@": str(tachometer.expected_engines),
        "@@DYN_SYSTEM_PORT@@": str(tachometer.dyn_system_port),
        "@@NIXL_PORT@@": str(tachometer.nixl_port),
        "@@FREQUENCY@@": str(tachometer.frequency),
        "@@ROWS_PER_PARQUET@@": str(tachometer.rows_per_parquet),
        "@@SAVE_INTERVAL_SECS@@": str(tachometer.save_interval_secs),
        "@@SYNC_INTERVAL_SECS@@": str(tachometer.sync_interval_secs),
        "@@DISCOVERY_TIMEOUT@@": str(tachometer.discovery_timeout),
    }
    for needle, value in replacements.items():
        script = script.replace(needle, value)
    return script


def start_tachometer_collector(
    durable_logs: DurableLogConfig,
    tachometer: TachometerConfig,
    ray_address: str,
    frontend_url: str,
) -> None:
    if not durable_logs.enabled or not tachometer.enabled:
        return

    durable_logs.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = durable_logs.log_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = metrics_dir / "tachometer-launcher.log"
    script = tachometer_collector_script(durable_logs, tachometer, ray_address, frontend_url)
    with launcher_log.open("a", encoding="utf-8") as monitor:
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
    tachometer = tachometer_config(env)

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
        if tachometer.enabled:
            print(
                "SWEPRO tachometer: "
                f"bin={tachometer.binary}, "
                f"freq={tachometer.frequency}Hz, "
                f"expected_engines={tachometer.expected_engines}, "
                f"storage={durable_logs.log_dir / 'metrics' / 'tachometer-data'}"
            )

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
        print(json.dumps({k: env[k] for k in sorted(env) if k.startswith(("DYNAMO_", "SWEPRO_", "SLIME_"))}, indent=2))
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
        start_tachometer_collector(
            durable_logs,
            tachometer,
            args.ray_address,
            env_str(env, "DYNAMO_FRONTEND_URL", "http://warnold-swepro-frontend:3000"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
