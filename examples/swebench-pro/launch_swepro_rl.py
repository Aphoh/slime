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

    run_id = env.get("SWEPRO_RUN_ID")
    if not run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"gcp02_speedscope_8gpu_tp1_pp1_cp8_ep8_131k_16kpg_{stamp}"
        env["SWEPRO_RUN_ID"] = run_id

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


def runtime_env(env: dict[str, str], repo_root: Path, has_nvlink: str) -> dict[str, dict[str, str]]:
    swepro_dir = str(repo_root / "examples/swebench-pro")
    fully_async_dir = str(repo_root / "examples/fully_async")
    trace_path = env.get("SLIME_SPEEDSCOPE_TRACE_PATH", "")
    swepro_trace_path = env.get("SWEPRO_SPEEDSCOPE_TRACE_PATH", trace_path)
    return {
        "env_vars": {
            "PYTHONPATH": f"/root/src/Megatron-LM/:{swepro_dir}:{fully_async_dir}:{repo_root}",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": has_nvlink,
            "NCCL_MNNVL_ENABLE": env_str(env, "NCCL_MNNVL_ENABLE", "0"),
            "NCCL_NET": env_str(env, "NCCL_NET", "Socket"),
            "NCCL_IB_DISABLE": env_str(env, "NCCL_IB_DISABLE", "1"),
            "NCCL_SHM_DISABLE": env_str(env, "NCCL_SHM_DISABLE", "0"),
            "NCCL_SOCKET_IFNAME": env_str(env, "NCCL_SOCKET_IFNAME", "eth0"),
            "GLOO_SOCKET_IFNAME": env_str(env, "GLOO_SOCKET_IFNAME", "eth0"),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": env_str(env, "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
            "NCCL_DEBUG": env_str(env, "NCCL_DEBUG", "WARN"),
            "RAY_DEDUP_LOGS": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SLIME_ASYNC_ALLOW_STALE_ROLLOUTS": env_str(env, "SLIME_ASYNC_ALLOW_STALE_ROLLOUTS", "1"),
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
            "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT": env_str(env, "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT", "180"),
            "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT": env_str(env, "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT", "300"),
            "SWEPRO_SESSION_CLOSE_TIMEOUT": env_str(env, "SWEPRO_SESSION_CLOSE_TIMEOUT", "60"),
            "SWEPRO_SESSION_HEALTH_TIMEOUT": env_str(env, "SWEPRO_SESSION_HEALTH_TIMEOUT", "30"),
            "SWEPRO_SESSION_CAPACITY_RETRY_DELAY": env_str(env, "SWEPRO_SESSION_CAPACITY_RETRY_DELAY", "10"),
            "SWEPRO_SESSION_ROLLOUT_RETRIES": env_str(env, "SWEPRO_SESSION_ROLLOUT_RETRIES", "0"),
            "SWEPRO_ASYNC_MAX_INFLIGHT": env.get("SWEPRO_ASYNC_MAX_INFLIGHT", ""),
        }
    }


def build_command(
    env: dict[str, str],
    repo_root: Path,
    derived: DerivedConfig,
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
        *kl_args,
        *weight_backuper_args,
        "--advantage-estimator",
        "grpo",
        "--disable-rewards-normalization",
        "--update-weights-interval",
        env_str(env, "SWEPRO_UPDATE_WEIGHTS_INTERVAL", "2"),
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
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
        *extra_train_args,
    ]

    has_nvlink = detect_nvlink()
    ray_env = runtime_env(env, repo_root, has_nvlink)
    return [
        "ray",
        "job",
        "submit",
        f"--address={ray_address}",
        *ray_job_submit_args,
        f"--working-dir={repo_root}",
        f"--runtime-env-json={json.dumps(ray_env, separators=(',', ':'))}",
        "--",
        *train_args,
    ]


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

    command = build_command(
        env,
        repo_root,
        derived,
        extra_train_args,
        args.ray_address,
        create_dirs=not args.dry_run,
    )
    if args.print_env:
        print(json.dumps({k: env[k] for k in sorted(env) if k.startswith(("SWEPRO_", "SLIME_"))}, indent=2))
    if args.dry_run:
        print(shlex.join(command))
        return 0
    subprocess.run(command, check=True, env=env, cwd=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
