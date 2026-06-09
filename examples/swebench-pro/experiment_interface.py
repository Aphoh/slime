#!/usr/bin/env python3
"""Small, reproducible SWE-Pro/Dynamo experiment interface.

The public surface is intentionally narrow: a cluster YAML describes launch
infrastructure, an experiment YAML lists normal slime CLI arguments, and this
module combines them into an ordinary ``ray job submit -- python3 train*.py``
command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


class ConfigError(ValueError):
    """Raised when cluster.yaml or experiment_config.yaml fails validation."""


CLUSTER_KEYS = {"version", "name", "repo_root", "ray", "resources", "runtime_env", "dynamo", "artifacts"}
RAY_KEYS = {"address", "start_head", "job"}
RAY_START_KEYS = {
    "enabled",
    "node_ip_address",
    "num_gpus",
    "dashboard_host",
    "dashboard_port",
    "disable_usage_stats",
    "extra_args",
}
RAY_JOB_KEYS = {"submission_id", "no_wait", "working_dir", "metadata"}
RESOURCE_KEYS = {
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "num_gpus_per_node",
    "colocate",
}
RUNTIME_ENV_KEYS = {"env_vars"}
DYNAMO_KEYS = {
    "enabled",
    "frontend_url",
    "nats_url",
    "worker_system_port",
    "frontend_wait_timeout",
    "tool_events_zmq_endpoint",
    "tool_events_zmq_port",
}
ARTIFACT_KEYS = {"root_dir", "run_id", "durable_logs"}

EXPERIMENT_KEYS = {
    "version",
    "name",
    "description",
    "entrypoint",
    "model_args_script",
    "argument_groups",
    "modes",
    "swepro",
}
MODE_KEYS = {"description", "entrypoint", "include_groups", "extra_args", "swepro"}
SWEPRO_KEYS = {
    "agent_mode",
    "max_tool_calls",
    "episode_wall_timeout",
    "turn_max_tokens",
    "model_call_timeout",
    "request_timeout",
    "request_retries",
    "session_start_timeout",
    "session_step_request_timeout",
    "session_submit_request_timeout",
    "session_close_timeout",
    "session_health_timeout",
    "session_rollout_retries",
    "async_max_inflight",
    "async_max_started_groups",
    "async_group_max_attempts",
    "async_shutdown_drain",
    "async_shutdown_timeout",
    "trace_replay_path",
    "mock_env_trace_path",
    "mock_env_scale",
    "trace_replay_force_fixed_decode",
    "mock_env_reward",
    "trace_replay_reward",
    "completions_debug",
}
SWEPRO_ENV_MAP = {
    "agent_mode": "SWEPRO_AGENT_MODE",
    "max_tool_calls": "SWEPRO_MAX_TOOL_CALLS",
    "episode_wall_timeout": "SWEPRO_EPISODE_WALL_TIMEOUT",
    "turn_max_tokens": "SWEPRO_TURN_MAX_TOKENS",
    "model_call_timeout": "SWEPRO_MODEL_CALL_TIMEOUT",
    "request_timeout": "SWEPRO_REQUEST_TIMEOUT",
    "request_retries": "SWEPRO_REQUEST_RETRIES",
    "session_start_timeout": "SWEPRO_SESSION_START_TIMEOUT",
    "session_step_request_timeout": "SWEPRO_SESSION_STEP_REQUEST_TIMEOUT",
    "session_submit_request_timeout": "SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT",
    "session_close_timeout": "SWEPRO_SESSION_CLOSE_TIMEOUT",
    "session_health_timeout": "SWEPRO_SESSION_HEALTH_TIMEOUT",
    "session_rollout_retries": "SWEPRO_SESSION_ROLLOUT_RETRIES",
    "async_max_inflight": "SWEPRO_ASYNC_MAX_INFLIGHT",
    "async_max_started_groups": "SWEPRO_ASYNC_MAX_STARTED_GROUPS",
    "async_group_max_attempts": "SWEPRO_ASYNC_GROUP_MAX_ATTEMPTS",
    "async_shutdown_drain": "SWEPRO_ASYNC_SHUTDOWN_DRAIN",
    "async_shutdown_timeout": "SWEPRO_ASYNC_SHUTDOWN_TIMEOUT",
    "trace_replay_path": "SWEPRO_TRACE_REPLAY_PATH",
    "mock_env_trace_path": "SWEPRO_MOCK_ENV_TRACE_PATH",
    "mock_env_scale": "SWEPRO_MOCK_ENV_SCALE",
    "trace_replay_force_fixed_decode": "SWEPRO_TRACE_REPLAY_FORCE_FIXED_DECODE",
    "mock_env_reward": "SWEPRO_MOCK_ENV_REWARD",
    "trace_replay_reward": "SWEPRO_TRACE_REPLAY_REWARD",
    "completions_debug": "SWEPRO_COMPLETIONS_DEBUG",
}

RESOURCE_FLAG_MAP = {
    "actor_num_nodes": "--actor-num-nodes",
    "actor_num_gpus_per_node": "--actor-num-gpus-per-node",
    "rollout_num_gpus": "--rollout-num-gpus",
    "rollout_num_gpus_per_engine": "--rollout-num-gpus-per-engine",
    "num_gpus_per_node": "--num-gpus-per-node",
}
DYNAMO_MANAGED_FLAGS = {
    "--rollout-backend",
    "--dynamo-frontend-url",
    "--dynamo-worker-system-port",
    "--dynamo-frontend-wait-timeout",
}
ENTRYPOINTS = {"train.py", "train_async.py"}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: dict[str, Any]


@dataclass(frozen=True)
class RunPlan:
    mode: str
    run_id: str
    repo_root: Path
    artifact_dir: Path | None
    train_command: list[str]
    ray_command: list[str]
    runtime_env: dict[str, dict[str, str]]
    ray_start_command: list[str] | None


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if unknown:
        raise ConfigError(f"{path} has unknown key(s): {', '.join(unknown)}")


def _env_value(value: Any, path: str) -> str:
    if value is None:
        raise ConfigError(f"{path} cannot be null")
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ConfigError(f"{path} must be a scalar")


def _require_version(mapping: dict[str, Any], path: str) -> None:
    if type(mapping.get("version")) is not int or mapping["version"] != 1:
        raise ConfigError(f"{path}.version must be 1")


def _optional_bool(mapping: dict[str, Any], key: str, path: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be a boolean")
    return value


def _optional_positive_int(mapping: dict[str, Any], key: str, path: str) -> int | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path}.{key} must be a positive integer")
    return value


def _optional_nonnegative_int(mapping: dict[str, Any], key: str, path: str) -> int | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{path}.{key} must be a non-negative integer")
    return value


def _optional_url(mapping: dict[str, Any], key: str, path: str, schemes: set[str]) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, str):
        raise ConfigError(f"{path}.{key} must be a URL string")
    parsed = urlparse(value)
    if parsed.scheme not in schemes or not parsed.hostname:
        raise ConfigError(f"{path}.{key} must use {', '.join(sorted(schemes))} and include a host")
    return value


def _argv(value: Any, path: str) -> list[str]:
    result: list[str] = []
    for index, item in enumerate(_require_list(value, path)):
        if item is None or isinstance(item, bool) or isinstance(item, (list, dict)):
            raise ConfigError(f"{path}[{index}] must be a string or number")
        result.append(str(item))
    return result


def _resolve_path(value: str | None, base_dir: Path, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ConfigError("missing path")
        return default
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_yaml_config(path: str | Path, *, expected_name: str) -> LoadedConfig:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ConfigError(f"{expected_name} not found: {resolved}")
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{expected_name} must be a YAML mapping: {resolved}")
    return LoadedConfig(path=resolved, data=loaded)


def validate_cluster(cluster: dict[str, Any]) -> None:
    _reject_unknown(cluster, CLUSTER_KEYS, "cluster")
    _require_version(cluster, "cluster")
    ray = _require_mapping(cluster.get("ray"), "cluster.ray")
    _reject_unknown(ray, RAY_KEYS, "cluster.ray")
    _optional_url(ray, "address", "cluster.ray", {"http", "https"})
    start_head = _require_mapping(ray.get("start_head"), "cluster.ray.start_head")
    _reject_unknown(start_head, RAY_START_KEYS, "cluster.ray.start_head")
    _optional_bool(start_head, "enabled", "cluster.ray.start_head")
    _optional_bool(start_head, "disable_usage_stats", "cluster.ray.start_head")
    _optional_nonnegative_int(start_head, "num_gpus", "cluster.ray.start_head")
    _optional_positive_int(start_head, "dashboard_port", "cluster.ray.start_head")
    _argv(start_head.get("extra_args"), "cluster.ray.start_head.extra_args")
    job = _require_mapping(ray.get("job"), "cluster.ray.job")
    _reject_unknown(job, RAY_JOB_KEYS, "cluster.ray.job")
    _optional_bool(job, "no_wait", "cluster.ray.job")
    _require_mapping(job.get("metadata"), "cluster.ray.job.metadata")

    resources = _require_mapping(cluster.get("resources"), "cluster.resources")
    _reject_unknown(resources, RESOURCE_KEYS, "cluster.resources")
    for key in RESOURCE_FLAG_MAP:
        _optional_positive_int(resources, key, "cluster.resources")
    _optional_bool(resources, "colocate", "cluster.resources")
    rollout_num_gpus = resources.get("rollout_num_gpus")
    rollout_num_gpus_per_engine = resources.get("rollout_num_gpus_per_engine")
    if (
        rollout_num_gpus is not None
        and rollout_num_gpus_per_engine is not None
        and rollout_num_gpus % rollout_num_gpus_per_engine != 0
    ):
        raise ConfigError("cluster.resources.rollout_num_gpus must be divisible by rollout_num_gpus_per_engine")

    runtime_env = _require_mapping(cluster.get("runtime_env"), "cluster.runtime_env")
    _reject_unknown(runtime_env, RUNTIME_ENV_KEYS, "cluster.runtime_env")
    for key, value in _require_mapping(runtime_env.get("env_vars"), "cluster.runtime_env.env_vars").items():
        _env_value(value, f"cluster.runtime_env.env_vars.{key}")

    dynamo = _require_mapping(cluster.get("dynamo"), "cluster.dynamo")
    _reject_unknown(dynamo, DYNAMO_KEYS, "cluster.dynamo")
    _optional_bool(dynamo, "enabled", "cluster.dynamo")
    _optional_url(dynamo, "frontend_url", "cluster.dynamo", {"http", "https"})
    _optional_url(dynamo, "nats_url", "cluster.dynamo", {"nats", "tls"})
    _optional_positive_int(dynamo, "worker_system_port", "cluster.dynamo")
    _optional_positive_int(dynamo, "frontend_wait_timeout", "cluster.dynamo")
    _optional_positive_int(dynamo, "tool_events_zmq_port", "cluster.dynamo")
    if dynamo.get("enabled") is True and not dynamo.get("frontend_url"):
        raise ConfigError("cluster.dynamo.frontend_url is required when Dynamo is enabled")
    if dynamo.get("enabled") is True and resources.get("colocate") is True:
        raise ConfigError("cluster.resources.colocate is incompatible with external Dynamo")

    artifacts = _require_mapping(cluster.get("artifacts"), "cluster.artifacts")
    _reject_unknown(artifacts, ARTIFACT_KEYS, "cluster.artifacts")
    _optional_bool(artifacts, "durable_logs", "cluster.artifacts")


def validate_experiment(experiment: dict[str, Any]) -> None:
    _reject_unknown(experiment, EXPERIMENT_KEYS, "experiment")
    _require_version(experiment, "experiment")
    entrypoint = experiment.get("entrypoint", "train.py")
    if entrypoint not in ENTRYPOINTS:
        raise ConfigError(f"experiment.entrypoint must be one of {sorted(ENTRYPOINTS)}, got {entrypoint!r}")
    groups = _require_mapping(experiment.get("argument_groups"), "experiment.argument_groups")
    if not groups:
        raise ConfigError("experiment.argument_groups must define at least one group")
    for group_name, args in groups.items():
        _argv(args, f"experiment.argument_groups.{group_name}")
    modes = _require_mapping(experiment.get("modes"), "experiment.modes")
    validate_swepro_mapping(_require_mapping(experiment.get("swepro"), "experiment.swepro"), "experiment.swepro")
    if not modes:
        return
    for mode_name, mode in modes.items():
        mode_path = f"experiment.modes.{mode_name}"
        mode_mapping = _require_mapping(mode, mode_path)
        _reject_unknown(mode_mapping, MODE_KEYS, mode_path)
        if "entrypoint" in mode_mapping and mode_mapping["entrypoint"] not in ENTRYPOINTS:
            raise ConfigError(f"{mode_path}.entrypoint must be one of {sorted(ENTRYPOINTS)}")
        for group_name in _require_list(mode_mapping.get("include_groups"), f"{mode_path}.include_groups"):
            if not isinstance(group_name, str):
                raise ConfigError(f"{mode_path}.include_groups entries must be strings")
            if group_name not in groups:
                raise ConfigError(f"{mode_path}.include_groups references unknown group {group_name!r}")
        _argv(mode_mapping.get("extra_args"), f"{mode_path}.extra_args")
        validate_swepro_mapping(
            _require_mapping(mode_mapping.get("swepro"), f"{mode_path}.swepro"), f"{mode_path}.swepro"
        )


def validate_swepro_mapping(mapping: dict[str, Any], path: str) -> None:
    _reject_unknown(mapping, SWEPRO_KEYS, path)
    for key, value in mapping.items():
        _env_value(value, f"{path}.{key}")


def _default_run_id(experiment_name: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_name}_{mode}_{stamp}"


def _submission_id(run_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-")
    return sanitized or _default_run_id("swepro", "run")


def detect_nvlink() -> str:
    try:
        topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "0"
    return "1" if re.search(r"NV[0-9][0-9]*", topo) else "0"


def load_model_args(repo_root: Path, model_script: str | None) -> list[str]:
    if not model_script:
        return []
    script = _resolve_path(model_script, repo_root)
    if not script.exists():
        raise ConfigError(f"model args script not found: {script}")
    bash = f"source {shlex.quote(str(script))}; printf '%s\\0' \"${{MODEL_ARGS[@]}}\""
    try:
        output = subprocess.check_output(["bash", "-lc", bash], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise ConfigError(f"failed to load MODEL_ARGS from {script}") from exc
    return [part.decode() for part in output.split(b"\0") if part]


def _selected_mode(experiment: dict[str, Any], mode: str | None) -> tuple[str, dict[str, Any]]:
    modes = _require_mapping(experiment.get("modes"), "experiment.modes")
    if mode is None:
        mode = "run" if "run" in modes else next(iter(modes), "run")
    if modes and mode not in modes:
        raise ConfigError(f"unknown mode {mode!r}; available modes: {', '.join(sorted(modes))}")
    return mode, _require_mapping(modes.get(mode), f"experiment.modes.{mode}") if modes else {}


def _contains_flag(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _cluster_stock_args(cluster: dict[str, Any], *, selected_args: list[str]) -> list[str]:
    result: list[str] = []
    resources = _require_mapping(cluster.get("resources"), "cluster.resources")
    resource_flags = {flag for key, flag in RESOURCE_FLAG_MAP.items() if key in resources}
    if resources.get("colocate") is True:
        resource_flags.add("--colocate")
    duplicated_resources = sorted(flag for flag in resource_flags if _contains_flag(selected_args, flag))
    if duplicated_resources:
        raise ConfigError(
            "cluster-managed resource flag(s) also appear in experiment args: "
            f"{', '.join(duplicated_resources)}; keep them in cluster.yaml"
        )
    for key, flag in RESOURCE_FLAG_MAP.items():
        if key in resources and resources[key] is not None:
            result.extend([flag, str(resources[key])])
    if resources.get("colocate") is True:
        result.append("--colocate")

    dynamo = _require_mapping(cluster.get("dynamo"), "cluster.dynamo")
    dynamo_enabled = dynamo.get("enabled", bool(dynamo.get("frontend_url")))
    duplicated_dynamo = sorted(
        flag for flag in DYNAMO_MANAGED_FLAGS if dynamo_enabled and _contains_flag(selected_args, flag)
    )
    if duplicated_dynamo:
        raise ConfigError(
            "cluster-managed Dynamo flag(s) also appear in experiment args: "
            f"{', '.join(duplicated_dynamo)}; keep them in cluster.yaml"
        )
    if dynamo_enabled:
        frontend_url = dynamo.get("frontend_url")
        if not frontend_url:
            raise ConfigError("cluster.dynamo.frontend_url is required when Dynamo is enabled")
        result.extend(["--rollout-backend", "dynamo", "--dynamo-frontend-url", str(frontend_url)])
        if "worker_system_port" in dynamo:
            result.extend(["--dynamo-worker-system-port", str(dynamo["worker_system_port"])])
        if "frontend_wait_timeout" in dynamo:
            result.extend(["--dynamo-frontend-wait-timeout", str(dynamo["frontend_wait_timeout"])])
    return result


def _included_group_args(experiment: dict[str, Any], mode_config: dict[str, Any], mode_name: str) -> list[str]:
    groups = _require_mapping(experiment.get("argument_groups"), "experiment.argument_groups")
    include_groups = mode_config.get("include_groups")
    if include_groups is None:
        selected_groups = list(groups)
    else:
        selected_groups = [
            str(group) for group in _require_list(include_groups, f"experiment.modes.{mode_name}.include_groups")
        ]
    result: list[str] = []
    for group in selected_groups:
        result.extend(_argv(groups[group], f"experiment.argument_groups.{group}"))
    result.extend(_argv(mode_config.get("extra_args"), f"experiment.modes.{mode_name}.extra_args"))
    return result


def _last_arg_value(argv: list[str], flag: str, default: int) -> int:
    value = default
    for index, token in enumerate(argv):
        if token.startswith(f"{flag}="):
            raw_value = token.split("=", 1)[1]
        elif token == flag and index + 1 < len(argv):
            raw_value = argv[index + 1]
        else:
            continue
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ConfigError(f"{flag} must be an integer") from exc
    return value


def _validate_topology(cluster: dict[str, Any], train_args: list[str]) -> None:
    if _contains_flag(train_args, "--debug-rollout-only"):
        return
    resources = _require_mapping(cluster.get("resources"), "cluster.resources")
    actor_world_size = int(resources.get("actor_num_nodes", 1)) * int(resources.get("actor_num_gpus_per_node", 8))
    parallel_size = (
        _last_arg_value(train_args, "--tensor-model-parallel-size", 1)
        * _last_arg_value(train_args, "--pipeline-model-parallel-size", 1)
        * _last_arg_value(train_args, "--context-parallel-size", 1)
    )
    if actor_world_size % parallel_size != 0:
        raise ConfigError(
            "actor GPU count must be divisible by tensor * pipeline * context parallel size: "
            f"{actor_world_size} % {parallel_size} != 0"
        )
    expert_parallel_size = _last_arg_value(train_args, "--expert-model-parallel-size", 1)
    if actor_world_size % expert_parallel_size != 0:
        raise ConfigError(
            "actor GPU count must be divisible by expert parallel size: "
            f"{actor_world_size} % {expert_parallel_size} != 0"
        )


def _swepro_env(experiment: dict[str, Any], mode_config: dict[str, Any], train_args: list[str]) -> dict[str, str]:
    merged: dict[str, Any] = {}
    merged.update(_require_mapping(experiment.get("swepro"), "experiment.swepro"))
    merged.update(_require_mapping(mode_config.get("swepro"), "mode.swepro"))
    if merged.get("async_max_started_groups") == "auto":
        merged["async_max_started_groups"] = _last_arg_value(train_args, "--num-rollout", 1) * _last_arg_value(
            train_args, "--rollout-batch-size", 1
        )
    return {SWEPRO_ENV_MAP[key]: _env_value(value, f"swepro.{key}") for key, value in merged.items()}


def _runtime_env(cluster: dict[str, Any], swepro_env: dict[str, str]) -> dict[str, dict[str, str]]:
    runtime_env = _require_mapping(cluster.get("runtime_env"), "cluster.runtime_env")
    raw_env = _require_mapping(runtime_env.get("env_vars"), "cluster.runtime_env.env_vars")
    env_vars = {str(key): _env_value(value, f"cluster.runtime_env.env_vars.{key}") for key, value in raw_env.items()}
    env_vars.setdefault("PYTHONPATH", "/root/Megatron-LM/:.:examples/swebench-pro")
    env_vars.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if env_vars.get("NCCL_NVLS_ENABLE", "auto") == "auto":
        env_vars["NCCL_NVLS_ENABLE"] = detect_nvlink()

    dynamo = _require_mapping(cluster.get("dynamo"), "cluster.dynamo")
    if dynamo.get("enabled", bool(dynamo.get("frontend_url"))):
        frontend_url = str(dynamo["frontend_url"])
        env_vars.setdefault("DYNAMO_FRONTEND_URL", frontend_url)
        env_vars.setdefault("SWEPRO_DYNAMO_FRONTEND_URL", frontend_url)
        if nats_url := dynamo.get("nats_url"):
            env_vars.setdefault("SWEPRO_NATS_URL", str(nats_url))
        if endpoint := dynamo.get("tool_events_zmq_endpoint"):
            env_vars.setdefault("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT", str(endpoint))
        if port := dynamo.get("tool_events_zmq_port"):
            env_vars.setdefault("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT", str(port))
    env_vars.update(swepro_env)
    return {"env_vars": {key: value for key, value in sorted(env_vars.items()) if value != ""}}


def _artifact_dir(cluster: dict[str, Any], run_id: str, cluster_path: Path) -> Path | None:
    artifacts = _require_mapping(cluster.get("artifacts"), "cluster.artifacts")
    if artifacts.get("durable_logs") is False:
        return None
    root = _resolve_path(str(artifacts.get("root_dir", ".context/swepro-runs")), cluster_path.parent)
    return root / _submission_id(run_id)


def _ray_start_command(cluster: dict[str, Any]) -> list[str] | None:
    start = _require_mapping(
        _require_mapping(cluster.get("ray"), "cluster.ray").get("start_head"), "cluster.ray.start_head"
    )
    if not start.get("enabled", False):
        return None
    command = ["ray", "start", "--head"]
    if node_ip := start.get("node_ip_address"):
        command.extend(["--node-ip-address", str(node_ip)])
    if "num_gpus" in start:
        command.extend(["--num-gpus", str(start["num_gpus"])])
    if dashboard_host := start.get("dashboard_host"):
        command.extend(["--dashboard-host", str(dashboard_host)])
    if dashboard_port := start.get("dashboard_port"):
        command.extend(["--dashboard-port", str(dashboard_port)])
    if start.get("disable_usage_stats", True):
        command.append("--disable-usage-stats")
    command.extend(_argv(start.get("extra_args"), "cluster.ray.start_head.extra_args"))
    return command


def build_run_plan(cluster_config: LoadedConfig, experiment_config: LoadedConfig, mode: str | None = None) -> RunPlan:
    cluster = cluster_config.data
    experiment = experiment_config.data
    validate_cluster(cluster)
    validate_experiment(experiment)

    repo_root = _resolve_path(cluster.get("repo_root"), cluster_config.path.parent, default=Path.cwd()).resolve()
    mode_name, mode_config = _selected_mode(experiment, mode)
    selected_args = _included_group_args(experiment, mode_config, mode_name)
    cluster_args = _cluster_stock_args(cluster, selected_args=selected_args)
    entrypoint = mode_config.get("entrypoint", experiment.get("entrypoint", "train.py"))
    model_args = load_model_args(repo_root, experiment.get("model_args_script"))
    train_command = ["python3", str(entrypoint), *cluster_args, *model_args, *selected_args]
    _validate_topology(cluster, train_command)

    experiment_name = str(experiment.get("name") or cluster.get("name") or "swepro")
    artifacts = _require_mapping(cluster.get("artifacts"), "cluster.artifacts")
    run_id = str(artifacts.get("run_id") or _default_run_id(experiment_name, mode_name))
    artifact_dir = _artifact_dir(cluster, run_id, cluster_config.path)
    runtime_env = _runtime_env(cluster, _swepro_env(experiment, mode_config, train_command))

    ray = _require_mapping(cluster.get("ray"), "cluster.ray")
    job = _require_mapping(ray.get("job"), "cluster.ray.job")
    ray_address = str(ray.get("address", "http://127.0.0.1:8265"))
    metadata = {
        "swepro_run_id": run_id,
        "swepro_mode": mode_name,
        "swepro_experiment": experiment_name,
    }
    metadata.update(_require_mapping(job.get("metadata"), "cluster.ray.job.metadata"))
    if artifact_dir is not None:
        metadata["swepro_artifact_dir"] = str(artifact_dir)
    working_dir = _resolve_path(job.get("working_dir"), cluster_config.path.parent, default=repo_root)
    ray_command = [
        "ray",
        "job",
        "submit",
        f"--address={ray_address}",
        f"--submission-id={_submission_id(str(job.get('submission_id') or run_id))}",
    ]
    if job.get("no_wait", False):
        ray_command.append("--no-wait")
    ray_command.extend(
        [
            f"--working-dir={working_dir}",
            f"--metadata-json={json.dumps(metadata, separators=(',', ':'))}",
            f"--runtime-env-json={json.dumps(runtime_env, separators=(',', ':'))}",
            "--",
            *train_command,
        ]
    )
    return RunPlan(
        mode=mode_name,
        run_id=run_id,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        train_command=train_command,
        ray_command=ray_command,
        runtime_env=runtime_env,
        ray_start_command=_ray_start_command(cluster),
    )


def write_run_artifacts(plan: RunPlan, cluster_path: Path, experiment_path: Path) -> None:
    if plan.artifact_dir is None:
        return
    plan.artifact_dir.mkdir(parents=True, exist_ok=True)
    (plan.artifact_dir / "cluster.yaml").write_text(cluster_path.read_text(encoding="utf-8"), encoding="utf-8")
    (plan.artifact_dir / "experiment_config.yaml").write_text(
        experiment_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (plan.artifact_dir / "runtime_env.json").write_text(
        json.dumps(plan.runtime_env, indent=2) + "\n", encoding="utf-8"
    )
    (plan.artifact_dir / "train_command.sh").write_text(shlex.join(plan.train_command) + "\n", encoding="utf-8")
    (plan.artifact_dir / "ray_job_submit.sh").write_text(shlex.join(plan.ray_command) + "\n", encoding="utf-8")


def execute_plan(plan: RunPlan, *, cluster_path: Path, experiment_path: Path) -> int:
    write_run_artifacts(plan, cluster_path, experiment_path)
    if plan.ray_start_command is not None:
        result = subprocess.run(plan.ray_start_command, cwd=plan.repo_root, env=os.environ)
        if result.returncode != 0:
            return result.returncode
    return subprocess.run(plan.ray_command, cwd=plan.repo_root, env=os.environ).returncode


def format_plan(plan: RunPlan) -> str:
    lines = [
        f"mode: {plan.mode}",
        f"run_id: {plan.run_id}",
        f"repo_root: {plan.repo_root}",
    ]
    if plan.artifact_dir is not None:
        lines.append(f"artifact_dir: {plan.artifact_dir}")
    if plan.ray_start_command is not None:
        lines.append(f"ray_start: {shlex.join(plan.ray_start_command)}")
    lines.append(f"train: {shlex.join(plan.train_command)}")
    lines.append(f"ray_job: {shlex.join(plan.ray_command)}")
    return "\n".join(lines)


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, help="Path to cluster.yaml")
    parser.add_argument("--experiment", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--mode", default=None, help="Experiment mode from experiment_config.yaml; defaults to run")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the Ray command without submitting")
    parser.add_argument("--validate-only", action="store_true", help="Validate configs and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    try:
        cluster_config = load_yaml_config(args.cluster, expected_name="cluster.yaml")
        experiment_config = load_yaml_config(args.experiment, expected_name="experiment_config.yaml")
        plan = build_run_plan(cluster_config, experiment_config, mode=args.mode)
    except ConfigError as exc:
        print(f"config error: {exc}", file=os.sys.stderr)
        return 2
    if args.validate_only:
        print(f"valid: mode={plan.mode} run_id={plan.run_id}")
        return 0
    if args.dry_run:
        print(format_plan(plan))
        return 0
    return execute_plan(plan, cluster_path=cluster_config.path, experiment_path=experiment_config.path)


if __name__ == "__main__":
    raise SystemExit(main())
