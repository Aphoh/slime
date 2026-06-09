import importlib.util
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_PATH = REPO_ROOT / "examples" / "swebench-pro" / "experiment_interface.py"
CLUSTER_PATH = REPO_ROOT / "examples" / "swebench-pro" / "reproducible" / "cluster.yaml"
EXPERIMENT_PATH = REPO_ROOT / "examples" / "swebench-pro" / "reproducible" / "experiment_config.yaml"


def _load_interface_module():
    spec = importlib.util.spec_from_file_location("swepro_experiment_interface", INTERFACE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_example_plan(mode=None):
    interface = _load_interface_module()
    cluster = interface.load_yaml_config(CLUSTER_PATH, expected_name="cluster.yaml")
    experiment = interface.load_yaml_config(EXPERIMENT_PATH, expected_name="experiment_config.yaml")
    return interface, interface.build_run_plan(cluster, experiment, mode=mode)


def test_reproducible_run_mode_builds_stock_train_async_command():
    _interface, plan = _load_example_plan()

    assert plan.mode == "run"
    assert plan.train_command[:2] == ["python3", "train_async.py"]
    assert plan.train_command[plan.train_command.index("--actor-num-nodes") + 1] == "4"
    assert plan.train_command[plan.train_command.index("--seq-length") + 1] == "131072"
    assert plan.train_command[plan.train_command.index("--tensor-model-parallel-size") + 1] == "2"
    assert plan.train_command[plan.train_command.index("--pipeline-model-parallel-size") + 1] == "8"
    assert plan.train_command[plan.train_command.index("--context-parallel-size") + 1] == "1"
    assert plan.train_command[plan.train_command.index("--dynamo-frontend-url") + 1] == (
        "http://warnold-swepro-frontend:3000"
    )
    assert plan.train_command[plan.train_command.index("--rollout-function-path") + 1] == (
        "slime.rollout.sglang_rollout.generate_rollout"
    )
    assert plan.train_command[plan.train_command.index("--advantage-estimator") + 1] == "grpo"
    assert "--debug-rollout-only" not in plan.train_command
    assert not any(token.startswith("SWEPRO_") for token in plan.train_command)
    assert plan.runtime_env["env_vars"]["DYNAMO_FRONTEND_URL"] == "http://warnold-swepro-frontend:3000"
    assert plan.runtime_env["env_vars"]["PYTHONPATH"] == "/root/Megatron-LM/:.:examples/swebench-pro"
    assert "SWEPRO_TRACE_REPLAY_PATH" not in plan.runtime_env["env_vars"]
    assert "--no-wait" not in plan.ray_command


def test_perf_mode_uses_trace_replay_and_mock_trainer_without_changing_entrypoint():
    _interface, plan = _load_example_plan(mode="perf-test")

    assert plan.mode == "perf-test"
    assert plan.train_command[:2] == ["python3", "train_async.py"]
    assert "--debug-rollout-only" in plan.train_command
    assert plan.train_command[plan.train_command.index("--rollout-function-path") + 1] == (
        "slime.rollout.fully_async_rollout.rollout_with_mock_trainer"
    )
    assert plan.train_command[plan.train_command.index("--mock-trainer-tokens-per-second") + 1] == "15000"
    assert plan.runtime_env["env_vars"]["SWEPRO_TRACE_REPLAY_FORCE_FIXED_DECODE"] == "1"
    assert plan.runtime_env["env_vars"]["SWEPRO_TRACE_REPLAY_PATH"].endswith(
        "trace-replay-sample-qwen35-v1-8192gen-tool120-20260605.jsonl"
    )
    assert plan.runtime_env["env_vars"]["SWEPRO_ASYNC_MAX_STARTED_GROUPS"] == "160"


def test_schema_rejects_unknown_keys_and_duplicate_cluster_owned_flags():
    interface = _load_interface_module()

    with pytest.raises(interface.ConfigError, match="unknown key"):
        interface.validate_cluster({"version": 1, "surprise": True})

    cluster = interface.LoadedConfig(
        path=CLUSTER_PATH,
        data={
            "version": 1,
            "repo_root": "../../..",
            "resources": {"actor_num_nodes": 1},
            "ray": {"address": "http://127.0.0.1:8265"},
        },
    )
    experiment = interface.LoadedConfig(
        path=EXPERIMENT_PATH,
        data={
            "version": 1,
            "entrypoint": "train.py",
            "argument_groups": {"train": ["--actor-num-nodes=2", "--num-rollout", "1"]},
        },
    )

    with pytest.raises(interface.ConfigError, match="cluster-managed resource"):
        interface.build_run_plan(cluster, experiment)


def test_schema_rejects_invalid_versions_boolean_strings_and_topology():
    interface = _load_interface_module()

    with pytest.raises(interface.ConfigError, match="cluster.version must be 1"):
        interface.validate_cluster({"version": 2})

    with pytest.raises(interface.ConfigError, match="cluster.dynamo.enabled must be a boolean"):
        interface.validate_cluster({"version": 1, "dynamo": {"enabled": "false"}})

    cluster = interface.LoadedConfig(
        path=CLUSTER_PATH,
        data={
            "version": 1,
            "resources": {
                "actor_num_nodes": 1,
                "actor_num_gpus_per_node": 4,
            },
        },
    )
    experiment = interface.LoadedConfig(
        path=EXPERIMENT_PATH,
        data={
            "version": 1,
            "argument_groups": {
                "train": [
                    "--tensor-model-parallel-size",
                    "2",
                    "--pipeline-model-parallel-size",
                    "4",
                    "--context-parallel-size",
                    "1",
                ]
            },
        },
    )

    with pytest.raises(interface.ConfigError, match="actor GPU count must be divisible"):
        interface.build_run_plan(cluster, experiment)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
