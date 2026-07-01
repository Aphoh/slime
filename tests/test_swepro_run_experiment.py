import importlib.util
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_PATH = REPO_ROOT / "examples" / "swebench-pro" / "run_experiment.py"
CLUSTER_PATH = REPO_ROOT / "examples" / "swebench-pro" / "reproducible" / "cluster.yaml"
EXPERIMENT_PATH = REPO_ROOT / "examples" / "swebench-pro" / "reproducible" / "experiment_config.yaml"


def _load_interface_module():
    spec = importlib.util.spec_from_file_location("swepro_run_experiment", INTERFACE_PATH)
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
        "http://YOUR_DYNAMO_FRONTEND_HOST:3000"
    )
    assert plan.train_command[plan.train_command.index("--dynamo-api-mode") + 1] == "responses"
    assert plan.train_command[plan.train_command.index("--dynamo-metadata-upload-format") + 1] == "msgpack"
    assert plan.train_command[plan.train_command.index("--dynamo-metadata-upload-url") + 1] == (
        "s3://YOUR_S3_BUCKET/slynamo/rollout-metadata"
    )
    assert "--use-rollout-routing-replay" in plan.train_command
    assert plan.train_command[plan.train_command.index("--rollout-function-path") + 1] == (
        "slime.rollout.sglang_rollout.generate_rollout"
    )
    assert plan.train_command[plan.train_command.index("--advantage-estimator") + 1] == "grpo"
    assert "--debug-rollout-only" not in plan.train_command
    assert not any(token.startswith("SWEPRO_") for token in plan.train_command)
    assert plan.runtime_env["env_vars"]["DYNAMO_FRONTEND_URL"] == "http://YOUR_DYNAMO_FRONTEND_HOST:3000"
    assert plan.runtime_env["env_vars"]["SWEPRO_DYNAMO_API_MODE"] == "responses"
    assert plan.runtime_env["env_vars"]["SWEPRO_DYNAMO_METADATA_UPLOAD_FORMAT"] == "msgpack"
    assert plan.runtime_env["env_vars"]["PYTHONPATH"] == "/root/Megatron-LM/:.:examples/swebench-pro"
    assert "SWEPRO_TRACE_REPLAY_PATH" not in plan.runtime_env["env_vars"]
    assert "--no-wait" not in plan.ray_command


def test_execution_rejects_unresolved_customer_placeholders():
    interface, plan = _load_example_plan()

    with pytest.raises(interface.ConfigError, match="YOUR_HF_CHECKPOINT"):
        interface.validate_plan_for_execution(plan)


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

    with pytest.raises(interface.ConfigError, match="api_mode must be completions or responses"):
        interface.validate_experiment({
            "version": 1,
            "argument_groups": {"train": []},
            "swepro": {"api_mode": "chat"},
        })

    with pytest.raises(interface.ConfigError, match="metadata_upload_format must be msgpack"):
        interface.validate_experiment({
            "version": 1,
            "argument_groups": {"train": []},
            "swepro": {"metadata_upload_format": "json"},
        })

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
