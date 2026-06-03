import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = REPO_ROOT / "examples" / "swebench-pro" / "launch_swepro_rl.py"
K8S_STACK_PATH = REPO_ROOT / "examples" / "swebench-pro" / "k8s-gcp02-swepro-stack.yaml"


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("swepro_launch", LAUNCH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_env_passes_dynamo_trace_inputs_without_legacy_swepro_traces():
    launch = _load_launch_module()
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "DYNAMO_FRONTEND_URL": "http://warnold-swepro-frontend:3000",
        "SLIME_SPEEDSCOPE_TRACE_PATH": "/tmp/old.jsonl",
        "SWEPRO_SPEEDSCOPE_TRACE_PATH": "/tmp/old-swe.jsonl",
        "SWEPRO_MODEL_TRACE_PATH": "/tmp/model.jsonl",
    }

    runtime = launch.runtime_env(env, REPO_ROOT, "0")["env_vars"]

    assert runtime["SWEPRO_RUN_ID"] == "run-123"
    assert runtime["DYNAMO_FRONTEND_URL"] == "http://warnold-swepro-frontend:3000"
    assert runtime["SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT"] == "20390"
    assert "SLIME_SPEEDSCOPE_TRACE_PATH" not in runtime
    assert "SWEPRO_SPEEDSCOPE_TRACE_PATH" not in runtime
    assert "SWEPRO_MODEL_TRACE_PATH" not in runtime


def test_default_log_probs_chunk_size_is_throughput_oriented():
    launch = _load_launch_module()
    env = {}

    launch.apply_profile_defaults(env, "env", REPO_ROOT, write_state=False)
    derived = launch.derive_config(env)

    assert derived.log_probs_chunk_size == 512

    profile_env = {}
    launch.apply_profile_defaults(profile_env, "speedscope-current", REPO_ROOT, write_state=False)
    assert profile_env["SWEPRO_LOG_PROBS_CHUNK_SIZE"] == "512"


def test_qwen35_command_uses_upstream_packed_sequence_shape():
    launch = _load_launch_module()
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "SWEPRO_RAY_SUBMISSION_ID": "run-123",
        "SWEPRO_DURABLE_LOGS": "0",
        "SWEPRO_MODEL_ARGS_SCRIPT": "scripts/models/qwen3.5-122B-A10B.sh",
        "SWEPRO_QKV_FORMAT": "thd",
        "SWEPRO_USE_DYNAMIC_BATCH_SIZE": "0",
        "SWEPRO_MOE_TOKEN_DISPATCHER_TYPE": "alltoall",
        "SLIME_SKIP_WEIGHT_UPDATES": "1",
    }

    derived = launch.derive_config(env)
    durable_logs = launch.durable_log_config(env)
    model_args = launch.load_model_args(REPO_ROOT, env["SWEPRO_MODEL_ARGS_SCRIPT"])
    command = launch.build_command(
        env,
        REPO_ROOT,
        derived,
        durable_logs,
        [],
        "http://127.0.0.1:8265",
        create_dirs=False,
    )

    assert model_args[model_args.index("--moe-token-dispatcher-type") + 1] == "alltoall"
    assert launch.runtime_env(env, REPO_ROOT, "0")["env_vars"]["SLIME_SKIP_WEIGHT_UPDATES"] == "1"
    assert command[command.index("--qkv-format") + 1] == "thd"
    assert "--use-dynamic-batch-size" not in command
    assert "--micro-batch-size" in command
    assert command[command.index("--moe-token-dispatcher-type") + 1] == "alltoall"


def test_runtime_env_defaults_to_unbounded_agent_turns_and_turn_tokens():
    launch = _load_launch_module()
    env = {"SWEPRO_RUN_ID": "run-123"}

    runtime = launch.runtime_env(env, REPO_ROOT, "0")["env_vars"]

    assert runtime["SWEPRO_MAX_TOOL_CALLS"] == "0"
    assert runtime["SWEPRO_TURN_MAX_TOKENS"] == "0"


def test_optimizer_cpu_offload_is_on_by_default():
    launch = _load_launch_module()
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "SWEPRO_RAY_SUBMISSION_ID": "run-123",
        "SWEPRO_DURABLE_LOGS": "0",
    }
    derived = launch.derive_config(env)
    durable_logs = launch.durable_log_config(env)

    command = launch.build_command(
        env,
        REPO_ROOT,
        derived,
        durable_logs,
        [],
        "http://127.0.0.1:8265",
        create_dirs=False,
    )
    command_text = " ".join(command)

    assert "--use-precision-aware-optimizer" in command
    assert "--optimizer-cpu-offload" in command
    assert "--overlap-cpu-optimizer-d2h-h2d" in command
    assert "--use-precision-aware-optimizer --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d" in command_text


def test_mock_trainer_mode_uses_rollout_only_with_mock_parameters():
    launch = _load_launch_module()
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "SWEPRO_RAY_SUBMISSION_ID": "run-123",
        "SWEPRO_DURABLE_LOGS": "0",
        "SWEPRO_ROLLOUT_WITH_MOCK_TRAINER": "1",
        "SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND": "7000",
        "SWEPRO_MOCK_WEIGHT_UPDATE_SECONDS": "12.5",
    }
    derived = launch.derive_config(env)
    durable_logs = launch.durable_log_config(env)

    command = launch.build_command(
        env,
        REPO_ROOT,
        derived,
        durable_logs,
        [],
        "http://127.0.0.1:8265",
        create_dirs=False,
    )
    runtime = launch.runtime_env(env, REPO_ROOT, "0")["env_vars"]

    assert (
        command[command.index("--rollout-function-path") + 1]
        == "slime.rollout.fully_async_rollout.rollout_with_mock_trainer"
    )
    assert "--debug-rollout-only" in command
    assert command[command.index("--mock-trainer-tokens-per-second") + 1] == "7000"
    assert command[command.index("--mock-weight-update-seconds") + 1] == "12.5"
    assert runtime["SWEPRO_ROLLOUT_WITH_MOCK_TRAINER"] == "1"
    assert runtime["SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND"] == "7000"
    assert runtime["SWEPRO_MOCK_WEIGHT_UPDATE_SECONDS"] == "12.5"


def test_tachometer_config_discovers_expected_engine_count_and_writes_run_metrics():
    launch = _load_launch_module()
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "SWEPRO_ROLLOUT_NUM_GPUS": "8",
        "SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE": "2",
    }
    durable_logs = launch.durable_log_config(env)
    tachometer = launch.tachometer_config(env)

    script = launch.tachometer_collector_script(
        durable_logs,
        tachometer,
        "http://127.0.0.1:8265",
        "http://warnold-swepro-frontend:3000",
    )

    assert tachometer.expected_engines == 4
    assert "expected_engines=4" in script
    assert "tachometer-scraper.toml" in script
    assert "tachometer-data" in script
    assert "frontend_url=http://warnold-swepro-frontend:3000" in script
    assert 'f"{frontend_url}/metrics"' in script
    assert "http://warnold-swepro-trainer:8265/metrics" in script
    assert "dynamo_engine_{idx}" in script
    assert "@@" not in script
    assert "{{" not in script
    assert "}}" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {
                    "instances": [
                        {"endpoint": "generate", "instance_id": 0, "transport": {"tcp": "10.0.0.1:30001/a"}},
                        {"endpoint": "generate", "instance_id": 1, "transport": {"tcp": "10.0.0.2:30001/a"}},
                        {"endpoint": "generate", "instance_id": 2, "transport": {"tcp": "10.0.0.3:30001/a"}},
                        {"endpoint": "generate", "instance_id": 3, "transport": {"tcp": "10.0.0.4:30001/a"}},
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "tachometer.toml"
            discovery_path = Path(tmp) / "workers.json"
            python_source = script.split('python3 - <<\'PY\' >> "$scraper_log" 2>&1\n', 1)[1].split("\nPY\n", 1)[0]
            env = dict(os.environ)
            env.update({
                "RUN_ID": "run-123",
                "FRONTEND_URL": f"http://127.0.0.1:{server.server_port}",
                "EXPECTED_ENGINES": "4",
                "DYN_SYSTEM_PORT": "30001",
                "NIXL_PORT": "19090",
                "FREQUENCY": "0.5",
                "ROWS_PER_PARQUET": "1000000",
                "SAVE_INTERVAL_SECS": "5",
                "DISCOVERY_TIMEOUT": "1",
                "CONFIG_PATH": str(config_path),
                "STORAGE_PATH": str(Path(tmp) / "tachometer-data"),
                "DISCOVERY_JSON": str(discovery_path),
            })
            subprocess.run(["python3", "-c", python_source], env=env, check=True)
            config_text = config_path.read_text()
            assert config_text.startswith("storage = ")
            assert "\\nrows_per_parquet" not in config_text
            assert config_text.count("[[endpoints]]") == 10
            assert '"endpoint" = "dynamo_engine"' in config_text
            assert '"endpoint" = "nixl"' in config_text
    finally:
        server.shutdown()


def test_k8s_stack_exposes_dynamo_agent_trace_port_and_session_dependencies():
    text = K8S_STACK_PATH.read_text()
    dockerfile = (REPO_ROOT / "examples" / "swebench-pro" / "Dockerfile.arm64-gb200").read_text()

    assert "DYN_AGENT_TRACE_SINKS=jsonl_gz" in text
    assert "DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://0.0.0.0:20390" in text
    assert "port: 20390" in text
    assert "/data/swebench-pro/pod-logs/$(hostname)" in text
    assert "start_ray_log_sync" in text
    assert "nats-server -js -m 8222 2>&1 | tee" in text
    assert "--enable-metrics" in text
    assert "python3 -m pip install --quiet nats-py docker pyyaml pyzmq msgpack" in text
    assert "COPY dynamo_agent_trace.py /opt/swepro-session/dynamo_agent_trace.py" in (
        REPO_ROOT / "docker" / "swepro-session" / "Dockerfile"
    ).read_text()
    assert "COPY --from=tachometer_scraper /usr/local/bin/tachometer-scraper" in dockerfile
    assert "artifact: scraper-aarch64-unknown-linux-gnu" in dockerfile
    assert "SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT" in text
    assert "SWEPRO_DYNAMO_ROUTER_KV_EVENTS" in text
    assert "--no-router-kv-events" in text
    assert "--router-predicted-ttl-secs" in text
    assert "KV_EVENTS_ARGS" in text
    assert "SWEPRO_SPEEDSCOPE_TRACE_PATH" not in text
