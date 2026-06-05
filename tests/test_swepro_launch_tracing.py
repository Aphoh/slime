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
ROUTER_BENCH_SCRIPT_PATH = REPO_ROOT / ".context" / "launch_router_bench.sh"
ROUTER_BENCH_CONFIG_PATH = REPO_ROOT / "examples" / "swebench-pro" / "router_bench_trace_replay.yaml"


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


def test_run_config_yaml_values_flow_into_command_and_runtime_env(tmp_path):
    launch = _load_launch_module()
    config_path = tmp_path / "swepro-run.yaml"
    config_path.write_text(
        """
model:
  args_script: scripts/models/qwen3.5-122B-A10B.sh
  hf_checkpoint: /models/hf
  ref_load: /models/ref
  prompt_data: /data/prompts.jsonl
  input_key: prompt
  label_key: instance_id
routing:
  frontend_url: http://cfg-frontend:3000
  request_plane: tcp
  tcp_request_timeout: 42
  sglang_server_concurrency: 17
mock_trainer:
  enabled: true
  tokens_per_second: 12345
  weight_update_seconds: 6.5
trace_replay:
  path: /data/traces/router.jsonl
  scale: 0.25
  force_fixed_decode: true
async_worker:
  max_inflight: 9
  max_started_groups: auto
  group_max_attempts: 3
rollout:
  num_rollout: 5
  batch_size: 6
  over_sampling_batch_size: 12
  global_batch_size: 6
request_timeouts:
  request_timeout: 777
  request_retries: 4
trainer:
  use_dynamic_batch_size: false
  micro_batch_size: 2
tachometer:
  dyn_system_port: 30123
env:
  SGLANG_CHUNKED_PREFILL_SIZE: 1024
""",
        encoding="utf-8",
    )
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "SWEPRO_RAY_SUBMISSION_ID": "run-123",
        "SWEPRO_DURABLE_LOGS": "0",
    }

    resolved, config_env = launch.load_run_config_env(str(config_path), REPO_ROOT)
    applied, skipped = launch.apply_run_config_env(env, config_env)
    launch.resolve_run_config_derived_values(env)
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

    assert resolved == config_path
    assert "SWEPRO_REQUEST_TIMEOUT" in applied
    assert skipped == {}
    assert command[command.index("--dynamo-frontend-url") + 1] == "http://cfg-frontend:3000"
    assert command[command.index("--dynamo-worker-system-port") + 1] == "30123"
    assert command[command.index("--prompt-data") + 1] == "/data/prompts.jsonl"
    assert command[command.index("--rollout-batch-size") + 1] == "6"
    assert command[command.index("--sglang-server-concurrency") + 1] == "17"
    assert command[command.index("--mock-trainer-tokens-per-second") + 1] == "12345"
    assert command[command.index("--mock-weight-update-seconds") + 1] == "6.5"
    assert "--debug-rollout-only" in command
    assert "--use-dynamic-batch-size" not in command
    assert command[command.index("--micro-batch-size") + 1] == "2"
    assert runtime["DYNAMO_FRONTEND_URL"] == "http://cfg-frontend:3000"
    assert runtime["DYN_REQUEST_PLANE"] == "tcp"
    assert runtime["DYN_TCP_REQUEST_TIMEOUT"] == "42"
    assert runtime["SGLANG_CHUNKED_PREFILL_SIZE"] == "1024"
    assert runtime["SWEPRO_TRACE_REPLAY_PATH"] == "/data/traces/router.jsonl"
    assert "SWEPRO_MOCK_ENV_TRACE_PATH" not in runtime
    assert runtime["SWEPRO_TRACE_REPLAY_FORCE_FIXED_DECODE"] == "1"
    assert runtime["SWEPRO_ASYNC_MAX_STARTED_GROUPS"] == "30"
    assert runtime["SWEPRO_REQUEST_TIMEOUT"] == "777"
    assert runtime["SWEPRO_REQUEST_RETRIES"] == "4"


def test_run_config_preserves_explicit_env_overrides(tmp_path):
    launch = _load_launch_module()
    config_path = tmp_path / "swepro-run.yaml"
    config_path.write_text(
        """
routing:
  frontend_url: http://cfg-frontend:3000
rollout:
  num_rollout: 4
  batch_size: 9
async_worker:
  max_started_groups: auto
""",
        encoding="utf-8",
    )
    env = {
        "DYNAMO_FRONTEND_URL": "http://env-frontend:3000",
        "SWEPRO_ROLLOUT_BATCH_SIZE": "3",
    }

    _resolved, config_env = launch.load_run_config_env(str(config_path), REPO_ROOT)
    _applied, skipped = launch.apply_run_config_env(env, config_env)
    launch.resolve_run_config_derived_values(env)

    assert skipped["DYNAMO_FRONTEND_URL"] == "http://env-frontend:3000"
    assert skipped["SWEPRO_ROLLOUT_BATCH_SIZE"] == "3"
    assert env["DYNAMO_FRONTEND_URL"] == "http://env-frontend:3000"
    assert "SWEPRO_DYNAMO_FRONTEND_URL" not in env
    assert env["SWEPRO_ASYNC_MAX_STARTED_GROUPS"] == "12"


def test_router_bench_launcher_and_config_use_yaml_calling_surface():
    launch = _load_launch_module()
    script = ROUTER_BENCH_SCRIPT_PATH.read_text(encoding="utf-8")
    _resolved, config_env = launch.load_run_config_env(str(ROUTER_BENCH_CONFIG_PATH), REPO_ROOT)

    assert "--run-config" in script
    assert "router_bench_trace_replay.yaml" in script
    assert "SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND" not in script
    assert "SWEPRO_MODEL_ARGS_SCRIPT" not in script
    assert "DYNAMO_FRONTEND_URL:-" not in script
    assert "SWEPRO_ROUTER_VARIANT:-unknown" not in script
    assert config_env["SWEPRO_ROLLOUT_WITH_MOCK_TRAINER"] == "1"
    assert config_env["SWEPRO_MOCK_TRAINER_TOKENS_PER_SECOND"] == "15000"
    assert config_env["SWEPRO_TRACE_REPLAY_PATH"].endswith("trace-replay-sample-qwen35-v1-8192gen-tool120-20260605.jsonl")
    assert config_env["SWEPRO_ASYNC_MAX_STARTED_GROUPS"] == "auto"


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


def test_runtime_env_passes_trace_replay_controls():
    launch = _load_launch_module()
    env = {
        "SWEPRO_RUN_ID": "run-123",
        "SWEPRO_MOCK_ENV_TRACE_PATH": "/data/runs/baseline/agent_trace.log",
        "SWEPRO_MOCK_ENV_SCALE": "0.25",
        "SWEPRO_TRACE_REPLAY_FORCE_FIXED_DECODE": "1",
        "SWEPRO_COMPLETIONS_DEBUG": "1",
        "SWEPRO_ASYNC_SHUTDOWN_DRAIN": "0",
        "SWEPRO_ASYNC_SHUTDOWN_TIMEOUT": "17",
        "SWEPRO_ASYNC_MAX_STARTED_GROUPS": "11",
    }

    runtime = launch.runtime_env(env, REPO_ROOT, "0")["env_vars"]

    assert runtime["SWEPRO_MOCK_ENV_TRACE_PATH"] == "/data/runs/baseline/agent_trace.log"
    assert runtime["SWEPRO_MOCK_ENV_SCALE"] == "0.25"
    assert runtime["SWEPRO_TRACE_REPLAY_FORCE_FIXED_DECODE"] == "1"
    assert runtime["SWEPRO_COMPLETIONS_DEBUG"] == "1"
    assert runtime["SWEPRO_ASYNC_SHUTDOWN_DRAIN"] == "0"
    assert runtime["SWEPRO_ASYNC_SHUTDOWN_TIMEOUT"] == "17"
    assert runtime["SWEPRO_ASYNC_MAX_STARTED_GROUPS"] == "11"


def test_engine_warmup_uses_prefill_chunk_size_and_pins_each_worker(tmp_path):
    launch = _load_launch_module()
    requests = []

    class WarmupHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = json.dumps(
                    {
                        "instances": [
                            {"endpoint": "generate", "instance_id": 10, "transport": {"tcp": "10.0.0.10:38117/a"}},
                            {"endpoint": "generate", "instance_id": 11, "transport": {"tcp": "10.0.0.11:38117/a"}},
                        ]
                    }
                ).encode()
            elif self.path == "/v1/models":
                body = json.dumps({"data": [{"id": "/shared/model"}]}).encode()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(body)
            requests.append({"headers": dict(self.headers), "payload": payload})
            response = json.dumps(
                {
                    "choices": [{"text": "x", "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": len(payload["prompt"]),
                        "completion_tokens": payload["max_tokens"],
                        "total_tokens": len(payload["prompt"]) + payload["max_tokens"],
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), WarmupHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            "DYNAMO_FRONTEND_URL": f"http://127.0.0.1:{server.server_port}",
            "SWEPRO_ENGINE_WARMUP_ENABLED": "1",
            "SGLANG_CHUNKED_PREFILL_SIZE": "7",
            "SWEPRO_ENGINE_WARMUP_OSL": "1",
            "SWEPRO_ENGINE_WARMUP_EXPECTED_ENGINES": "2",
            "SWEPRO_ENGINE_WARMUP_DISCOVERY_TIMEOUT": "1",
        }
        tachometer = launch.tachometer_config({"SWEPRO_ROLLOUT_NUM_GPUS": "2", "SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE": "1"})
        warmup = launch.engine_warmup_config(env, tachometer)
        durable_logs = launch.DurableLogConfig(
            enabled=True,
            run_id="run-123",
            submission_id="run-123",
            log_dir=tmp_path,
            poll_seconds=30,
        )

        results = launch.warmup_dynamo_workers(warmup, durable_logs, run_id="run-123")

        assert len(results) == 2
        assert (tmp_path / "warmup.json").exists()
        assert [request["headers"]["X-Worker-Instance-Id"] for request in requests] == ["10", "11"]
        assert [request["payload"]["model"] for request in requests] == ["/shared/model", "/shared/model"]
        assert [len(request["payload"]["prompt"]) for request in requests] == [7, 7]
        assert [request["payload"]["max_tokens"] for request in requests] == [1, 1]
        assert [request["payload"]["min_tokens"] for request in requests] == [1, 1]
        assert [request["payload"]["ignore_eos"] for request in requests] == [True, True]
    finally:
        server.shutdown()


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
