import importlib.util
import sys
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


def test_k8s_stack_exposes_dynamo_agent_trace_port_and_session_dependencies():
    text = K8S_STACK_PATH.read_text()

    assert "DYN_AGENT_TRACE_SINKS=jsonl_gz" in text
    assert "DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://0.0.0.0:20390" in text
    assert "port: 20390" in text
    assert "python3 -m pip install --quiet nats-py docker pyyaml pyzmq msgpack" in text
    assert "COPY dynamo_agent_trace.py /opt/swepro-session/dynamo_agent_trace.py" in (
        REPO_ROOT / "docker" / "swepro-session" / "Dockerfile"
    ).read_text()
    assert "SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT" in text
    assert "SWEPRO_SPEEDSCOPE_TRACE_PATH" not in text
