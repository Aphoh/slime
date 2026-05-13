import asyncio
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path


def _load_session_runner():
    nats_module = types.ModuleType("nats")
    nats_aio_module = types.ModuleType("nats.aio")
    nats_client_module = types.ModuleType("nats.aio.client")
    nats_client_module.Client = object
    sys.modules.setdefault("nats", nats_module)
    sys.modules.setdefault("nats.aio", nats_aio_module)
    sys.modules.setdefault("nats.aio.client", nats_client_module)

    module_path = Path(__file__).resolve().parents[1] / "docker" / "swepro-session" / "runner.py"
    spec = importlib.util.spec_from_file_location("swepro_session_runner", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _BlockingManager:
    def __init__(self):
        self.step_started = threading.Event()
        self.release_step = threading.Event()

    def step(self, request):
        self.step_started.set()
        if not self.release_step.wait(timeout=5):
            raise TimeoutError("test step was not released")
        return {"method": "step", "error": None}

    def health(self, request):
        return {"method": "health", "error": None}


class _Message:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode("utf-8")
        self.responses = []

    async def respond(self, payload):
        self.responses.append(json.loads(payload.decode("utf-8")))


class _FakePublisher:
    def __init__(self):
        self.records = []

    def publish(self, record):
        self.records.append(record)
        return True


def _session_trace_events(output):
    prefix = "SWEPRO_SESSION_TRACE "
    return [json.loads(line[len(prefix) :]) for line in output.splitlines() if line.startswith(prefix)]


def _trace_context():
    return {
        "session_type_id": "slime_swebench_pro",
        "session_id": "run-1",
        "trajectory_id": "run-1:swebench_pro:instance-0:sample:0:id:sample-a",
    }


def test_step_drops_session_when_deployment_disappears():
    runner = _load_session_runner()

    class DeploymentNotStartedError(Exception):
        pass

    class CommandTimeoutError(Exception):
        pass

    class BashIncorrectSyntaxError(Exception):
        pass

    class _Env:
        closed = False

        def communicate(self, **kwargs):
            raise DeploymentNotStartedError("Deployment not started")

        def close(self):
            self.closed = True

    class _Tools:
        config = types.SimpleNamespace(execution_timeout=1)

        def parse_actions(self, output):
            return "thought", "echo hi"

        def guard_multiline_input(self, action):
            return action

        def get_state(self, env):
            raise DeploymentNotStartedError("Deployment not started")

        def check_for_submission_cmd(self, observation):
            return False

    env = _Env()
    publisher = _FakePublisher()
    manager = runner.SessionManager.__new__(runner.SessionManager)
    manager.worker_id = "worker-0"
    manager.sessions = {
        "sid": runner.Session(
            session_id="sid",
            instance_id="instance-0",
            env=env,
            tools=_Tools(),
            started_at=0,
            last_used_at=0,
            agent_context=_trace_context(),
            tool_events_zmq_endpoint="tcp://trace:20390",
        )
    }
    manager._lock = threading.RLock()
    manager._valid_tools = {"bash"}
    manager._tools_config = types.SimpleNamespace(execution_timeout=1)
    manager._tool_event_publishers = {"tcp://trace:20390": publisher}
    manager._imports = {
        "CommandTimeoutError": CommandTimeoutError,
        "BashIncorrectSyntaxError": BashIncorrectSyntaxError,
    }

    result = manager.step({"session_id": "sid", "tool_call": {"function": {"name": "bash"}}})

    assert result["error"] is None
    assert result["session_dropped"] is True
    assert result["tool_error"] == "DeploymentNotStartedError"
    assert "Deployment not started" in result["state_error"]
    assert "sid" not in manager.sessions
    assert env.closed is True
    assert [record["event_type"] for record in publisher.records] == ["tool_start", "tool_error"]
    assert publisher.records[-1]["tool"]["error_type"] == "DeploymentNotStartedError"


def test_step_emits_dynamo_tool_start_and_end(capsys):
    runner = _load_session_runner()

    class CommandTimeoutError(Exception):
        pass

    class BashIncorrectSyntaxError(Exception):
        pass

    class _Env:
        def communicate(self, **kwargs):
            return "hello\n"

    class _Tools:
        config = types.SimpleNamespace(execution_timeout=1)

        def parse_actions(self, output):
            return "thought", "echo hello"

        def guard_multiline_input(self, action):
            return action

        def get_state(self, env):
            return {"ok": True}

        def check_for_submission_cmd(self, observation):
            return False

    publisher = _FakePublisher()
    manager = runner.SessionManager.__new__(runner.SessionManager)
    manager.worker_id = "worker-0"
    manager.sessions = {
        "sid": runner.Session(
            session_id="sid",
            instance_id="instance-0",
            env=_Env(),
            tools=_Tools(),
            started_at=0,
            last_used_at=0,
            agent_context=_trace_context(),
            tool_events_zmq_endpoint="tcp://trace:20390",
        )
    }
    manager._lock = threading.RLock()
    manager._valid_tools = {"bash"}
    manager._tools_config = types.SimpleNamespace(execution_timeout=1)
    manager._tool_event_publishers = {"tcp://trace:20390": publisher}
    manager._imports = {
        "CommandTimeoutError": CommandTimeoutError,
        "BashIncorrectSyntaxError": BashIncorrectSyntaxError,
    }

    result = manager.step({"session_id": "sid", "tool_call": {"id": "call-1", "function": {"name": "bash"}}})

    assert result["tool_error"] is None
    assert [record["event_type"] for record in publisher.records] == ["tool_start", "tool_end"]
    assert publisher.records[0]["tool"]["status"] == "running"
    assert publisher.records[1]["tool"]["status"] == "succeeded"
    assert publisher.records[1]["tool"]["tool_call_id"] == "call-1"
    assert publisher.records[1]["tool"]["output_bytes"] == len("hello\n".encode("utf-8"))
    trace_events = _session_trace_events(capsys.readouterr().out)
    assert [event["event"] for event in trace_events] == [
        "tool_step_request",
        "tool_start",
        "tool_step_response",
    ]
    assert trace_events[0]["trajectory_id"] == _trace_context()["trajectory_id"]
    assert trace_events[1]["tool_name"] == "bash"
    assert trace_events[1]["trace_published"] is True
    assert trace_events[2]["status"] == "succeeded"
    assert trace_events[2]["submitted"] is False


def test_invalid_tool_emits_dynamo_tool_error():
    runner = _load_session_runner()

    class _Tools:
        def get_state(self, env):
            return {"ok": True}

    publisher = _FakePublisher()
    manager = runner.SessionManager.__new__(runner.SessionManager)
    manager.worker_id = "worker-0"
    manager.sessions = {
        "sid": runner.Session(
            session_id="sid",
            instance_id="instance-0",
            env=object(),
            tools=_Tools(),
            started_at=0,
            last_used_at=0,
            agent_context=_trace_context(),
            tool_events_zmq_endpoint="tcp://trace:20390",
        )
    }
    manager._lock = threading.RLock()
    manager._valid_tools = {"str_replace_editor"}
    manager._tool_event_publishers = {"tcp://trace:20390": publisher}

    result = manager.step({"session_id": "sid", "tool_call": {"id": "call-bad", "function": {"name": "bash"}}})

    assert result["tool_error"] == "invalid_tool"
    assert [record["event_type"] for record in publisher.records] == ["tool_error"]
    assert publisher.records[0]["tool"]["error_type"] == "invalid_tool"


async def _health_completes_while_step_is_blocked(*, shared_semaphore: bool) -> bool:
    runner = _load_session_runner()
    manager = _BlockingManager()
    data_semaphore = asyncio.Semaphore(1)
    control_semaphore = data_semaphore if shared_semaphore else asyncio.Semaphore(1)

    step_msg = _Message({})
    step_task = asyncio.create_task(
        runner.handle_request(manager, "step", step_msg, data_semaphore, control_semaphore)
    )
    assert await asyncio.to_thread(manager.step_started.wait, 2)

    health_msg = _Message({})
    health_task = asyncio.create_task(
        runner.handle_request(manager, "health", health_msg, data_semaphore, control_semaphore)
    )
    try:
        await asyncio.wait_for(asyncio.shield(health_task), timeout=0.1)
        health_completed = True
    except TimeoutError:
        health_completed = False
    finally:
        manager.release_step.set()
        await asyncio.wait_for(step_task, timeout=2)
        await asyncio.wait_for(health_task, timeout=2)

    if health_completed:
        assert health_msg.responses == [{"method": "health", "error": None}]
    return health_completed


def test_session_health_uses_control_semaphore_when_step_workers_are_saturated():
    assert not asyncio.run(_health_completes_while_step_is_blocked(shared_semaphore=True))
    assert asyncio.run(_health_completes_while_step_is_blocked(shared_semaphore=False))
