import builtins
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
sys.path.insert(0, str(EXAMPLE_DIR))

from dynamo_agent_trace import (  # noqa: E402
    DynamoToolEventPublisher,
    build_agent_context,
    build_agent_context_for_sample,
    build_tool_trace_record,
    derive_tool_events_zmq_endpoint,
    llm_request_id,
)


def test_derive_tool_events_zmq_endpoint_from_dynamo_url(monkeypatch):
    monkeypatch.delenv("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT", raising=False)
    monkeypatch.delenv("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT", raising=False)

    assert derive_tool_events_zmq_endpoint("http://warnold-swepro-frontend:3000") == (
        "tcp://warnold-swepro-frontend:20390"
    )


def test_derive_tool_events_zmq_endpoint_honors_override(monkeypatch):
    monkeypatch.setenv("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT", "tcp://trace-relay:20400")

    assert derive_tool_events_zmq_endpoint("http://warnold-swepro-frontend:3000") == "tcp://trace-relay:20400"


def test_build_agent_context_and_tool_record_match_dynamo_schema(monkeypatch):
    monkeypatch.setenv("SWEPRO_RUN_ID", "run-123")
    monkeypatch.delenv("SWEPRO_AGENT_TRACE_SESSION_TYPE_ID", raising=False)

    context = build_agent_context("instance-a", 5, sample_id="sample-a")
    record = build_tool_trace_record(
        agent_context=context,
        event_type="tool_end",
        tool_call_id="call-1",
        tool_class="bash",
        event_time_unix_ms=1000,
        started_at_unix_ms=900,
        ended_at_unix_ms=1000,
        status="succeeded",
        duration_ms=100.0,
        output_bytes=12,
    )

    assert context == {
        "session_type_id": "slime_swebench_pro",
        "session_id": "run-123",
        "trajectory_id": "run-123:swebench_pro:instance-a:sample:5:id:sample-a",
    }
    assert "phase" not in context
    assert llm_request_id(context, turn=3) == "run-123:swebench_pro:instance-a:sample:5:id:sample-a:llm:3"
    assert llm_request_id(context, turn=3, attempt=1).endswith(":llm:3:try:1")
    assert record["schema"] == "dynamo.agent.trace.v1"
    assert record["event_source"] == "harness"
    assert record["event_type"] == "tool_end"
    assert record["agent_context"] == context
    assert record["tool"]["tool_call_id"] == "call-1"
    assert record["tool"]["tool_class"] == "bash"
    assert record["tool"]["output_bytes"] == 12


def test_build_agent_context_uses_one_perfetto_lane_per_sample(monkeypatch):
    monkeypatch.setenv("SWEPRO_RUN_ID", "run-123")

    first_sample = SimpleNamespace(index=5, session_id="sample-a")
    first_turn = build_agent_context_for_sample("instance-a", first_sample)
    retry_turn = build_agent_context_for_sample("instance-a", first_sample)
    second_sample = build_agent_context_for_sample(
        "instance-a", SimpleNamespace(index=6, session_id="sample-b")
    )

    assert first_turn == retry_turn
    assert first_turn["session_id"] == second_sample["session_id"] == "run-123"
    assert first_turn["trajectory_id"] != second_sample["trajectory_id"]
    assert {
        (first_turn["session_id"], first_turn["trajectory_id"]),
        (retry_turn["session_id"], retry_turn["trajectory_id"]),
        (second_sample["session_id"], second_sample["trajectory_id"]),
    } == {
        ("run-123", "run-123:swebench_pro:instance-a:sample:5:id:sample-a"),
        ("run-123", "run-123:swebench_pro:instance-a:sample:6:id:sample-b"),
    }
    assert llm_request_id(first_turn, turn=0) == (
        "run-123:swebench_pro:instance-a:sample:5:id:sample-a:llm:0"
    )
    assert llm_request_id(first_turn, turn=1) == (
        "run-123:swebench_pro:instance-a:sample:5:id:sample-a:llm:1"
    )
    assert build_agent_context("instance-a", 7) == build_agent_context("instance-a", 7)


def test_tool_event_publisher_is_noop_without_endpoint():
    publisher = DynamoToolEventPublisher(None)

    assert publisher.enabled is False
    assert publisher.publish({"event_type": "tool_start"}) is False


def test_tool_event_publisher_is_noop_when_dependencies_are_missing(monkeypatch):
    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name in {"msgpack", "zmq"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    publisher = DynamoToolEventPublisher("tcp://trace:20390")

    assert publisher.enabled is False
    assert publisher.publish({"event_type": "tool_start"}) is False


def test_tool_event_publisher_sends_zmq_msgpack_frames():
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")
    context = zmq.Context.instance()
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://127.0.0.1:*")
    endpoint = pull.getsockopt_string(zmq.LAST_ENDPOINT)
    publisher = DynamoToolEventPublisher(endpoint, capacity=10)
    record = build_tool_trace_record(
        agent_context={
            "session_type_id": "slime_swebench_pro",
            "session_id": "run-1",
            "trajectory_id": "run-1:swebench_pro:instance-a:sample:5:id:sample-a",
        },
        event_type="tool_start",
        tool_call_id="call-1",
        tool_class="bash",
        event_time_unix_ms=1000,
        started_at_unix_ms=1000,
        status="running",
    )

    try:
        assert publisher.enabled is True
        assert publisher.publish(record) is True
        deadline = time.time() + 2
        frames = None
        while time.time() < deadline:
            if pull.poll(50):
                frames = pull.recv_multipart()
                break
        assert frames is not None
        topic, seq, payload = frames
        assert topic == b"agent-tool-events"
        assert int.from_bytes(seq, "big") == 0
        assert msgpack.unpackb(payload, raw=False) == record
    finally:
        publisher.close()
        pull.close(0)
