"""Helpers for Dynamo agent tracing in the SWE-bench Pro harness."""

from __future__ import annotations

import hashlib
import os
import queue
import sys
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlparse

TRACE_SCHEMA = "dynamo.agent.trace.v1"
TRACE_SOURCE = "harness"
DEFAULT_SESSION_TYPE_ID = "slime_swebench_pro"
DEFAULT_TOOL_EVENTS_TOPIC = "agent-tool-events"
DEFAULT_TOOL_EVENTS_PORT = "20390"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def build_agent_context(instance_id: Any, sample_index: Any, *, suffix: str | None = None) -> dict[str, str]:
    session_id = _env("SWEPRO_RUN_ID")
    if not session_id:
        session_id = f"swepro_{uuid.uuid4().hex[:12]}"
        os.environ["SWEPRO_RUN_ID"] = session_id
    suffix = suffix or uuid.uuid4().hex[:8]
    return {
        "session_type_id": _env("SWEPRO_AGENT_TRACE_SESSION_TYPE_ID", DEFAULT_SESSION_TYPE_ID)
        or DEFAULT_SESSION_TYPE_ID,
        "session_id": session_id,
        "trajectory_id": f"{session_id}:swebench_pro:{instance_id}:{sample_index}:{suffix}",
    }


def llm_request_id(agent_context: dict[str, Any] | None, *, turn: int, attempt: int | None = None) -> str:
    trajectory_id = (agent_context or {}).get("trajectory_id") or "unknown-trajectory"
    prefix = f"{trajectory_id}:llm:{turn}"
    return f"{prefix}:try:{attempt}" if attempt is not None else prefix


def derive_tool_events_zmq_endpoint(dynamo_frontend_url: str | None) -> str | None:
    override = _env("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT")
    if override:
        return override
    if not dynamo_frontend_url:
        return None
    parsed = urlparse(dynamo_frontend_url if "://" in dynamo_frontend_url else f"http://{dynamo_frontend_url}")
    host = parsed.hostname
    if not host:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = _env("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT", DEFAULT_TOOL_EVENTS_PORT) or DEFAULT_TOOL_EVENTS_PORT
    return f"tcp://{host}:{port}"


def tool_name_hash(tool_class: str) -> str:
    return hashlib.sha256(tool_class.encode("utf-8")).hexdigest()[:16]


def now_ms() -> int:
    return int(time.time() * 1000)


def build_tool_trace_record(
    *,
    agent_context: dict[str, Any],
    event_type: str,
    tool_call_id: str,
    tool_class: str,
    event_time_unix_ms: int | None = None,
    started_at_unix_ms: int | None = None,
    ended_at_unix_ms: int | None = None,
    status: str | None = None,
    duration_ms: float | None = None,
    output_tokens: int | None = None,
    output_bytes: int | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    tool = {
        "tool_call_id": str(tool_call_id),
        "tool_class": str(tool_class),
        "tool_name_hash": tool_name_hash(str(tool_class)),
    }
    optional = {
        "started_at_unix_ms": started_at_unix_ms,
        "ended_at_unix_ms": ended_at_unix_ms,
        "status": status,
        "duration_ms": duration_ms,
        "output_tokens": output_tokens,
        "output_bytes": output_bytes,
        "error_type": error_type,
    }
    tool.update({key: value for key, value in optional.items() if value is not None})
    return {
        "schema": TRACE_SCHEMA,
        "event_type": event_type,
        "event_time_unix_ms": event_time_unix_ms if event_time_unix_ms is not None else now_ms(),
        "event_source": TRACE_SOURCE,
        "agent_context": dict(agent_context),
        "tool": tool,
    }


class DynamoToolEventPublisher:
    """Best-effort PUSH publisher for Dynamo harness tool events."""

    def __init__(self, endpoint: str | None, *, topic: str = DEFAULT_TOOL_EVENTS_TOPIC, capacity: int = 10000) -> None:
        self.endpoint = endpoint
        self.topic = topic
        self._queue: queue.Queue[dict[str, Any] | None] | None = None
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._msgpack = None
        self._zmq = None
        if not endpoint:
            return
        try:
            import msgpack  # type: ignore
            import zmq  # type: ignore
        except Exception:
            return
        self._msgpack = msgpack
        self._zmq = zmq
        self._queue = queue.Queue(maxsize=capacity)
        self._thread = threading.Thread(target=self._run, name="dynamo-tool-events", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._queue is not None

    @property
    def dropped(self) -> int:
        return self._dropped

    def publish(self, record: dict[str, Any]) -> bool:
        if self._queue is None:
            return False
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self._record_drop("queue_full")
            return False

    def close(self) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        assert self._queue is not None
        assert self._msgpack is not None
        assert self._zmq is not None
        context = self._zmq.Context.instance()
        socket = context.socket(self._zmq.PUSH)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.SNDHWM, self._queue.maxsize)
        send_timeout_ms = int(_env("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_SEND_TIMEOUT_MS", "100") or "100")
        socket.setsockopt(self._zmq.SNDTIMEO, send_timeout_ms)
        socket.connect(self.endpoint)
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    return
                seq = self._seq
                self._seq += 1
                payload = self._msgpack.packb(record, use_bin_type=True)
                frames = [self.topic.encode("utf-8"), seq.to_bytes(8, "big"), payload]
                try:
                    socket.send_multipart(frames)
                except Exception as exc:
                    self._record_drop(type(exc).__name__)
        finally:
            socket.close(0)

    def _record_drop(self, reason: str) -> None:
        with self._drop_lock:
            self._dropped += 1
            dropped = self._dropped
        if dropped == 1 or dropped & (dropped - 1) == 0:
            print(
                f"dynamo-agent-trace: dropped {dropped} tool event(s); reason={reason}",
                file=sys.stderr,
                flush=True,
            )
