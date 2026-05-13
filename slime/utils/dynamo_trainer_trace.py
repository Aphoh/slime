from __future__ import annotations

import hashlib
import os
import queue
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

TRACE_SCHEMA = "dynamo.agent.trace.v1"
TRACE_SOURCE = "harness"
DEFAULT_SESSION_TYPE_ID = "slime_swebench_pro"
DEFAULT_TOOL_EVENTS_TOPIC = "agent-tool-events"
DEFAULT_TOOL_EVENTS_PORT = "20390"

_PUBLISHER: _DynamoTracePublisher | None = None
_PUBLISHER_LOCK = threading.Lock()
_DISABLE_REASON_LOGGED = False


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _enabled() -> bool:
    return (_env("SLIME_DYNAMO_TRAINER_TRACE", "1") or "1").lower() not in {"0", "false", "no"}


def _endpoint() -> str | None:
    override = _env("SLIME_DYNAMO_TRAINER_TRACE_ZMQ_ENDPOINT") or _env("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_ENDPOINT")
    if override:
        return override
    frontend_url = _env("SWEPRO_DYNAMO_FRONTEND_URL") or _env("DYNAMO_FRONTEND_URL")
    if not frontend_url:
        return None
    parsed = urlparse(frontend_url if "://" in frontend_url else f"http://{frontend_url}")
    host = parsed.hostname
    if not host:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = _env("SWEPRO_DYNAMO_TOOL_EVENTS_ZMQ_PORT", DEFAULT_TOOL_EVENTS_PORT) or DEFAULT_TOOL_EVENTS_PORT
    return f"tcp://{host}:{port}"


def _context() -> dict[str, str]:
    run_id = _env("SWEPRO_RUN_ID") or _env("SLIME_RUN_ID") or "slime-trainer"
    return {
        "session_type_id": _env("SWEPRO_AGENT_TRACE_SESSION_TYPE_ID", DEFAULT_SESSION_TYPE_ID)
        or DEFAULT_SESSION_TYPE_ID,
        "session_id": run_id,
        "trajectory_id": _env("SLIME_DYNAMO_TRAINER_TRACE_TRAJECTORY_ID", f"{run_id}:trainer") or f"{run_id}:trainer",
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _tool_name_hash(tool_class: str) -> str:
    return hashlib.sha256(tool_class.encode("utf-8")).hexdigest()[:16]


class _DynamoTracePublisher:
    def __init__(self, endpoint: str, *, topic: str = DEFAULT_TOOL_EVENTS_TOPIC, capacity: int = 10000) -> None:
        self.endpoint = endpoint
        self.topic = topic
        self._queue: queue.Queue[dict | None] | None = None
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._msgpack = None
        self._zmq = None
        try:
            import msgpack  # type: ignore
            import zmq  # type: ignore
        except Exception as exc:
            _log_disable_reason(f"missing dependency: {type(exc).__name__}: {exc}")
            return
        self._msgpack = msgpack
        self._zmq = zmq
        self._queue = queue.Queue(maxsize=capacity)
        self._thread = threading.Thread(target=self._run, name="dynamo-trainer-trace", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._queue is not None

    def publish(self, record: dict) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._record_drop("queue_full")

    def _run(self) -> None:
        assert self._queue is not None
        assert self._msgpack is not None
        assert self._zmq is not None
        context = self._zmq.Context.instance()
        socket = context.socket(self._zmq.PUSH)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.SNDHWM, self._queue.maxsize)
        send_timeout_ms = int(_env("SLIME_DYNAMO_TRAINER_TRACE_ZMQ_SEND_TIMEOUT_MS", "100") or "100")
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
                f"dynamo-trainer-trace: dropped {dropped} event(s); reason={reason}",
                file=sys.stderr,
                flush=True,
            )


def _log_disable_reason(reason: str) -> None:
    global _DISABLE_REASON_LOGGED
    if _DISABLE_REASON_LOGGED:
        return
    _DISABLE_REASON_LOGGED = True
    print(f"dynamo-trainer-trace: disabled ({reason})", file=sys.stderr, flush=True)


def _publisher() -> _DynamoTracePublisher | None:
    global _PUBLISHER
    if not _enabled():
        return None
    endpoint = _endpoint()
    if not endpoint:
        return None
    with _PUBLISHER_LOCK:
        if _PUBLISHER is None or _PUBLISHER.endpoint != endpoint:
            _PUBLISHER = _DynamoTracePublisher(endpoint)
        return _PUBLISHER if _PUBLISHER.enabled else None


def start_trainer_span(name: str, **metadata: object) -> str | None:
    publisher = _publisher()
    if publisher is None:
        return None
    started_at_ms = _now_ms()
    tool_call_id = f"trainer:{name}:{uuid.uuid4().hex[:8]}"
    publisher.publish(
        _build_record(
            event_type="tool_start",
            name=name,
            tool_call_id=tool_call_id,
            event_time_unix_ms=started_at_ms,
            started_at_unix_ms=started_at_ms,
            status="running",
            metadata=metadata,
        )
    )
    return f"{tool_call_id}:{started_at_ms}"


def end_trainer_span(name: str, token: str | None, *, error_type: str | None = None, **metadata: object) -> None:
    publisher = _publisher()
    if publisher is None or token is None:
        return
    try:
        tool_call_id, started_at_raw = token.rsplit(":", 1)
        started_at_ms = int(started_at_raw)
    except ValueError:
        tool_call_id = token
        started_at_ms = _now_ms()
    ended_at_ms = _now_ms()
    publisher.publish(
        _build_record(
            event_type="tool_error" if error_type else "tool_end",
            name=name,
            tool_call_id=tool_call_id,
            event_time_unix_ms=ended_at_ms,
            started_at_unix_ms=started_at_ms,
            ended_at_unix_ms=ended_at_ms,
            status="error" if error_type else "succeeded",
            duration_ms=float(max(0, ended_at_ms - started_at_ms)),
            error_type=error_type,
            metadata=metadata,
        )
    )


def _build_record(
    *,
    event_type: str,
    name: str,
    tool_call_id: str,
    event_time_unix_ms: int,
    started_at_unix_ms: int | None = None,
    ended_at_unix_ms: int | None = None,
    status: str | None = None,
    duration_ms: float | None = None,
    error_type: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict:
    tool_class = f"trainer.{name}"
    tool = {
        "tool_call_id": tool_call_id,
        "tool_class": tool_class,
        "tool_name_hash": _tool_name_hash(tool_class),
    }
    optional = {
        "started_at_unix_ms": started_at_unix_ms,
        "ended_at_unix_ms": ended_at_unix_ms,
        "status": status,
        "duration_ms": duration_ms,
        "error_type": error_type,
        "metadata": metadata,
    }
    tool.update({key: value for key, value in optional.items() if value is not None})
    return {
        "schema": TRACE_SCHEMA,
        "event_type": event_type,
        "event_time_unix_ms": event_time_unix_ms,
        "event_source": TRACE_SOURCE,
        "agent_context": _context(),
        "tool": tool,
    }
