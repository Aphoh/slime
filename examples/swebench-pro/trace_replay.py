"""Deterministic SWE-agent trace replay helpers.

The replay path uses a recorded agent trace to keep router ablations honest:
real `/v1/completions` requests still go to Dynamo/SGLang, but environment
steps are mocked with the same tool durations and token-shape progression.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_LLM_TURN_RE = re.compile(r":llm:(\d+)(?::try:(\d+))?$")
_MOCK_SUBMISSION = """diff --git a/mock_trace_replay.txt b/mock_trace_replay.txt
new file mode 100644
index 0000000..d4c3b2a
--- /dev/null
+++ b/mock_trace_replay.txt
@@ -0,0 +1 @@
+trace replay mock submission
"""


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _ms_to_s(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return numeric / 1000.0


def _payload_timestamp(payload: dict[str, Any]) -> float | None:
    return (
        _parse_timestamp(payload.get("timestamp"))
        or _parse_timestamp(payload.get("ts"))
        or _parse_timestamp(payload.get("time"))
    )


def _turn_attempt_from_request_id(x_request_id: Any) -> tuple[int, int] | None:
    if not x_request_id:
        return None
    match = _LLM_TURN_RE.search(str(x_request_id))
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2) or 0)
    except ValueError:
        return None


def _turn_from_request_id(x_request_id: Any) -> int | None:
    parsed = _turn_attempt_from_request_id(x_request_id)
    return parsed[0] if parsed is not None else None


def _event_key(payload: dict[str, Any]) -> tuple[str, str]:
    trajectory_id = str(payload.get("trajectory_id") or payload.get("session_id") or "unknown")
    instance_id = str(payload.get("instance_id") or "")
    return trajectory_id, instance_id


def _instance_id_from_trajectory(trajectory_id: str) -> str:
    match = re.search(r":swebench_pro:(.+?):sample:", trajectory_id)
    return match.group(1) if match else ""


def _normalize_dynamo_event(outer: dict[str, Any]) -> list[tuple[dict[str, Any], float | None]]:
    event = outer.get("event")
    if not isinstance(event, dict) or event.get("schema") != "dynamo.agent.trace.v1":
        return []

    ctx = event.get("agent_context") or {}
    trajectory_id = str(ctx.get("trajectory_id") or "")
    if not trajectory_id or trajectory_id.endswith(":trainer"):
        return []
    instance_id = _instance_id_from_trajectory(trajectory_id)
    timestamp = _ms_to_s(event.get("event_time_unix_ms"))
    if timestamp is None:
        timestamp = _payload_timestamp(outer)
    event_type = event.get("event_type")

    if event_type == "request_end":
        request = event.get("request") or {}
        turn_attempt = _turn_attempt_from_request_id(request.get("x_request_id"))
        if turn_attempt is None:
            return []
        turn, attempt = turn_attempt
        request_ts = _ms_to_s(request.get("request_received_ms"))
        total_s = _ms_to_s(request.get("total_time_ms"))
        if request_ts is None and timestamp is not None and total_s is not None:
            request_ts = timestamp - total_s
        response_ts = timestamp
        common = {
            "trajectory_id": trajectory_id,
            "instance_id": instance_id,
            "turn": turn,
            "turn_attempt": attempt,
            "session_id": ctx.get("session_id"),
            "x_request_id": request.get("x_request_id"),
            "dynamo_request_id": request.get("request_id"),
        }
        request_payload = {
            "event": "model_request",
            "prompt_tokens": int(request.get("input_tokens") or 0),
            "max_tokens": int(request.get("requested_max_tokens") or request.get("output_tokens") or 0),
            "cached_tokens": int(request.get("cached_tokens") or 0),
            "kv_hit_rate": request.get("kv_hit_rate"),
            **common,
        }
        response_payload = {
            "event": "model_response",
            "generated_tokens": int(request.get("output_tokens") or 0),
            "backend_generated_tokens": int(request.get("output_tokens") or 0),
            "finish_reason": request.get("finish_reason") or "stop",
            "stop_reason": request.get("stop_reason"),
            "model_duration_s": total_s,
            "ttft_s": _ms_to_s(request.get("ttft_ms")),
            "prefill_time_s": _ms_to_s(request.get("prefill_time_ms")),
            "prefill_wait_time_s": _ms_to_s(request.get("prefill_wait_time_ms")),
            "cached_tokens": int(request.get("cached_tokens") or 0),
            "kv_hit_rate": request.get("kv_hit_rate"),
            "queue_depth": request.get("queue_depth"),
            **common,
        }
        return [(request_payload, request_ts), (response_payload, response_ts)]

    if event_type in {"tool_start", "tool_end", "tool_error"}:
        tool = event.get("tool") or {}
        started_ts = _ms_to_s(tool.get("started_at_unix_ms"))
        if started_ts is None:
            started_ts = timestamp
        ended_ts = _ms_to_s(tool.get("ended_at_unix_ms"))
        if ended_ts is None:
            ended_ts = timestamp
        tool_name = tool.get("tool_class")
        payload = {
            "event": "tool_step_request" if event_type == "tool_start" else "tool_step_response",
            "trajectory_id": trajectory_id,
            "instance_id": instance_id,
            "session_id": ctx.get("session_id"),
            "tool_call_id": tool.get("tool_call_id"),
            "tool_name": tool_name,
            "submitted": bool(tool.get("submitted")),
            "submission": tool.get("submission") or "",
            "tool_error": tool.get("error_type") if event_type == "tool_error" else None,
            "tool_status": tool.get("status"),
            "observation_bytes": tool.get("output_bytes"),
            "observation_tokens": tool.get("output_tokens"),
            "_unindexed_tool_event": True,
        }
        if event_type in {"tool_end", "tool_error"} and tool.get("duration_ms") is not None:
            payload["tool_duration_s"] = _ms_to_s(tool.get("duration_ms"))
        return [(payload, started_ts if event_type == "tool_start" else ended_ts)]

    return []


def _extract_trace_records(line: str) -> list[tuple[dict[str, Any], float | None]]:
    payload_text = line.strip()
    if not payload_text.startswith("{"):
        return []
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return []
    return _normalize_dynamo_event(payload)


def _turn(payload: dict[str, Any]) -> int | None:
    value = payload.get("turn")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _turn_key(payload: dict[str, Any]) -> tuple[int, int] | None:
    turn = _turn(payload)
    if turn is None:
        return None
    try:
        attempt = int(payload.get("turn_attempt") or 0)
    except (TypeError, ValueError):
        attempt = 0
    return turn, attempt


def _duration_s(start: float | None, end: float | None) -> float:
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


@dataclass(frozen=True)
class TraceReplayTurn:
    turn: int
    prompt_tokens: int
    generated_tokens: int
    attempt: int = 0
    backend_generated_tokens: int | None = None
    model_duration_s: float = 0.0
    ttft_s: float = 0.0
    prefill_time_s: float = 0.0
    prefill_wait_time_s: float = 0.0
    cached_tokens: int = 0
    kv_hit_rate: float | None = None
    finish_reason: str = "stop"
    stop_reason: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_duration_s: float = 0.0
    observation_tokens: int = 0
    submitted: bool = False
    submission: str = ""
    model_start_offset_s: float = 0.0
    tool_start_offset_s: float = 0.0

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_name)

    def tool_call(self) -> dict[str, Any]:
        name = self.tool_name or "bash"
        return {
            "id": self.tool_call_id or f"trace-replay-tool-{self.turn}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"command": f"trace replay turn {self.turn}"}),
            },
        }


@dataclass(frozen=True)
class TraceReplayPlan:
    trajectory_id: str
    instance_id: str
    sample_index: int | None
    session_id: str | None
    initial_prompt_tokens: int
    tool_count: int
    turns: tuple[TraceReplayTurn, ...]
    submit_duration_s: float = 0.0
    close_duration_s: float = 0.0
    duration_s: float = 0.0

    def turn_for(self, turn_index: int) -> TraceReplayTurn | None:
        if 0 <= turn_index < len(self.turns):
            return self.turns[turn_index]
        return None

    @property
    def tools(self) -> list[dict[str, Any]]:
        names = [turn.tool_name for turn in self.turns if turn.tool_name]
        if not names:
            names = ["bash", "str_replace_editor"]
        unique_names = list(dict.fromkeys(str(name) for name in names))
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Trace replay mock tool: {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "cmd": {"type": "string"},
                            "path": {"type": "string"},
                        },
                    },
                },
            }
            for name in unique_names
        ]


def _build_plan(
    trajectory_id: str,
    instance_id: str,
    records: list[tuple[int, float | None, dict[str, Any]]],
) -> TraceReplayPlan | None:
    records = sorted(records, key=lambda item: item[0])
    session_started: dict[str, Any] | None = None
    session_id: str | None = None
    sample_index: int | None = None
    initial_prompt_tokens = 0
    tool_count = 0
    request_by_turn: dict[tuple[int, int], dict[str, Any]] = {}
    response_by_turn: dict[tuple[int, int], dict[str, Any]] = {}
    tool_req_by_turn: dict[tuple[int, int], dict[str, Any]] = {}
    tool_resp_by_turn: dict[tuple[int, int], dict[str, Any]] = {}
    unindexed_tool_reqs: list[dict[str, Any]] = []
    unindexed_tool_resps: list[dict[str, Any]] = []
    submit_request_ts: float | None = None
    submit_response_ts: float | None = None
    close_request_ts: float | None = None
    close_response_ts: float | None = None
    timestamps = [timestamp for _, timestamp, _ in records if timestamp is not None]
    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None

    for _, timestamp, payload in records:
        event = payload.get("event")
        if event == "session_started":
            session_started = payload
            session_id = str(payload.get("session_id") or "") or None
            sample_index = _turn({"turn": payload.get("sample_index")})
            initial_prompt_tokens = int(payload.get("prompt_tokens") or 0)
            tool_count = int(payload.get("tool_count") or 0)
        elif event == "model_request" and (turn_key := _turn_key(payload)) is not None:
            payload = dict(payload)
            payload["_trace_ts"] = timestamp
            request_by_turn[turn_key] = payload
        elif event == "model_response" and (turn_key := _turn_key(payload)) is not None:
            payload = dict(payload)
            payload["_trace_ts"] = timestamp
            response_by_turn[turn_key] = payload
        elif event == "tool_step_request" and (turn_key := _turn_key(payload)) is not None:
            payload = dict(payload)
            payload["_trace_ts"] = timestamp
            tool_req_by_turn[turn_key] = payload
        elif event == "tool_step_request":
            payload = dict(payload)
            payload["_trace_ts"] = timestamp
            unindexed_tool_reqs.append(payload)
        elif event == "tool_step_response" and (turn_key := _turn_key(payload)) is not None:
            payload = dict(payload)
            payload["_trace_ts"] = timestamp
            tool_resp_by_turn[turn_key] = payload
        elif event == "tool_step_response":
            payload = dict(payload)
            payload["_trace_ts"] = timestamp
            unindexed_tool_resps.append(payload)
        elif event == "submit_request":
            submit_request_ts = timestamp
        elif event == "submit_response":
            submit_response_ts = timestamp
        elif event == "session_close_request":
            close_request_ts = timestamp
        elif event == "session_close_response":
            close_response_ts = timestamp

    turns: list[TraceReplayTurn] = []
    sorted_turn_keys = sorted(request_by_turn)
    for idx, turn_key in enumerate(sorted_turn_keys):
        turn, attempt = turn_key
        request = request_by_turn[turn_key]
        response = response_by_turn.get(turn_key, {})
        tool_req = tool_req_by_turn.get(turn_key, {})
        tool_resp = tool_resp_by_turn.get(turn_key, {})
        if not tool_req and idx < len(unindexed_tool_reqs):
            tool_req = unindexed_tool_reqs[idx]
        if not tool_resp and idx < len(unindexed_tool_resps):
            tool_resp = unindexed_tool_resps[idx]
        prompt_tokens = int(request.get("prompt_tokens") or 0)
        generated_tokens = int(response.get("generated_tokens") or response.get("backend_generated_tokens") or 0)
        backend_generated_tokens = int(response.get("backend_generated_tokens") or generated_tokens or 0)
        next_prompt_tokens = None
        if idx + 1 < len(sorted_turn_keys):
            next_prompt_tokens = int(request_by_turn[sorted_turn_keys[idx + 1]].get("prompt_tokens") or 0)
        observation_tokens = 0
        if next_prompt_tokens is not None:
            observation_tokens = max(0, next_prompt_tokens - prompt_tokens - generated_tokens)
        elif tool_resp.get("observation_tokens") is not None:
            observation_tokens = int(tool_resp.get("observation_tokens") or 0)
        tool_name = tool_req.get("tool_name") or None
        request_ts = request.get("_trace_ts")
        response_ts = response.get("_trace_ts")
        tool_req_ts = tool_req.get("_trace_ts")
        tool_resp_ts = tool_resp.get("_trace_ts")
        model_duration_s = _coerce_float(response.get("model_duration_s"))
        if model_duration_s is None:
            model_duration_s = _duration_s(request_ts, response_ts)
        tool_duration_s = _coerce_float(tool_resp.get("tool_duration_s"))
        if tool_duration_s is None:
            tool_duration_s = _duration_s(tool_req_ts, tool_resp_ts)
        model_start_offset_s = _duration_s(first_ts, request_ts)
        tool_start_offset_s = _duration_s(first_ts, tool_req_ts)
        inferred_submitted = bool(tool_resp.get("submitted"))
        submission = str(tool_resp.get("submission") or "") or (_MOCK_SUBMISSION if inferred_submitted else "")
        turns.append(
            TraceReplayTurn(
                turn=turn,
                attempt=attempt,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                backend_generated_tokens=backend_generated_tokens,
                model_duration_s=model_duration_s,
                ttft_s=_coerce_float(response.get("ttft_s")) or 0.0,
                prefill_time_s=_coerce_float(response.get("prefill_time_s")) or 0.0,
                prefill_wait_time_s=_coerce_float(response.get("prefill_wait_time_s")) or 0.0,
                cached_tokens=int(response.get("cached_tokens") or request.get("cached_tokens") or 0),
                kv_hit_rate=_coerce_float(response.get("kv_hit_rate") or request.get("kv_hit_rate")),
                finish_reason=str(response.get("finish_reason") or "stop"),
                stop_reason=str(response.get("stop_reason")) if response.get("stop_reason") is not None else None,
                tool_name=str(tool_name) if tool_name else None,
                tool_call_id=str(tool_req.get("tool_call_id") or "") or (f"trace-replay-tool-{turn}-try-{attempt}" if attempt else None),
                tool_duration_s=tool_duration_s,
                observation_tokens=observation_tokens,
                submitted=inferred_submitted,
                submission=submission,
                model_start_offset_s=model_start_offset_s,
                tool_start_offset_s=tool_start_offset_s,
            )
        )

    if not turns:
        return None
    if session_started is None:
        first_request = request_by_turn[sorted_turn_keys[0]]
        session_id = str(first_request.get("session_id") or "") or None
        sample_index = _turn({"turn": first_request.get("sample_index")})
        initial_prompt_tokens = int(first_request.get("prompt_tokens") or 0)
    return TraceReplayPlan(
        trajectory_id=trajectory_id,
        instance_id=instance_id,
        sample_index=sample_index,
        session_id=session_id,
        initial_prompt_tokens=initial_prompt_tokens,
        tool_count=tool_count,
        turns=tuple(turns),
        submit_duration_s=_duration_s(submit_request_ts, submit_response_ts),
        close_duration_s=_duration_s(close_request_ts, close_response_ts),
        duration_s=_duration_s(first_ts, last_ts),
    )


class TraceReplayStore:
    def __init__(self, plans: list[TraceReplayPlan], *, sleep_scale: float = 1.0):
        if not plans:
            raise ValueError("trace replay file did not contain any dynamo.agent.trace.v1 request_end events")
        self.plans = tuple(plans)
        self.sleep_scale = max(0.0, float(sleep_scale))
        self._by_instance: dict[str, deque[TraceReplayPlan]] = defaultdict(deque)
        for plan in self.plans:
            if plan.instance_id:
                self._by_instance[plan.instance_id].append(plan)
        self._round_robin = deque(self.plans)
        self._lock = threading.Lock()

    @classmethod
    def from_path(cls, path: str | Path, *, sleep_scale: float = 1.0) -> "TraceReplayStore":
        groups: dict[tuple[str, str], list[tuple[int, float | None, dict[str, Any]]]] = defaultdict(list)
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", errors="replace") as handle:
            for idx, line in enumerate(handle):
                for payload, timestamp in _extract_trace_records(line):
                    if not payload or not payload.get("event"):
                        continue
                    groups[_event_key(payload)].append((idx, timestamp, payload))
        plans = [
            plan
            for (trajectory_id, instance_id), records in groups.items()
            if (plan := _build_plan(trajectory_id, instance_id, records)) is not None
        ]
        return cls(plans, sleep_scale=sleep_scale)

    def claim(self, instance_id: str | None = None) -> TraceReplayPlan:
        with self._lock:
            if instance_id and self._by_instance.get(instance_id):
                plans = self._by_instance[instance_id]
                plan = plans[0]
                plans.rotate(-1)
                return plan
            plan = self._round_robin[0]
            self._round_robin.rotate(-1)
            return plan


class TraceReplaySessionClient:
    def __init__(self, plan: TraceReplayPlan, *, sleep_scale: float = 1.0):
        self.plan = plan
        self.sleep_scale = max(0.0, float(sleep_scale))
        self._step_index = 0

    async def health(self) -> bool:
        return True

    async def start(self, **kwargs) -> dict[str, Any]:
        session_id = self.plan.session_id or f"trace-replay-{self.plan.trajectory_id}"
        return {
            "session_id": session_id,
            "tools": self.plan.tools,
            "mock_trace_replay": True,
            "trace_replay_trajectory_id": self.plan.trajectory_id,
            "trace_replay_original_session_id": self.plan.session_id,
        }

    async def step(self, session_id: str, tool_call: dict[str, Any], *, thought: str = "") -> dict[str, Any]:
        turn = self.plan.turn_for(self._step_index)
        self._step_index += 1
        if turn is None:
            return {"observation": "", "submitted": False}
        if turn.tool_duration_s > 0 and self.sleep_scale > 0:
            await asyncio.sleep(turn.tool_duration_s * self.sleep_scale)
        observation = (
            f"[trace replay observation turn={turn.turn} "
            f"tokens={turn.observation_tokens} tool={turn.tool_name or 'none'}]"
        )
        return {
            "observation": observation,
            "submitted": turn.submitted,
            "submission": turn.submission,
            "mock_trace_replay": True,
            "trace_replay_tool_duration_s": turn.tool_duration_s,
            "trace_replay_observation_tokens": turn.observation_tokens,
        }

    async def submit(self, session_id: str) -> dict[str, Any]:
        if self.plan.submit_duration_s > 0 and self.sleep_scale > 0:
            await asyncio.sleep(self.plan.submit_duration_s * self.sleep_scale)
        return {
            "submission": _MOCK_SUBMISSION,
            "mock_trace_replay": True,
            "patch_diagnostics": {"patch_chars": len(_MOCK_SUBMISSION), "patch_file_count": 1},
        }

    async def close(self, session_id: str) -> dict[str, Any]:
        if self.plan.close_duration_s > 0 and self.sleep_scale > 0:
            await asyncio.sleep(self.plan.close_duration_s * self.sleep_scale)
        return {"closed": True, "mock_trace_replay": True}
