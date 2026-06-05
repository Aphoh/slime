from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from dataclasses import dataclass, field
import importlib
import sys
import types
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLE_DIR))

from trace_replay import TraceReplaySessionClient, TraceReplayStore  # noqa: E402
from trace_replay_workload import _write_replay_jsonl  # noqa: E402


@dataclass
class _SampleStub:
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    label: str | None = None
    response: str = ""
    metadata: dict = field(default_factory=dict)


def _import_generate_hook(monkeypatch):
    for name in [
        "generate_with_swebench_pro",
        "slime",
        "slime.rollout",
        "slime.rollout.sglang_rollout",
        "slime.utils",
        "slime.utils.mask_utils",
        "slime.utils.types",
    ]:
        sys.modules.pop(name, None)

    slime_mod = types.ModuleType("slime")
    slime_mod.__path__ = []
    rollout_mod = types.ModuleType("slime.rollout")
    rollout_mod.__path__ = []
    sglang_rollout_mod = types.ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout_mod.GenerateState = object
    utils_mod = types.ModuleType("slime.utils")
    utils_mod.__path__ = []
    mask_utils_mod = types.ModuleType("slime.utils.mask_utils")
    mask_utils_mod.MultiTurnLossMaskGenerator = object
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _SampleStub

    monkeypatch.setitem(sys.modules, "slime", slime_mod)
    monkeypatch.setitem(sys.modules, "slime.rollout", rollout_mod)
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", sglang_rollout_mod)
    monkeypatch.setitem(sys.modules, "slime.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.mask_utils", mask_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)
    return importlib.import_module("generate_with_swebench_pro")


def _request_end(
    trajectory_id: str,
    *,
    turn: int,
    attempt: int = 0,
    input_tokens: int,
    output_tokens: int,
    received_ms: int,
    total_ms: int,
    session_id: str = "run-1",
) -> dict:
    return {
        "timestamp": received_ms + total_ms,
        "event": {
            "schema": "dynamo.agent.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": received_ms + total_ms,
            "event_source": "dynamo",
            "agent_context": {"session_id": session_id, "trajectory_id": trajectory_id},
            "request": {
                "request_id": f"req-{turn}",
                "x_request_id": f"{trajectory_id}:llm:{turn}:try:{attempt}",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": 0,
                "request_received_ms": received_ms,
                "total_time_ms": total_ms,
            },
        },
    }


def _tool_start(
    trajectory_id: str,
    *,
    tool_call_id: str,
    tool_class: str,
    started_ms: int,
    session_id: str = "run-1",
) -> dict:
    return {
        "event": {
            "schema": "dynamo.agent.trace.v1",
            "event_type": "tool_start",
            "event_time_unix_ms": started_ms,
            "event_source": "harness",
            "agent_context": {"session_id": session_id, "trajectory_id": trajectory_id},
            "tool": {
                "tool_call_id": tool_call_id,
                "tool_class": tool_class,
                "started_at_unix_ms": started_ms,
                "status": "running",
            },
        },
    }


def _tool_end(
    trajectory_id: str,
    *,
    tool_call_id: str,
    tool_class: str,
    started_ms: int,
    ended_ms: int,
    session_id: str = "run-1",
    status: str = "succeeded",
    submitted: bool = False,
    submission: str = "",
) -> dict:
    return {
        "event": {
            "schema": "dynamo.agent.trace.v1",
            "event_type": "tool_end",
            "event_time_unix_ms": ended_ms,
            "event_source": "harness",
            "agent_context": {"session_id": session_id, "trajectory_id": trajectory_id},
            "tool": {
                "tool_call_id": tool_call_id,
                "tool_class": tool_class,
                "started_at_unix_ms": started_ms,
                "ended_at_unix_ms": ended_ms,
                "status": status,
                "submitted": submitted,
                "submission": submission,
                "duration_ms": ended_ms - started_ms,
                "output_bytes": 80,
                "output_tokens": 17,
            },
        },
    }


def test_trace_replay_store_parses_agent_trace_shapes_and_tool_durations(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trajectory_id = "run-1:swebench_pro:inst-a:sample:7:id:sid-1"
    records = [
        _request_end(trajectory_id, turn=0, input_tokens=100, output_tokens=20, received_ms=1_000, total_ms=1_000),
        _tool_start(trajectory_id, tool_call_id="call-1", tool_class="bash", started_ms=3_000),
        _tool_end(trajectory_id, tool_call_id="call-1", tool_class="bash", started_ms=3_000, ended_ms=8_000),
        _request_end(trajectory_id, turn=1, input_tokens=150, output_tokens=11, received_ms=9_000, total_ms=1_000),
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records))

    store = TraceReplayStore.from_path(trace_path, sleep_scale=0.0)
    plan = store.claim("inst-a")

    assert plan.trajectory_id == trajectory_id
    assert plan.instance_id == "inst-a"
    assert plan.session_id == "run-1"
    assert plan.sample_index is None
    assert plan.initial_prompt_tokens == 100
    assert plan.submit_duration_s == 0
    assert [turn.generated_tokens for turn in plan.turns] == [20, 11]
    assert [turn.backend_generated_tokens for turn in plan.turns] == [20, 11]
    assert plan.turns[0].tool_name == "bash"
    assert plan.turns[0].tool_duration_s == 5
    assert plan.turns[0].observation_tokens == 30
    assert plan.turns[1].observation_tokens == 0


def test_trace_replay_preserves_request_retries_as_separate_turns(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    trajectory_id = "run-1:swebench_pro:inst-a:sample:7:id:sid-1"
    records = [
        _request_end(
            trajectory_id,
            turn=0,
            attempt=0,
            input_tokens=100,
            output_tokens=20,
            received_ms=1_000,
            total_ms=1_000,
        ),
        _request_end(
            trajectory_id,
            turn=0,
            attempt=1,
            input_tokens=100,
            output_tokens=9,
            received_ms=2_500,
            total_ms=500,
        ),
        _request_end(
            trajectory_id,
            turn=1,
            attempt=0,
            input_tokens=140,
            output_tokens=11,
            received_ms=4_000,
            total_ms=1_000,
        ),
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records))

    plan = TraceReplayStore.from_path(trace_path, sleep_scale=0.0).claim("inst-a")
    _write_replay_jsonl(replay_path, [plan])
    reparsed = TraceReplayStore.from_path(replay_path, sleep_scale=0.0).claim("inst-a")

    assert [(turn.turn, turn.attempt, turn.generated_tokens) for turn in plan.turns] == [
        (0, 0, 20),
        (0, 1, 9),
        (1, 0, 11),
    ]
    assert [(turn.turn, turn.attempt, turn.generated_tokens) for turn in reparsed.turns] == [
        (0, 0, 20),
        (0, 1, 9),
        (1, 0, 11),
    ]


def test_trace_replay_workload_preserves_submitted_tool_end(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    trajectory_id = "run-1:swebench_pro:inst-a:sample:0:id:sid-1"
    submission = "diff --git a/file.py b/file.py\n"
    records = [
        _request_end(trajectory_id, turn=0, input_tokens=100, output_tokens=10, received_ms=1_000, total_ms=1_000),
        _tool_start(trajectory_id, tool_call_id="call-1", tool_class="bash", started_ms=2_000),
        _tool_end(
            trajectory_id,
            tool_call_id="call-1",
            tool_class="bash",
            started_ms=2_000,
            ended_ms=3_000,
            submitted=True,
            submission=submission,
        ),
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records))

    plan = TraceReplayStore.from_path(trace_path, sleep_scale=0.0).claim("inst-a")
    _write_replay_jsonl(replay_path, [plan])
    reparsed = TraceReplayStore.from_path(replay_path, sleep_scale=0.0).claim("inst-a")
    client = TraceReplaySessionClient(reparsed, sleep_scale=0.0)

    async def _run():
        started = await client.start()
        return await client.step(started["session_id"], reparsed.turns[0].tool_call())

    step = asyncio.run(_run())

    assert reparsed.turns[0].submitted is True
    assert reparsed.turns[0].submission == submission
    assert reparsed.turns[0].observation_tokens == 17
    assert step["submitted"] is True
    assert step["submission"] == submission


def test_trace_replay_session_client_sleeps_and_returns_mock_submission(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trajectory_id = "run-1:swebench_pro:inst-a:sample:0:id:sid-1"
    records = [
        _request_end(trajectory_id, turn=0, input_tokens=100, output_tokens=10, received_ms=1_000, total_ms=1_000),
        _tool_start(
            trajectory_id,
            tool_call_id="call-1",
            tool_class="str_replace_editor",
            started_ms=2_000,
        ),
        _tool_end(
            trajectory_id,
            tool_call_id="call-1",
            tool_class="str_replace_editor",
            started_ms=2_000,
            ended_ms=3_000,
        ),
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records))
    plan = TraceReplayStore.from_path(trace_path, sleep_scale=0.0).claim("inst-a")
    client = TraceReplaySessionClient(plan, sleep_scale=0.0)

    async def _run():
        started = await client.start()
        step = await client.step(started["session_id"], plan.turns[0].tool_call())
        submit = await client.submit(started["session_id"])
        return started, step, submit

    started, step, submit = asyncio.run(_run())

    assert started["session_id"] == "run-1"
    assert started["tools"][0]["function"]["name"] == "str_replace_editor"
    assert step["submitted"] is False
    assert step["trace_replay_tool_duration_s"] == 1
    assert submit["submission"].startswith("diff --git")


def test_trace_replay_store_parses_raw_dynamo_events(tmp_path):
    trace_path = tmp_path / "dynamo.jsonl"
    trajectory_id = "run-1:swebench_pro:inst-a:sample:0:id:sid-a"
    request_id = f"{trajectory_id}:llm:0:try:0"
    records = [
        {
            "timestamp": 10,
            "event": {
                "schema": "dynamo.agent.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": 12_500,
                "event_source": "dynamo",
                "agent_context": {"session_id": "run-1", "trajectory_id": trajectory_id},
                "request": {
                    "request_id": "req-1",
                    "x_request_id": request_id,
                    "input_tokens": 123,
                    "output_tokens": 45,
                    "cached_tokens": 64,
                    "kv_hit_rate": 0.5,
                    "request_received_ms": 10_000,
                    "total_time_ms": 2_500,
                    "ttft_ms": 800,
                    "prefill_time_ms": 700,
                    "prefill_wait_time_ms": 100,
                    "queue_depth": 3,
                },
            },
        },
        {
            "event": {
                "schema": "dynamo.agent.trace.v1",
                "event_type": "tool_start",
                "event_time_unix_ms": 13_000,
                "event_source": "harness",
                "agent_context": {"session_id": "run-1", "trajectory_id": trajectory_id},
                "tool": {
                    "tool_call_id": "call_0_bash",
                    "tool_class": "bash",
                    "started_at_unix_ms": 13_000,
                    "status": "running",
                },
            },
        },
        {
            "event": {
                "schema": "dynamo.agent.trace.v1",
                "event_type": "tool_end",
                "event_time_unix_ms": 16_250,
                "event_source": "harness",
                "agent_context": {"session_id": "run-1", "trajectory_id": trajectory_id},
                "tool": {
                    "tool_call_id": "call_0_bash",
                    "tool_class": "bash",
                    "started_at_unix_ms": 13_000,
                    "ended_at_unix_ms": 16_250,
                    "status": "succeeded",
                    "duration_ms": 3_250,
                    "output_bytes": 80,
                },
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records))

    plan = TraceReplayStore.from_path(trace_path, sleep_scale=0.0).claim("inst-a")

    assert plan.trajectory_id == trajectory_id
    assert plan.instance_id == "inst-a"
    assert len(plan.turns) == 1
    turn = plan.turns[0]
    assert turn.prompt_tokens == 123
    assert turn.generated_tokens == 45
    assert turn.model_duration_s == 2.5
    assert turn.ttft_s == 0.8
    assert turn.prefill_time_s == 0.7
    assert turn.prefill_wait_time_s == 0.1
    assert turn.cached_tokens == 64
    assert turn.kv_hit_rate == 0.5
    assert turn.tool_name == "bash"
    assert turn.tool_duration_s == 3.25
    assert plan.duration_s == 6.25


def test_trace_replay_store_rejects_legacy_flattened_events(tmp_path):
    trace_path = tmp_path / "legacy.jsonl"
    trace_path.write_text(
        '{"event":"model_request","trajectory_id":"traj-1","instance_id":"inst-a",'
        '"turn":0,"prompt_tokens":100}\n'
        '{"event":"model_response","trajectory_id":"traj-1","instance_id":"inst-a",'
        '"turn":0,"generated_tokens":10}\n'
    )

    with pytest.raises(ValueError, match="dynamo.agent.trace.v1"):
        TraceReplayStore.from_path(trace_path, sleep_scale=0.0)


def test_trace_replay_reward_bypasses_nats(monkeypatch):
    generate_with_swebench_pro = _import_generate_hook(monkeypatch)
    monkeypatch.setenv("SWEPRO_MOCK_ENV_REWARD", "0.25")
    sample = generate_with_swebench_pro.Sample(
        label="inst-a",
        response="diff --git a/file.py b/file.py\n",
        metadata={
            "trace_replay": True,
            "trace_replay_trajectory_id": "traj-1",
            "instance_id": "inst-a",
            "patch": "diff --git a/file.py b/file.py\n",
        },
    )

    reward = asyncio.run(generate_with_swebench_pro.reward_func(SimpleNamespace(), sample))

    assert reward == 0.25
    assert sample.metadata["raw_reward"] == 0.25
    assert sample.metadata["eval"] == {
        "passed": True,
        "reward": 0.25,
        "mock_trace_replay": True,
        "reason": "trace replay reward bypass",
    }
