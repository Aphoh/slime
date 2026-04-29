import json

from slime.utils.speedscope_trace import record_span, trace_span, write_perfetto_file, write_speedscope_file


def test_trace_span_writes_jsonl(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_PATH", str(trace_path))

    with trace_span("rollout/example", "inference.complete", turn=1):
        pass

    shard_paths = list(tmp_path.glob("trace.*.jsonl"))
    assert len(shard_paths) == 1
    rows = [json.loads(line) for line in shard_paths[0].read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "span"
    assert rows[0]["profile"] == "rollout/example"
    assert rows[0]["name"] == "inference.complete"
    assert rows[0]["meta"]["turn"] == 1
    assert rows[0]["end"] >= rows[0]["start"]


def test_speedscope_export_assigns_lanes_for_overlapping_spans(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "trace.speedscope.json"
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_PATH", str(trace_path))

    record_span("session/sid", "tool.run bash", 10.0, 20.0)
    record_span("session/sid", "session.close", 15.0, 16.0)
    record_span("trainer/rank0", "trainer.batch.train", 12.0, 18.0)

    write_speedscope_file([trace_path], output_path, name="test")

    data = json.loads(output_path.read_text())
    assert data["$schema"] == "https://www.speedscope.app/file-format-schema.json"
    assert data["shared"]["frames"]
    profile_names = {profile["name"] for profile in data["profiles"]}
    assert "session/sid lane 1" in profile_names
    assert "session/sid lane 2" in profile_names
    assert "trainer/rank0" in profile_names
    for profile in data["profiles"]:
        assert profile["type"] == "evented"
        assert len(profile["events"]) % 2 == 0
        stack = []
        for event in profile["events"]:
            if event["type"] == "O":
                stack.append(event["frame"])
            else:
                assert stack.pop() == event["frame"]
        assert not stack


def test_perfetto_export_preserves_parallel_profiles(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "trace.perfetto.json"
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_PATH", str(trace_path))

    record_span("rollout/sample-a", "inference.complete", 10.0, 20.0, turn=1)
    record_span("session/sample-a", "tool.run bash", 12.0, 18.0, command="pytest")
    record_span("trainer/rank0", "weights.update", 13.0, 14.5)

    write_perfetto_file([trace_path], output_path, name="test")

    data = json.loads(output_path.read_text())
    assert data["displayTimeUnit"] == "s"
    assert data["otherData"]["spanCount"] == 3

    span_events = [event for event in data["traceEvents"] if event["ph"] == "X"]
    assert {event["name"] for event in span_events} == {"inference.complete turn=1", "tool.run bash", "weights.update"}
    assert {event["cat"] for event in span_events} == {"rollout", "session", "trainer"}
    assert min(event["ts"] for event in span_events) == 0
    assert any(event["dur"] == 10_000_000 for event in span_events)
    assert any(event["args"].get("meta", {}).get("command") == "pytest" for event in span_events)

    thread_names = [
        event["args"]["name"]
        for event in data["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    ]
    assert {"rollout/sample-a", "session/sample-a", "trainer/rank0"}.issubset(thread_names)


def test_perfetto_export_can_group_session_worker_spans_by_rollout_batch(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "trace.perfetto.json"
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_SHARD", "0")

    session_id = "session-123456789"
    record_span("rollout/instance-a/5/traceabcd", "inference.complete", 10.0, 11.0, session_id=session_id)
    record_span("session/worker-a/session-123456789", "tool.run bash", 11.0, 13.0, session_id=session_id)
    record_span("rollout/instance-b/10/traceefgh", "inference.complete", 12.0, 14.0)

    write_perfetto_file(
        [trace_path],
        output_path,
        name="test",
        layout="rollout-batch",
        samples_per_group=4,
        groups_per_batch=2,
    )

    data = json.loads(output_path.read_text())
    process_names = [
        event["args"]["name"]
        for event in data["traceEvents"]
        if event["ph"] == "M" and event["name"] == "process_name"
    ]
    assert "rollout batch 0" in process_names
    assert "rollout batch 1" in process_names

    batch_pid_by_name = {
        event["args"]["name"]: event["pid"]
        for event in data["traceEvents"]
        if event["ph"] == "M" and event["name"] == "process_name"
    }
    batch0_pid = batch_pid_by_name["rollout batch 0"]
    batch0_spans = [
        event
        for event in data["traceEvents"]
        if event.get("ph") == "X" and event["pid"] == batch0_pid
    ]
    assert {event["name"] for event in batch0_spans} == {"inference.complete", "tool.run bash"}

    thread_names = [
        event["args"]["name"]
        for event in data["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name" and event["pid"] == batch0_pid
    ]
    assert "rollout sample 5 group 1 / instance-a / traceabcd" in thread_names
    assert "session sample 5 group 1 / instance-a / worker-a / session-" in thread_names


def test_perfetto_export_surfaces_inference_token_counts(monkeypatch, tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "trace.perfetto.json"
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SLIME_SPEEDSCOPE_TRACE_SHARD", "0")

    record_span(
        "rollout/instance-a/0/traceabcd",
        "inference.complete",
        10.0,
        11.0,
        turn=3,
        prompt_tokens=123,
        completion_tokens=45,
        max_tokens=8192,
    )

    write_perfetto_file([trace_path], output_path, name="test", layout="rollout-batch", samples_per_group=4, groups_per_batch=16)

    data = json.loads(output_path.read_text())
    span_events = [event for event in data["traceEvents"] if event.get("ph") == "X"]
    assert len(span_events) == 1
    assert span_events[0]["name"] == "inference.complete turn=3 prompt=123 completion=45"
    assert span_events[0]["args"]["turn"] == 3
    assert span_events[0]["args"]["prompt_tokens"] == 123
    assert span_events[0]["args"]["completion_tokens"] == 45
    assert span_events[0]["args"]["max_tokens"] == 8192
