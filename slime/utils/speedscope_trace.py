from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterable

_LOCK = threading.Lock()
_HOST = socket.gethostname()


def _trace_path() -> str | None:
    value = os.getenv("SLIME_SPEEDSCOPE_TRACE_PATH") or os.getenv("SWEPRO_SPEEDSCOPE_TRACE_PATH")
    return value if value not in (None, "") else None


def _shard_enabled() -> bool:
    return os.getenv("SLIME_SPEEDSCOPE_TRACE_SHARD", "1").lower() not in {"0", "false", "no"}


def trace_enabled() -> bool:
    return _trace_path() is not None


def trace_now() -> float:
    return time.time()


def _json_default(value: Any) -> str:
    return str(value)


def _writer_path(path_value: str) -> Path:
    path = Path(path_value)
    if not _shard_enabled():
        return path
    return path.with_name(f"{path.stem}.{_HOST}.{os.getpid()}{path.suffix}")


def record_span(
    profile: str,
    name: str,
    start: float,
    end: float | None = None,
    **meta: Any,
) -> None:
    path_value = _trace_path()
    if not path_value:
        return

    end = trace_now() if end is None else end
    if end < start:
        end = start
    record = {
        "kind": "span",
        "profile": str(profile),
        "name": str(name),
        "start": float(start),
        "end": float(end),
        "pid": os.getpid(),
        "host": _HOST,
    }
    clean_meta = {key: value for key, value in meta.items() if value is not None}
    if clean_meta:
        record["meta"] = clean_meta

    path = _writer_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=_json_default, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


@contextlib.contextmanager
def trace_span(profile: str, name: str, **meta: Any):
    start = trace_now()
    try:
        yield
    finally:
        record_span(profile, name, start, **meta)


def _expand_input_paths(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            paths.append(path)

    for item in inputs:
        value = str(item)
        expanded = glob.glob(value) if any(ch in value for ch in "*?[") else [value]
        for path_value in expanded:
            path = Path(path_value)
            if path.is_dir():
                for child in sorted(path.glob("*.jsonl")):
                    add(child)
            else:
                add(path)
                if path.suffix:
                    for child in sorted(path.parent.glob(f"{path.stem}.*{path.suffix}")):
                        add(child)
    return paths


def _iter_spans(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path in _expand_input_paths(paths):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") != "span":
                    continue
                start = record.get("start")
                end = record.get("end")
                profile = record.get("profile")
                name = record.get("name")
                if start is None or end is None or not profile or not name:
                    continue
                try:
                    record["start"] = float(start)
                    record["end"] = max(float(end), float(start))
                except (TypeError, ValueError):
                    continue
                yield record


def _assign_non_overlapping_lanes(spans: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    lanes: list[tuple[float, list[dict[str, Any]]]] = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"], item["name"])):
        for lane_idx, (last_end, lane_spans) in enumerate(lanes):
            if span["start"] >= last_end:
                lane_spans.append(span)
                lanes[lane_idx] = (span["end"], lane_spans)
                break
        else:
            lanes.append((span["end"], [span]))
    return [(idx, lane_spans) for idx, (_, lane_spans) in enumerate(lanes)]


def write_speedscope_file(input_paths: Iterable[str | Path], output_path: str | Path, *, name: str = "slime") -> None:
    spans_by_profile: dict[str, list[dict[str, Any]]] = {}
    min_start: float | None = None
    max_end: float | None = None
    for span in _iter_spans(input_paths):
        spans_by_profile.setdefault(str(span["profile"]), []).append(span)
        min_start = span["start"] if min_start is None else min(min_start, span["start"])
        max_end = span["end"] if max_end is None else max(max_end, span["end"])

    frames: list[dict[str, Any]] = []
    frame_by_name: dict[str, int] = {}

    def frame_index(frame_name: str) -> int:
        idx = frame_by_name.get(frame_name)
        if idx is not None:
            return idx
        idx = len(frames)
        frame_by_name[frame_name] = idx
        frames.append({"name": frame_name})
        return idx

    start_value = min_start if min_start is not None else 0.0
    end_value = max_end if max_end is not None else start_value
    profiles: list[dict[str, Any]] = []

    for profile_name in sorted(spans_by_profile):
        lanes = _assign_non_overlapping_lanes(spans_by_profile[profile_name])
        for lane_idx, lane_spans in lanes:
            display_name = profile_name if len(lanes) == 1 else f"{profile_name} lane {lane_idx + 1}"
            events = []
            for span in lane_spans:
                idx = frame_index(str(span["name"]))
                events.append({"type": "O", "at": span["start"], "frame": idx})
                events.append({"type": "C", "at": span["end"], "frame": idx})
            profiles.append(
                {
                    "type": "evented",
                    "name": display_name,
                    "unit": "seconds",
                    "startValue": start_value,
                    "endValue": end_value,
                    "events": events,
                }
            )

    output = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {"frames": frames},
        "profiles": profiles,
        "name": name,
        "activeProfileIndex": 0 if profiles else None,
        "exporter": "slime-speedscope-trace@1",
    }
    if output["activeProfileIndex"] is None:
        del output["activeProfileIndex"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


def _perfetto_category(profile: str) -> str:
    return profile.split("/", 1)[0] if "/" in profile else profile


def _rollout_profile_info(profile: str) -> dict[str, Any] | None:
    parts = profile.split("/")
    if len(parts) < 4 or parts[0] != "rollout":
        return None
    try:
        sample_index = int(parts[2])
    except ValueError:
        return None
    return {
        "instance_id": parts[1],
        "sample_index": sample_index,
        "trace_id": parts[3],
    }


def _task_label(instance_id: Any) -> str:
    return str(instance_id or "unknown-task")


def _session_id_from_span(span: dict[str, Any]) -> str | None:
    meta = span.get("meta")
    if isinstance(meta, dict) and meta.get("session_id"):
        return str(meta["session_id"])
    profile = str(span.get("profile") or "")
    parts = profile.split("/")
    if len(parts) >= 3 and parts[0] == "session":
        return parts[2]
    return None


def _worker_id_from_profile(profile: str) -> str | None:
    parts = profile.split("/")
    if len(parts) >= 3 and parts[0] == "session":
        return parts[1]
    return None


def _session_to_rollout_profile(spans: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for span in spans:
        profile = str(span.get("profile") or "")
        if not _rollout_profile_info(profile):
            continue
        session_id = _session_id_from_span(span)
        if session_id and session_id not in mapping:
            mapping[session_id] = profile
    return mapping


def _associated_rollout_profile(span: dict[str, Any], session_to_rollout: dict[str, str]) -> str | None:
    profile = str(span.get("profile") or "")
    if _rollout_profile_info(profile):
        return profile
    session_id = _session_id_from_span(span)
    if session_id:
        return session_to_rollout.get(session_id)
    return None


def _logical_process_key(
    span: dict[str, Any],
    *,
    layout: str,
    session_to_rollout: dict[str, str],
    samples_per_group: int | None,
    groups_per_batch: int | None,
) -> tuple[str, str]:
    rollout_profile = _associated_rollout_profile(span, session_to_rollout)
    info = _rollout_profile_info(rollout_profile) if rollout_profile else None
    if info is None:
        host = str(span.get("host") or "unknown-host")
        pid = str(span.get("pid") or "unknown-pid")
        return ("unmapped", f"unmapped / {host} pid {pid}")

    sample_index = int(info["sample_index"])
    instance_id = _task_label(info["instance_id"])
    if layout == "sample":
        return (f"sample:{sample_index}", f"sample {sample_index} / {instance_id}")

    if samples_per_group is None:
        raise ValueError(f"{layout} layout requires samples_per_group")
    sample_group = sample_index // samples_per_group
    if layout == "sample-group":
        return (f"group:{sample_group}", f"group {sample_group} / {instance_id}")

    if groups_per_batch is None:
        raise ValueError("rollout-batch layout requires groups_per_batch")
    rollout_batch = sample_group // groups_per_batch
    return (f"rollout-batch:{rollout_batch}", f"rollout batch {rollout_batch}")


def _thread_name_for_span(
    span: dict[str, Any],
    session_to_rollout: dict[str, str],
    *,
    samples_per_group: int | None,
) -> str:
    profile = str(span.get("profile") or "unknown")
    rollout_info = _rollout_profile_info(profile)
    if rollout_info is not None:
        sample_index = int(rollout_info["sample_index"])
        sample_group = sample_index // samples_per_group if samples_per_group else None
        group_label = f" group {sample_group}" if sample_group is not None else ""
        return (
            f"rollout sample {sample_index}{group_label} / "
            f"{_task_label(rollout_info['instance_id'])} / {rollout_info['trace_id']}"
        )
    worker_id = _worker_id_from_profile(profile)
    session_id = _session_id_from_span(span)
    if session_id:
        associated = session_to_rollout.get(session_id)
        associated_info = _rollout_profile_info(associated) if associated else None
        if associated_info is not None:
            sample_index = int(associated_info["sample_index"])
            sample_group = sample_index // samples_per_group if samples_per_group else None
            group_label = f" group {sample_group}" if sample_group is not None else ""
            task = _task_label(associated_info["instance_id"])
            if worker_id:
                return f"session sample {sample_index}{group_label} / {task} / {worker_id} / {session_id[:8]}"
            return f"session sample {sample_index}{group_label} / {task} / {session_id[:8]}"
    if worker_id and session_id:
        return f"session worker {worker_id} / {session_id[:8]}"
    return profile


def _sample_context(
    span: dict[str, Any],
    *,
    session_to_rollout: dict[str, str],
    samples_per_group: int | None,
    groups_per_batch: int | None,
) -> dict[str, Any]:
    rollout_profile = _associated_rollout_profile(span, session_to_rollout)
    info = _rollout_profile_info(rollout_profile) if rollout_profile else None
    if info is None:
        return {}
    sample_index = int(info["sample_index"])
    context: dict[str, Any] = {
        "instance_id": _task_label(info["instance_id"]),
        "sample_index": sample_index,
    }
    if samples_per_group:
        sample_group = sample_index // samples_per_group
        context["sample_group"] = sample_group
        if groups_per_batch:
            context["rollout_batch"] = sample_group // groups_per_batch
    return context


def _perfetto_event_name(span: dict[str, Any]) -> str:
    name = str(span["name"])
    if name != "inference.complete":
        return name
    meta = span.get("meta")
    if not isinstance(meta, dict):
        return name
    parts = [name]
    if meta.get("turn") is not None:
        parts.append(f"turn={meta['turn']}")
    if meta.get("prompt_tokens") is not None:
        parts.append(f"prompt={meta['prompt_tokens']}")
    completion_tokens = meta.get("completion_tokens", meta.get("generated_tokens"))
    if completion_tokens is not None:
        parts.append(f"completion={completion_tokens}")
    elif meta.get("max_tokens") is not None:
        parts.append(f"max={meta['max_tokens']}")
    return " ".join(parts)


def write_perfetto_file(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    name: str = "slime",
    layout: str = "physical",
    samples_per_group: int | None = None,
    groups_per_batch: int | None = None,
) -> None:
    if layout not in {"physical", "sample", "sample-group", "rollout-batch"}:
        raise ValueError(f"unknown Perfetto layout: {layout}")
    if samples_per_group is not None and samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    if groups_per_batch is not None and groups_per_batch <= 0:
        raise ValueError("groups_per_batch must be positive")

    spans = sorted(_iter_spans(input_paths), key=lambda item: (item["start"], item["end"], item["profile"], item["name"]))
    min_start = min((span["start"] for span in spans), default=0.0)
    session_to_rollout = _session_to_rollout_profile(spans)

    process_keys: list[tuple[str, str]] = []
    process_key_to_pid: dict[tuple[str, str], int] = {}
    for span in spans:
        if layout == "physical":
            key = (str(span.get("host") or "unknown-host"), f"{span.get('host') or 'unknown-host'} pid {span.get('pid') or 'unknown-pid'}")
        else:
            key = _logical_process_key(
                span,
                layout=layout,
                session_to_rollout=session_to_rollout,
                samples_per_group=samples_per_group,
                groups_per_batch=groups_per_batch,
            )
        if key not in process_key_to_pid:
            process_key_to_pid[key] = len(process_key_to_pid) + 1
            process_keys.append(key)

    metadata_events: list[dict[str, Any]] = []
    thread_ids: dict[tuple[str, str], dict[str, int]] = {}
    emitted_threads: set[tuple[int, int]] = set()

    for process_sort_index, key in enumerate(process_keys):
        pid = process_key_to_pid[key]
        _, display_name = key
        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": display_name},
            }
        )
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": process_sort_index},
            }
        )

    trace_events: list[dict[str, Any]] = []
    for span in spans:
        if layout == "physical":
            process_key = (
                str(span.get("host") or "unknown-host"),
                f"{span.get('host') or 'unknown-host'} pid {span.get('pid') or 'unknown-pid'}",
            )
        else:
            process_key = _logical_process_key(
                span,
                layout=layout,
                session_to_rollout=session_to_rollout,
                samples_per_group=samples_per_group,
                groups_per_batch=groups_per_batch,
            )
        pid = process_key_to_pid[process_key]
        profile = str(span["profile"])
        thread_name = (
            profile
            if layout == "physical"
            else _thread_name_for_span(span, session_to_rollout, samples_per_group=samples_per_group)
        )
        tids_for_process = thread_ids.setdefault(process_key, {})
        tid = tids_for_process.setdefault(thread_name, len(tids_for_process) + 1)
        thread_key = (pid, tid)
        if thread_key not in emitted_threads:
            emitted_threads.add(thread_key)
            metadata_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": thread_name},
                }
            )
            metadata_events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": tid},
                }
            )

        args = {
            "profile": profile,
            "host": span.get("host"),
            "source_pid": span.get("pid"),
            "layout": layout,
        }
        args.update(
            _sample_context(
                span,
                session_to_rollout=session_to_rollout,
                samples_per_group=samples_per_group,
                groups_per_batch=groups_per_batch,
            )
        )
        meta = span.get("meta")
        if isinstance(meta, dict) and meta:
            args["meta"] = meta
            for key in ("turn", "prompt_tokens", "completion_tokens", "generated_tokens", "max_tokens", "session_id"):
                if key in meta:
                    args[key] = meta[key]

        trace_events.append(
            {
                "name": _perfetto_event_name(span),
                "cat": _perfetto_category(profile),
                "ph": "X",
                "ts": int(round((span["start"] - min_start) * 1_000_000)),
                "dur": max(0, int(round((span["end"] - span["start"]) * 1_000_000))),
                "pid": pid,
                "tid": tid,
                "args": args,
            }
        )

    output = {
        "traceEvents": metadata_events + trace_events,
        "displayTimeUnit": "s",
        "otherData": {
            "name": name,
            "exporter": "slime-speedscope-trace@1",
            "startWallTimeSeconds": min_start,
            "spanCount": len(spans),
            "layout": layout,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, default=_json_default, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert slime span JSONL traces to speedscope and Perfetto JSON.")
    parser.add_argument("inputs", nargs="+", help="Span JSONL files, directories, or globs.")
    parser.add_argument("-o", "--output", help="Output .speedscope.json path.")
    parser.add_argument("--perfetto-output", help="Output Perfetto/Chrome trace JSON path.")
    parser.add_argument(
        "--perfetto-layout",
        choices=("physical", "sample", "sample-group", "rollout-batch"),
        default="physical",
        help="Perfetto track grouping. physical groups by source process; logical layouts group by rollout sample/group/batch.",
    )
    parser.add_argument("--n-samples-per-prompt", type=int, help="Needed for sample-group and rollout-batch layouts.")
    parser.add_argument("--rollout-batch-size", type=int, help="Needed for rollout-batch layout.")
    parser.add_argument("--name", default="slime", help="Profile group name.")
    args = parser.parse_args()
    if not args.output and not args.perfetto_output:
        parser.error("at least one of --output or --perfetto-output is required")
    if args.output:
        write_speedscope_file(args.inputs, args.output, name=args.name)
    if args.perfetto_output:
        write_perfetto_file(
            args.inputs,
            args.perfetto_output,
            name=args.name,
            layout=args.perfetto_layout,
            samples_per_group=args.n_samples_per_prompt,
            groups_per_batch=args.rollout_batch_size,
        )


if __name__ == "__main__":
    main()
