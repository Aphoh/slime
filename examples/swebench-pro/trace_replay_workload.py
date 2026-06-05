"""Build and visualize SWE-bench Pro trace replay workloads."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from pathlib import Path
from statistics import median
from typing import Iterable

from trace_replay import TraceReplayPlan, TraceReplayStore


TRACE_SCHEMA = "dynamo.agent.trace.v1"
REPLAY_BASE_UNIX_MS = 1_800_000_000_000


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)

    def q(frac: float) -> float:
        return values[min(len(values) - 1, int(frac * (len(values) - 1)))]

    return {
        "min": values[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": values[-1],
    }


def _load_plans(paths: Iterable[Path]) -> list[TraceReplayPlan]:
    plans_by_id: dict[str, TraceReplayPlan] = {}
    for path in paths:
        store = TraceReplayStore.from_path(path, sleep_scale=0.0)
        for plan in store.plans:
            plans_by_id.setdefault(plan.trajectory_id, plan)
    return list(plans_by_id.values())


def _plan_metrics(plan: TraceReplayPlan) -> dict[str, float | int | str]:
    total_model_s = sum(turn.model_duration_s for turn in plan.turns)
    total_tool_s = sum(turn.tool_duration_s for turn in plan.turns)
    total_duration_s = plan.duration_s or (total_model_s + total_tool_s + plan.submit_duration_s + plan.close_duration_s)
    generated = [turn.generated_tokens for turn in plan.turns]
    prompts = [turn.prompt_tokens for turn in plan.turns]
    cached = [turn.cached_tokens for turn in plan.turns]
    return {
        "trajectory_id": plan.trajectory_id,
        "instance_id": plan.instance_id,
        "turns": len(plan.turns),
        "tool_turns": sum(1 for turn in plan.turns if turn.has_tool_call),
        "sum_prompt_tokens": sum(prompts),
        "max_prompt_tokens": max(prompts) if prompts else 0,
        "sum_generated_tokens": sum(generated),
        "max_generated_tokens": max(generated) if generated else 0,
        "sum_cached_tokens": sum(cached),
        "total_model_s": total_model_s,
        "total_tool_s": total_tool_s,
        "total_duration_s": total_duration_s,
        "max_tool_s": max([turn.tool_duration_s for turn in plan.turns] or [0.0]),
        "submit_duration_s": plan.submit_duration_s,
        "close_duration_s": plan.close_duration_s,
    }


def _passes_filters(plan: TraceReplayPlan, args: argparse.Namespace) -> tuple[bool, str | None]:
    metrics = _plan_metrics(plan)
    checks = [
        ("max_turns", metrics["turns"], args.max_turns),
        ("max_generated_tokens_per_turn", metrics["max_generated_tokens"], args.max_generated_tokens_per_turn),
        ("max_total_generated_tokens", metrics["sum_generated_tokens"], args.max_total_generated_tokens),
        ("max_prompt_tokens", metrics["max_prompt_tokens"], args.max_prompt_tokens),
        ("max_total_duration_s", metrics["total_duration_s"], args.max_total_duration_s),
        ("max_model_duration_s", metrics["total_model_s"], args.max_model_duration_s),
        ("max_tool_duration_s", metrics["total_tool_s"], args.max_tool_duration_s),
    ]
    for name, value, limit in checks:
        if limit is not None and float(value) > float(limit):
            return False, f"{name}>{limit}"
    if args.min_turns is not None and int(metrics["turns"]) < args.min_turns:
        return False, f"turns<{args.min_turns}"
    return True, None


def _session_id(plan: TraceReplayPlan) -> str:
    if plan.session_id:
        return plan.session_id
    marker = ":swebench_pro:"
    if marker in plan.trajectory_id:
        return plan.trajectory_id.split(marker, 1)[0]
    return "trace-replay"


def _ms(offset_s: float) -> int:
    return REPLAY_BASE_UNIX_MS + int(max(0.0, offset_s) * 1000)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def _dynamo_events_for_plan(plan: TraceReplayPlan) -> list[dict]:
    events: list[dict] = []
    session_id = _session_id(plan)
    for turn in plan.turns:
        model_start = max(0.0, turn.model_start_offset_s)
        model_end = model_start + max(0.0, turn.model_duration_s)
        tool_start = turn.tool_start_offset_s if turn.tool_start_offset_s > 0 else model_end
        tool_end = tool_start + max(0.0, turn.tool_duration_s)
        agent_context = {
            "session_type_id": "slime_swebench_pro",
            "session_id": session_id,
            "trajectory_id": plan.trajectory_id,
        }
        x_request_id = f"{plan.trajectory_id}:llm:{turn.turn}:try:{turn.attempt}"
        events.append(
            {
                "timestamp": model_end,
                "event": {
                    "schema": TRACE_SCHEMA,
                    "event_type": "request_end",
                    "event_time_unix_ms": _ms(model_end),
                    "event_source": "dynamo",
                    "agent_context": agent_context,
                    "request": {
                        "request_id": f"trace-replay-{_stable_id(plan.trajectory_id, turn.turn, turn.attempt)}",
                        "x_request_id": x_request_id,
                        "input_tokens": turn.prompt_tokens,
                        "output_tokens": turn.generated_tokens,
                        "cached_tokens": turn.cached_tokens,
                        "request_received_ms": _ms(model_start),
                        "prefill_wait_time_ms": turn.prefill_wait_time_s * 1000,
                        "prefill_time_ms": turn.prefill_time_s * 1000,
                        "ttft_ms": turn.ttft_s * 1000,
                        "total_time_ms": turn.model_duration_s * 1000,
                        "kv_hit_rate": turn.kv_hit_rate,
                        "queue_depth": 0,
                        "finish_reason": turn.finish_reason,
                        "stop_reason": turn.stop_reason,
                    },
                },
            }
        )
        if turn.tool_name:
            tool_call_id = turn.tool_call_id or f"trace-replay-tool-{turn.turn}-try-{turn.attempt}"
            events.append(
                {
                    "timestamp": tool_start,
                    "event": {
                        "schema": TRACE_SCHEMA,
                        "event_type": "tool_start",
                        "event_time_unix_ms": _ms(tool_start),
                        "event_source": "harness",
                        "agent_context": agent_context,
                        "tool": {
                            "tool_call_id": tool_call_id,
                            "tool_class": turn.tool_name,
                            "started_at_unix_ms": _ms(tool_start),
                            "status": "running",
                        },
                    },
                }
            )
            events.append(
                {
                    "timestamp": tool_end,
                    "event": {
                        "schema": TRACE_SCHEMA,
                        "event_type": "tool_end",
                        "event_time_unix_ms": _ms(tool_end),
                        "event_source": "harness",
                        "agent_context": agent_context,
                        "tool": {
                            "tool_call_id": tool_call_id,
                            "tool_class": turn.tool_name,
                            "started_at_unix_ms": _ms(tool_start),
                            "ended_at_unix_ms": _ms(tool_end),
                            "status": "succeeded",
                            "duration_ms": turn.tool_duration_s * 1000,
                            "output_tokens": turn.observation_tokens,
                            "submitted": turn.submitted,
                            "submission": turn.submission,
                        },
                    },
                }
            )
    return events


def _write_replay_jsonl(path: Path, plans: list[TraceReplayPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for plan in plans:
            for event in _dynamo_events_for_plan(plan):
                handle.write(json.dumps(event, sort_keys=True) + "\n")


def _bar_svg(values: list[float], *, width: int, height: int, color: str) -> str:
    if not values:
        return ""
    bins = 24
    max_value = max(values) or 1.0
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value / max_value) * bins))
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_w = width / bins
    rects = []
    for idx, count in enumerate(counts):
        h = (count / max_count) * height
        rects.append(
            f'<rect x="{idx * bar_w:.1f}" y="{height - h:.1f}" width="{max(1, bar_w - 1):.1f}" '
            f'height="{h:.1f}" fill="{color}" />'
        )
    return "\n".join(rects)


def _timeline_svg(plans: list[TraceReplayPlan], *, max_rows: int = 90) -> str:
    selected = sorted(plans, key=lambda plan: _plan_metrics(plan)["total_duration_s"], reverse=True)[:max_rows]
    row_h = 18
    left = 220
    width = 980
    height = max(1, len(selected)) * row_h + 40
    max_duration = max((_plan_metrics(plan)["total_duration_s"] for plan in selected), default=1.0)
    scale = width / max_duration if max_duration else 1.0
    parts = [
        f'<svg viewBox="0 0 {left + width + 220} {height}" xmlns="http://www.w3.org/2000/svg" role="img">',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;fill:#1f2937}'
        '.muted{fill:#64748b}.llm{fill:#2f8f9d}.tool{fill:#d970a8}.submit{fill:#8a8f2d}</style>',
        f'<text x="{left}" y="14" class="muted">duration axis, longest selected trajectory = {max_duration:.1f}s</text>',
    ]
    for row, plan in enumerate(selected):
        y = 30 + row * row_h
        metrics = _plan_metrics(plan)
        label = plan.instance_id[:32] if plan.instance_id else plan.trajectory_id[-32:]
        parts.append(f'<text x="0" y="{y + 11}">{html.escape(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 8}" x2="{left + width}" y2="{y + 8}" stroke="#e5e7eb" />')
        for turn in plan.turns:
            model_x = left + turn.model_start_offset_s * scale
            model_w = max(1.0, turn.model_duration_s * scale)
            title = (
                f"turn {turn.turn}: LLM in={turn.prompt_tokens} out={turn.generated_tokens} "
                f"duration={turn.model_duration_s:.2f}s"
            )
            parts.append(
                f'<rect class="llm" x="{model_x:.1f}" y="{y}" width="{model_w:.1f}" height="7">'
                f"<title>{html.escape(title)}</title></rect>"
            )
            if turn.tool_name:
                tool_x = left + turn.tool_start_offset_s * scale
                tool_w = max(1.0, turn.tool_duration_s * scale)
                title = f"turn {turn.turn}: Tool {turn.tool_name} duration={turn.tool_duration_s:.2f}s"
                parts.append(
                    f'<rect class="tool" x="{tool_x:.1f}" y="{y + 8}" width="{tool_w:.1f}" height="7">'
                    f"<title>{html.escape(title)}</title></rect>"
                )
        parts.append(
            f'<text x="{left + width + 8}" y="{y + 11}" class="muted">'
            f"{int(metrics['sum_generated_tokens'])} out, {metrics['total_duration_s']:.1f}s</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _scatter_svg(metrics: list[dict[str, float | int | str]], *, width: int = 560, height: int = 300) -> str:
    if not metrics:
        return ""
    pad = 36
    xs = [float(item["sum_generated_tokens"]) for item in metrics]
    ys = [float(item["total_duration_s"]) for item in metrics]
    max_x = max(xs) or 1.0
    max_y = max(ys) or 1.0
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;fill:#475569}</style>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#94a3b8"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#94a3b8"/>',
        f'<text x="{pad}" y="{height-8}">generated tokens</text>',
        f'<text x="4" y="{pad-10}">seconds</text>',
    ]
    for item in metrics:
        x = pad + (float(item["sum_generated_tokens"]) / max_x) * (width - 2 * pad)
        y = height - pad - (float(item["total_duration_s"]) / max_y) * (height - 2 * pad)
        tool_frac = float(item["total_tool_s"]) / max(0.001, float(item["total_duration_s"]))
        color = "#d970a8" if tool_frac > 0.5 else "#2f8f9d"
        title = (
            f"{item['instance_id']}\\n"
            f"out={item['sum_generated_tokens']} duration={float(item['total_duration_s']):.2f}s "
            f"tool={float(item['total_tool_s']):.2f}s"
        )
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"><title>{html.escape(title)}</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)


def _write_html(path: Path, source_plans: list[TraceReplayPlan], selected_plans: list[TraceReplayPlan], summary: dict) -> None:
    selected_metrics = [_plan_metrics(plan) for plan in selected_plans]
    source_metrics = [_plan_metrics(plan) for plan in source_plans]
    token_values = [float(item["sum_generated_tokens"]) for item in selected_metrics]
    duration_values = [float(item["total_duration_s"]) for item in selected_metrics]
    max_turn_values = [float(item["max_generated_tokens"]) for item in selected_metrics]
    stats_table = "".join(
        f"<tr><th>{html.escape(name)}</th><td><pre>{html.escape(json.dumps(value, indent=2))}</pre></td></tr>"
        for name, value in summary.get("selected_quantiles", {}).items()
    )
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>SWE-Pro Trace Replay Workload Shape</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; background: #fbfcfe; }}
h1, h2 {{ margin: 0 0 12px; }}
section {{ margin: 22px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }}
.metric {{ border: 1px solid #dbe3ef; border-radius: 6px; padding: 10px 12px; background: white; }}
.metric b {{ display: block; font-size: 22px; }}
.panel {{ border: 1px solid #dbe3ef; border-radius: 6px; padding: 14px; background: white; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; vertical-align: top; border-top: 1px solid #e5e7eb; padding: 8px; }}
pre {{ white-space: pre-wrap; margin: 0; }}
.legend span {{ display: inline-block; margin-right: 16px; }}
.swatch {{ width: 12px; height: 12px; vertical-align: -1px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>SWE-Pro Trace Replay Workload Shape</h1>
<p>Whole trajectories are preserved. LLM blocks are teal; tool blocks are pink. Hover blocks or points for exact token and duration details.</p>
<section class="grid">
  <div class="metric"><span>Source trajectories</span><b>{len(source_plans)}</b></div>
  <div class="metric"><span>Selected trajectories</span><b>{len(selected_plans)}</b></div>
  <div class="metric"><span>Median total output tokens</span><b>{median(token_values) if token_values else 0:.0f}</b></div>
  <div class="metric"><span>Median e2e duration</span><b>{median(duration_values) if duration_values else 0:.1f}s</b></div>
</section>
<section class="panel">
  <h2>Selected Trajectory Timeline</h2>
  <p class="legend"><span><i class="swatch" style="background:#2f8f9d"></i> LLM</span><span><i class="swatch" style="background:#d970a8"></i> Tool</span></p>
  {_timeline_svg(selected_plans)}
</section>
<section class="panel">
  <h2>Duration vs Output Tokens</h2>
  {_scatter_svg(selected_metrics)}
</section>
<section class="panel">
  <h2>Selected Histograms</h2>
  <svg viewBox="0 0 760 330" xmlns="http://www.w3.org/2000/svg">
    <text x="0" y="14">total generated tokens</text>
    <g transform="translate(0,24)">{_bar_svg(token_values, width=220, height=120, color="#2f8f9d")}</g>
    <text x="260" y="14">total duration seconds</text>
    <g transform="translate(260,24)">{_bar_svg(duration_values, width=220, height=120, color="#d970a8")}</g>
    <text x="520" y="14">max turn output tokens</text>
    <g transform="translate(520,24)">{_bar_svg(max_turn_values, width=220, height=120, color="#8a8f2d")}</g>
  </svg>
</section>
<section class="panel">
  <h2>Summary</h2>
  <table>{stats_table}</table>
</section>
<section class="panel">
  <h2>Largest Selected Trajectories</h2>
  <pre>{html.escape(json.dumps(summary.get("top_selected", []), indent=2))}</pre>
</section>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def build_summary(
    source_plans: list[TraceReplayPlan],
    clean_plans: list[TraceReplayPlan],
    selected_plans: list[TraceReplayPlan],
    rejected: dict[str, int],
    args: argparse.Namespace,
) -> dict:
    selected_metrics = [_plan_metrics(plan) for plan in selected_plans]
    source_metrics = [_plan_metrics(plan) for plan in source_plans]
    metric_keys = [
        "turns",
        "tool_turns",
        "sum_generated_tokens",
        "max_generated_tokens",
        "max_prompt_tokens",
        "total_duration_s",
        "total_model_s",
        "total_tool_s",
        "max_tool_s",
    ]
    return {
        "source_files": [str(path) for path in args.source],
        "source_trajectories": len(source_plans),
        "clean_trajectories": len(clean_plans),
        "selected_trajectories": len(selected_plans),
        "target_trajectories": args.target_trajectories,
        "seed": args.seed,
        "filters": {
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "max_generated_tokens_per_turn": args.max_generated_tokens_per_turn,
            "max_total_generated_tokens": args.max_total_generated_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_total_duration_s": args.max_total_duration_s,
            "max_model_duration_s": args.max_model_duration_s,
            "max_tool_duration_s": args.max_tool_duration_s,
        },
        "reject_reason_counts": rejected,
        "source_quantiles": {
            key: _quantiles([float(item[key]) for item in source_metrics]) for key in metric_keys
        },
        "selected_quantiles": {
            key: _quantiles([float(item[key]) for item in selected_metrics]) for key in metric_keys
        },
        "top_selected": sorted(selected_metrics, key=lambda item: float(item["total_duration_s"]), reverse=True)[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--target-trajectories", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--min-turns", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-generated-tokens-per-turn", type=int)
    parser.add_argument("--max-total-generated-tokens", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-total-duration-s", type=float)
    parser.add_argument("--max-model-duration-s", type=float)
    parser.add_argument("--max-tool-duration-s", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_plans = _load_plans(args.source)
    clean_plans: list[TraceReplayPlan] = []
    rejected: dict[str, int] = {}
    for plan in source_plans:
        keep, reason = _passes_filters(plan, args)
        if keep:
            clean_plans.append(plan)
        elif reason:
            rejected[reason] = rejected.get(reason, 0) + 1

    selected_plans = list(clean_plans)
    if args.target_trajectories and len(selected_plans) > args.target_trajectories:
        rng = random.Random(args.seed)
        selected_plans = rng.sample(selected_plans, args.target_trajectories)

    if args.output_jsonl:
        _write_replay_jsonl(args.output_jsonl, selected_plans)

    summary = build_summary(source_plans, clean_plans, selected_plans, rejected, args)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_html(args.output_html, source_plans, selected_plans, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
