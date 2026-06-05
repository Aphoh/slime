#!/usr/bin/env python3
import argparse
import html
import json
import math
import re
from collections import defaultdict

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


GAUGE_METRICS = [
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_retracted_reqs",
    "sglang:token_usage",
    "sglang:num_used_tokens",
    "sglang:mamba_usage",
    "sglang:swa_token_usage",
    "sglang:pending_prealloc_token_usage",
    "sglang:cache_hit_rate",
    "sglang:gen_throughput",
    "dynamo_frontend_inflight_requests",
    "dynamo_frontend_queued_requests",
    "dynamo_frontend_router_queue_pending_requests",
    "dynamo_frontend_router_queue_pending_isl_tokens",
    "dynamo_request_plane_inflight_requests",
    "dynamo_request_plane_queue_seconds",
    "dynamo_frontend_worker_active_prefill_tokens",
    "dynamo_frontend_worker_active_decode_blocks",
]

HIST_METRICS = [
    "dynamo_frontend_time_to_first_token_seconds",
    "dynamo_frontend_request_duration_seconds",
]


def is_finite(value):
    return value is not None and math.isfinite(float(value))


def label_from_metric(metric_name, key):
    match = re.search(rf'{re.escape(key)}="([^"]*)"', metric_name or "")
    return match.group(1) if match else None


def parse_le(metric_name):
    le = label_from_metric(metric_name, "le")
    if le is None:
        return None
    if le == "+Inf":
        return math.inf
    try:
        return float(le)
    except ValueError:
        return None


def quantile_from_delta(delta_by_le, quantile):
    if not delta_by_le:
        return None
    total = delta_by_le.get(math.inf)
    if total is None:
        total = max(delta_by_le.values())
    if total <= 0:
        return None
    target = total * quantile
    for le in sorted(delta_by_le):
        if le is math.inf:
            continue
        if delta_by_le[le] >= target:
            return le
    finite_buckets = [le for le in delta_by_le if math.isfinite(le)]
    return max(finite_buckets) if finite_buckets else None


def load_plot_data(parquet_path, run_id):
    if pq is None:
        raise RuntimeError("pyarrow is required when reading parquet")
    cols = [
        "scraper_endpoint",
        "metric_name",
        "metric_name_clean",
        "metric_value",
        "time_since_start",
        "node",
    ]
    table = pq.read_table(parquet_path, columns=cols)
    rows = {column: table[column].to_pylist() for column in cols}

    gauges = defaultdict(dict)
    hist_bucket_values = defaultdict(lambda: defaultdict(dict))

    for i, clean_name in enumerate(rows["metric_name_clean"]):
        value = rows["metric_value"][i]
        if not is_finite(value):
            continue
        endpoint = rows["scraper_endpoint"][i] or rows["node"][i] or "unknown"
        time_sec = float(rows["time_since_start"][i] or 0.0)

        if clean_name in GAUGE_METRICS:
            bucket = round(time_sec / 5) * 5
            key = (clean_name, endpoint)
            previous = gauges[key].get(bucket)
            value = float(value)
            gauges[key][bucket] = value if previous is None else max(previous, value)

        if clean_name in HIST_METRICS:
            le = parse_le(rows["metric_name"][i])
            if le is None:
                continue
            bucket = round(time_sec / 10) * 10
            hist_bucket_values[(clean_name, endpoint)][bucket][le] = float(value)

    series = []
    for (metric, endpoint), points in sorted(gauges.items()):
        series.append(
            {
                "metric": metric,
                "scraper_endpoint": endpoint,
                "points": [[float(t), float(v)] for t, v in sorted(points.items())],
            }
        )

    for (metric, endpoint), by_time in sorted(hist_bucket_values.items()):
        previous = None
        quantile_lines = {"p50": [], "p90": [], "p99": [], "count": []}
        for time_sec in sorted(by_time):
            current = by_time[time_sec]
            if previous is None:
                previous = current
                continue
            all_buckets = set(current) | set(previous)
            delta = {
                le: max(0.0, current.get(le, 0.0) - previous.get(le, 0.0))
                for le in all_buckets
            }
            total = delta.get(math.inf, max(delta.values()) if delta else 0.0)
            quantile_lines["count"].append([float(time_sec), float(total)])
            for name, quantile in [("p50", 0.5), ("p90", 0.9), ("p99", 0.99)]:
                value = quantile_from_delta(delta, quantile)
                if value is not None and math.isfinite(value):
                    quantile_lines[name].append([float(time_sec), float(value)])
            previous = current
        for name, points in quantile_lines.items():
            series.append(
                {
                    "metric": f"{metric}:{name}",
                    "scraper_endpoint": endpoint,
                    "points": points,
                }
            )

    stats = []
    for item in series:
        values = [p[1] for p in item["points"] if is_finite(p[1])]
        if not values:
            continue
        stats.append(
            {
                "metric": item["metric"],
                "scraper_endpoint": item["scraper_endpoint"],
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "last": values[-1],
                "count": len(values),
            }
        )

    return {
        "run_id": run_id,
        "source": parquet_path,
        "rows": table.num_rows,
        "gauge_metrics": GAUGE_METRICS,
        "hist_metrics": HIST_METRICS,
        "series": series,
        "stats": stats,
    }


def grouped_series(data):
    grouped = defaultdict(list)
    for item in data["series"]:
        grouped[item["metric"]].append(item)
    return grouped


def stat_lookup(data):
    lookup = defaultdict(list)
    for item in data["stats"]:
        lookup[item["metric"]].append(item)
    return lookup


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def table(headers, rows):
    body = ["<table><thead><tr>"]
    body.extend(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body.append("</tr></thead><tbody>")
    for row in rows:
        body.append("<tr>")
        body.extend(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        body.append("</tr>")
    body.append("</tbody></table>")
    return "".join(body)


def svg_chart(metric, rows, width=1120, height=260):
    points = [
        (x, y)
        for row in rows
        for x, y in row["points"]
        if is_finite(x) and is_finite(y)
    ]
    if not points:
        return '<p class="muted">No Tachometer samples.</p>'

    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    if ymin == ymax:
        ymax = ymin + 1.0

    pad_left, pad_right, pad_top, pad_bottom = 58, 16, 16, 34

    def sx(x):
        return pad_left + (x - xmin) / (xmax - xmin or 1.0) * (
            width - pad_left - pad_right
        )

    def sy(y):
        return pad_top + (ymax - y) / (ymax - ymin) * (
            height - pad_top - pad_bottom
        )

    colors = [
        "#0f766e",
        "#2563eb",
        "#be123c",
        "#a16207",
        "#7c3aed",
        "#0891b2",
        "#db2777",
        "#475569",
    ]

    grid = []
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = pad_top + frac * (height - pad_top - pad_bottom)
        value = ymax - frac * (ymax - ymin)
        grid.append(
            f'<line x1="{pad_left}" x2="{width-pad_right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        grid.append(
            f'<text x="4" y="{y+4:.1f}" font-size="10" fill="#64748b">{html.escape(fmt(value))}</text>'
        )

    lines = []
    legend = []
    for idx, row in enumerate(rows):
        color = colors[idx % len(colors)]
        pts = " ".join(
            f"{sx(x):.1f},{sy(y):.1f}"
            for x, y in row["points"]
            if is_finite(x) and is_finite(y)
        )
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(row["scraper_endpoint"])}</span>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(grid)
        + f'<line x1="{pad_left}" x2="{width-pad_right}" y1="{height-pad_bottom}" y2="{height-pad_bottom}" stroke="#94a3b8"/>'
        + "".join(lines)
        + "</svg>"
        + '<div class="legend">'
        + "".join(legend)
        + "</div>"
    )


def render_html(data):
    grouped = grouped_series(data)
    stats = stat_lookup(data)

    charts = [
        ("Per-engine running requests", ["sglang:num_running_reqs"]),
        ("Per-engine queued requests", ["sglang:num_queue_reqs"]),
        ("Per-engine retractions", ["sglang:num_retracted_reqs"]),
        ("KV/token usage", ["sglang:token_usage", "sglang:num_used_tokens"]),
        ("Mamba/SWA usage", ["sglang:mamba_usage", "sglang:swa_token_usage"]),
        ("Pending prealloc token usage", ["sglang:pending_prealloc_token_usage"]),
        ("SGLang cache hit rate", ["sglang:cache_hit_rate"]),
        ("SGLang generation throughput", ["sglang:gen_throughput"]),
        (
            "Frontend request pressure",
            [
                "dynamo_frontend_inflight_requests",
                "dynamo_frontend_queued_requests",
                "dynamo_frontend_router_queue_pending_requests",
                "dynamo_request_plane_inflight_requests",
            ],
        ),
        (
            "Frontend token pressure",
            [
                "dynamo_frontend_router_queue_pending_isl_tokens",
                "dynamo_frontend_worker_active_prefill_tokens",
                "dynamo_frontend_worker_active_decode_blocks",
            ],
        ),
        (
            "Frontend TTFT histogram windows",
            [
                "dynamo_frontend_time_to_first_token_seconds:p50",
                "dynamo_frontend_time_to_first_token_seconds:p90",
                "dynamo_frontend_time_to_first_token_seconds:p99",
                "dynamo_frontend_time_to_first_token_seconds:count",
            ],
        ),
        (
            "Frontend request-duration histogram windows",
            [
                "dynamo_frontend_request_duration_seconds:p50",
                "dynamo_frontend_request_duration_seconds:p90",
                "dynamo_frontend_request_duration_seconds:p99",
                "dynamo_frontend_request_duration_seconds:count",
            ],
        ),
    ]

    summary_rows = []
    for metric in [
        "sglang:num_running_reqs",
        "sglang:num_queue_reqs",
        "sglang:token_usage",
        "sglang:mamba_usage",
        "sglang:cache_hit_rate",
        "sglang:gen_throughput",
        "dynamo_frontend_time_to_first_token_seconds:p90",
        "dynamo_frontend_request_duration_seconds:p90",
    ]:
        for item in stats.get(metric, []):
            summary_rows.append(
                [
                    metric,
                    item["scraper_endpoint"],
                    fmt(item["max"]),
                    fmt(item["avg"]),
                    fmt(item["last"]),
                    item["count"],
                ]
            )

    html_parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Tachometer-only Dynamo overview</title>",
        """
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}
h1{font-size:24px;margin-bottom:4px} h2{font-size:18px;margin:0 0 10px} h3{font-size:14px;margin:18px 0 8px}
.muted{color:#64748b}.card{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:14px 0;box-shadow:0 1px 2px #0001}
code{background:#e2e8f0;padding:1px 4px;border-radius:4px}.chart{width:100%;height:260px;background:#fff}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:#475569;margin-top:4px}.legend i{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}th{background:#f1f5f9}
</style>
""",
        f"<h1>Tachometer-only Dynamo overview</h1><p class='muted'><code>{html.escape(data['run_id'])}</code><br>{data['rows']:,} parquet rows from <code>{html.escape(data['source'])}</code></p>",
        "<div class='card'><h2>Summary Stats</h2>"
        + table(["metric", "endpoint", "max", "avg", "last", "points"], summary_rows)
        + "</div>",
    ]

    for title, metric_names in charts:
        html_parts.append(f"<div class='card'><h2>{html.escape(title)}</h2>")
        for metric in metric_names:
            if metric in grouped:
                html_parts.append(f"<h3><code>{html.escape(metric)}</code></h3>")
                html_parts.append(svg_chart(metric, grouped[metric]))
        html_parts.append("</div>")

    return "".join(html_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--html-out", required=True)
    parser.add_argument("--json-in")
    args = parser.parse_args()

    if args.json_in:
        with open(args.json_in) as f:
            data = json.load(f)
    else:
        data = load_plot_data(args.parquet, args.run_id)
        with open(args.json_out, "w") as f:
            json.dump(data, f)

    with open(args.html_out, "w") as f:
        f.write(render_html(data))

    print(args.json_out)
    print(args.html_out)


if __name__ == "__main__":
    main()
