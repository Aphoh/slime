#!/usr/bin/env python3
"""Probe whether specific RoCE HCAs can form an RC verbs connection.

This intentionally runs outside the training stack.  It starts
``ibv_rc_pingpong`` in a server pod and then connects from a client pod, with a
separate ``--ib-dev`` value on each side.  That makes asymmetric cases such as
``server mlx5_0`` <-> ``client mlx5_2`` testable, which is the kind of pairing
NCCL selected during the failed mixed trainer/engine weight update.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def kubectl_base(args: argparse.Namespace) -> list[str]:
    cmd = ["kubectl"]
    if args.context:
        cmd.extend(["--context", args.context])
    if args.namespace:
        cmd.extend(["-n", args.namespace])
    return cmd


def run_text(cmd: list[str], timeout: float | None = None) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def pod_ip(args: argparse.Namespace, pod: str) -> str:
    return run_text(
        kubectl_base(args) + ["get", "pod", pod, "-o", "jsonpath={.status.podIP}"],
        timeout=30,
    ).strip()


def parse_pair(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("pairs must look like server_hca:client_hca")
    server_hca, client_hca = raw.split(":", 1)
    if not server_hca or not client_hca:
        raise argparse.ArgumentTypeError("both server and client HCAs are required")
    return server_hca, client_hca


def exec_cmd(args: argparse.Namespace, pod: str, inner: list[str]) -> list[str]:
    return kubectl_base(args) + ["exec", pod, "--", *inner]


def run_pair(
    args: argparse.Namespace,
    server_host: str,
    index: int,
    server_hca: str,
    client_hca: str,
) -> bool:
    port = args.base_port + index
    common = [
        "-g",
        str(args.gid_index),
        "-p",
        str(port),
        "-s",
        str(args.message_size),
        "-n",
        str(args.iters),
        "-m",
        str(args.mtu),
    ]
    server_inner = [
        "timeout",
        f"{args.timeout_seconds}s",
        "ibv_rc_pingpong",
        "-d",
        server_hca,
        *common,
    ]
    client_inner = [
        "timeout",
        f"{args.timeout_seconds}s",
        "ibv_rc_pingpong",
        "-d",
        client_hca,
        *common,
        server_host,
    ]

    print(f"\n=== server {server_hca} <-> client {client_hca} port={port} ===", flush=True)
    server = subprocess.Popen(
        exec_cmd(args, args.server_pod, server_inner),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(args.server_start_delay)

    client = subprocess.run(
        exec_cmd(args, args.client_pod, client_inner),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout_seconds + 15,
    )
    try:
        server_out, _ = server.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server_out, _ = server.communicate(timeout=5)

    print("--- client ---")
    print(client.stdout.rstrip() or "<no output>")
    print("--- server ---")
    print((server_out or "").rstrip() or "<no output>")

    ok = client.returncode == 0 and server.returncode == 0
    print(f"RESULT {'PASS' if ok else 'FAIL'} client_rc={client.returncode} server_rc={server.returncode}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02")
    parser.add_argument("--namespace", "-n", default="warnold-dynamo")
    parser.add_argument("--server-pod", default="warnold-swepro-engine-0")
    parser.add_argument("--client-pod", default="warnold-swepro-trainer")
    parser.add_argument("--server-host", default="", help="TCP bootstrap host; defaults to server pod IP")
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=18515)
    parser.add_argument("--message-size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mtu", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--server-start-delay", type=float, default=1.0)
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        default=[],
        help="HCA pair as server_hca:client_hca. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = args.pair or [
        ("mlx5_0", "mlx5_0"),
        ("mlx5_1", "mlx5_1"),
        ("mlx5_0", "mlx5_2"),
        ("mlx5_2", "mlx5_0"),
    ]
    server_host = args.server_host or pod_ip(args, args.server_pod)
    print(
        f"server={args.server_pod} host={server_host} client={args.client_pod} "
        f"gid_index={args.gid_index}",
        flush=True,
    )

    failures = 0
    for index, (server_hca, client_hca) in enumerate(pairs):
        if not run_pair(args, server_host, index, server_hca, client_hca):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
