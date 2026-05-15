#!/usr/bin/env python3
"""Tiny NCCL probe for rail/PXN experiments.

This launches one trainer/source rank and one or more engine/receiver ranks
using ``kubectl exec``. Unlike ``ibv_rc_pingpong``, this goes through NCCL, so
NCCL variables such as ``NCCL_CROSS_NIC`` and ``NCCL_P2P_PXN_LEVEL`` are
meaningful.

The default shape mimics Slime's weight update group: one trainer rank on a
single CUDA device broadcasts a bucket to both TP ranks in one inference engine.
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import textwrap
import time


RANK_PROGRAM = r"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["PROBE_RANK"])
world_size = int(os.environ.get("PROBE_WORLD_SIZE", "2"))
device = int(os.environ.get("PROBE_CUDA_DEVICE", "0"))
numel = int(os.environ.get("PROBE_NUMEL", str(32 * 1024 * 1024)))
iters = int(os.environ.get("PROBE_ITERS", "1"))
mode = os.environ.get("PROBE_MODE", "broadcast")

torch.cuda.set_device(device)
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['PROBE_MASTER_ADDR']}:{os.environ['PROBE_MASTER_PORT']}",
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=int(os.environ.get("PROBE_TIMEOUT", "180"))),
)

tensor = torch.empty(numel, dtype=torch.float16, device=f"cuda:{device}")
if rank == 0:
    tensor.fill_(1.0)
else:
    tensor.fill_(0.0)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    if mode == "broadcast":
        dist.broadcast(tensor, src=0)
    elif mode == "sendrecv":
        if rank == 0:
            ops = [dist.P2POp(dist.isend, tensor, dst) for dst in range(1, world_size)]
        else:
            ops = [dist.P2POp(dist.irecv, tensor, 0)]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    else:
        raise ValueError(f"unknown PROBE_MODE={mode!r}")
torch.cuda.synchronize()
elapsed = time.time() - t0

sample = float(tensor[0].detach().cpu())
print(
    f"NCCL_PROBE rank={rank} device={device} mode={mode} bytes={tensor.numel() * tensor.element_size()} "
    f"iters={iters} elapsed={elapsed:.3f}s sample={sample}",
    flush=True,
)
dist.destroy_process_group()
"""


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


def pod_host(args: argparse.Namespace, pod: str) -> str:
    if args.master_host:
        return args.master_host
    if args.use_pod_ip:
        return run_text(
            kubectl_base(args) + ["get", "pod", pod, "-o", "jsonpath={.status.podIP}"],
            timeout=30,
        ).strip()
    return run_text(kubectl_base(args) + ["exec", pod, "--", "hostname", "-i"], timeout=30).split()[0]


def free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def rank_cmd(
    args: argparse.Namespace,
    pod: str,
    rank: int,
    world_size: int,
    device: int,
    master_host: str,
    master_port: int,
) -> list[str]:
    env = {
        "PROBE_RANK": str(rank),
        "PROBE_WORLD_SIZE": str(world_size),
        "PROBE_CUDA_DEVICE": str(device),
        "PROBE_MASTER_ADDR": master_host,
        "PROBE_MASTER_PORT": str(master_port),
        "PROBE_NUMEL": str(args.numel),
        "PROBE_ITERS": str(args.iters),
        "PROBE_MODE": args.mode,
        "PROBE_TIMEOUT": str(args.timeout_seconds),
        "NCCL_DEBUG": args.nccl_debug,
        "NCCL_DEBUG_SUBSYS": args.nccl_debug_subsys,
        "NCCL_SOCKET_IFNAME": args.nccl_socket_ifname,
        "GLOO_SOCKET_IFNAME": args.gloo_socket_ifname,
        "NCCL_IB_GID_INDEX": str(args.gid_index),
        "NCCL_CUMEM_ENABLE": str(args.cumem_enable),
        "NCCL_CUMEM_HOST_ENABLE": str(args.cumem_host_enable),
        "NCCL_NVLS_ENABLE": str(args.nvls_enable),
        "NCCL_P2P_PXN_LEVEL": str(args.p2p_pxn_level),
        "NCCL_PXN_DISABLE": str(args.pxn_disable),
        "NCCL_IB_MERGE_NICS": str(args.ib_merge_nics),
        "NCCL_MNNVL_ENABLE": str(args.mnnvl_enable),
        "NVIDIA_GDRCOPY": str(args.nvidia_gdrcopy),
    }
    if args.graph_mixing_support != "":
        env["NCCL_GRAPH_MIXING_SUPPORT"] = str(args.graph_mixing_support)
    if args.cross_nic != "":
        env["NCCL_CROSS_NIC"] = str(args.cross_nic)
    if args.hca:
        env["NCCL_IB_HCA"] = args.hca
    inner = f"{shell_env(env)} python3 - <<'PY'\n{RANK_PROGRAM}\nPY"
    return kubectl_base(args) + ["exec", pod, "--", "bash", "-lc", inner]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02")
    parser.add_argument("--namespace", "-n", default="warnold-dynamo")
    parser.add_argument("--source-pod", "--server-pod", dest="source_pod", default="warnold-swepro-trainer")
    parser.add_argument("--engine-pod", "--client-pod", dest="engine_pod", default="warnold-swepro-engine-0")
    parser.add_argument("--source-device", "--server-device", dest="source_device", type=int, default=0)
    parser.add_argument("--engine-device", "--client-device", dest="engine_device", type=int, default=0)
    parser.add_argument(
        "--engine-devices",
        "--client-devices",
        dest="engine_devices",
        default="",
        help="Comma-separated CUDA devices for engine ranks. Defaults to --engine-device.",
    )
    parser.add_argument(
        "--mode",
        choices=("broadcast", "sendrecv"),
        default="broadcast",
        help="NCCL operation to run. broadcast matches SGLang weight update.",
    )
    parser.add_argument("--master-host", default="")
    parser.add_argument("--use-pod-ip", action="store_true")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--hca", default="", help="NCCL_IB_HCA list to expose to both ranks")
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--cross-nic", default="0", help="Set NCCL_CROSS_NIC; pass empty string to leave unset.")
    parser.add_argument("--p2p-pxn-level", type=int, default=1)
    parser.add_argument("--pxn-disable", type=int, default=0)
    parser.add_argument("--ib-merge-nics", type=int, default=0)
    parser.add_argument("--mnnvl-enable", type=int, default=0)
    parser.add_argument("--cumem-enable", type=int, default=1)
    parser.add_argument("--cumem-host-enable", type=int, default=1)
    parser.add_argument("--nvls-enable", type=int, default=1)
    parser.add_argument("--nvidia-gdrcopy", type=int, default=1)
    parser.add_argument(
        "--graph-mixing-support",
        default="",
        help="Set NCCL_GRAPH_MIXING_SUPPORT; pass empty string to leave unset.",
    )
    parser.add_argument("--nccl-socket-ifname", default="eth0")
    parser.add_argument("--gloo-socket-ifname", default="eth0")
    parser.add_argument("--nccl-debug", default="INFO")
    parser.add_argument("--nccl-debug-subsys", default="INIT,ENV,NET")
    parser.add_argument("--numel", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    master_host = pod_host(args, args.source_pod)
    master_port = args.master_port or free_local_port()
    engine_devices = [
        int(device)
        for device in (args.engine_devices.split(",") if args.engine_devices else [str(args.engine_device)])
        if device != ""
    ]
    world_size = 1 + len(engine_devices)
    print(
        textwrap.dedent(
            f"""\
            NCCL topology probe:
              mode={args.mode}
              source={args.source_pod} cuda:{args.source_device}
              engine={args.engine_pod} cuda:{','.join(str(device) for device in engine_devices)}
              world_size={world_size}
              master={master_host}:{master_port}
              hca={args.hca or '<all visible>'}
              NCCL_CROSS_NIC={args.cross_nic}
              NCCL_P2P_PXN_LEVEL={args.p2p_pxn_level}
              NCCL_PXN_DISABLE={args.pxn_disable}
            """
        ).strip(),
        flush=True,
    )

    source = subprocess.Popen(
        rank_cmd(args, args.source_pod, 0, world_size, args.source_device, master_host, master_port),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(args.startup_delay)
    engines = [
        subprocess.Popen(
            rank_cmd(args, args.engine_pod, rank, world_size, device, master_host, master_port),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank, device in enumerate(engine_devices, start=1)
    ]
    engine_results: list[tuple[int, str]] = []
    deadline = time.time() + args.timeout_seconds + 30
    for engine in engines:
        remaining = max(1.0, deadline - time.time())
        try:
            stdout, _ = engine.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            engine.kill()
            stdout, _ = engine.communicate(timeout=10)
        engine_results.append((engine.returncode, stdout or ""))
    try:
        source_out, _ = source.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        source.kill()
        source_out, _ = source.communicate(timeout=10)

    for index, (returncode, stdout) in enumerate(engine_results, start=1):
        print(f"--- engine rank {index} rc={returncode} ---")
        print(stdout.rstrip() or "<no output>")
    print("--- source rank 0 ---")
    print((source_out or "").rstrip() or "<no output>")
    engine_ok = all(returncode == 0 for returncode, _ in engine_results)
    print(
        f"RESULT {'PASS' if engine_ok and source.returncode == 0 else 'FAIL'} "
        f"engine_rcs={[returncode for returncode, _ in engine_results]} source_rc={source.returncode}"
    )
    return 0 if engine_ok and source.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
