#!/usr/bin/env python3
"""Isolated Dynamo/SGLang online weight-transfer smoke test.

Run this from a GPU pod that can reach the Dynamo frontend. It creates only the
trainer rank of the custom weight-update NCCL group, asks the engine to join as
the remaining ranks, and then broadcasts synthetic tensors through the same
engine routes used by slime.

This intentionally mutates engine weights. Use a disposable engine pod and
recreate it before using it for rollouts.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta

import requests
import torch
import torch.distributed as dist

from slime.backends.megatron_utils.sglang import FlattenedTensorBucket
from slime.utils.distributed_utils import init_process_group


ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NET",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_SOCKET_IFNAME",
    "NCCL_SHM_DISABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_MNNVL_ENABLE",
    "MC_FORCE_MNNVL",
    "NCCL_CUMEM_ENABLE",
    "NCCL_CUMEM_HOST_ENABLE",
    "NCCL_GRAPH_MIXING_SUPPORT",
    "NCCL_STORE_TIMEOUT",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX",
    "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE",
    "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL",
    "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE",
    "SLIME_WEIGHT_UPDATE_NCCL_DEBUG",
    "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS",
]


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.")
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    if normalized == "uint8":
        return torch.uint8
    raise ValueError(f"Unsupported dtype {name!r}")


def dtype_to_wire_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def local_ip() -> str:
    if os.environ.get("POD_IP"):
        return os.environ["POD_IP"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


def weight_update_nccl_env() -> dict[str, str]:
    env: dict[str, str] = {}

    passthroughs = {
        "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE": "NCCL_IB_DISABLE",
        "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA": "NCCL_IB_HCA",
        "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX": "NCCL_IB_GID_INDEX",
        "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE": "NCCL_MNNVL_ENABLE",
        "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL": "MC_FORCE_MNNVL",
        "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE": "NCCL_NVLS_ENABLE",
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG": "NCCL_DEBUG",
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS": "NCCL_DEBUG_SUBSYS",
    }
    for source, target in passthroughs.items():
        value = os.getenv(source)
        if value:
            env[target] = value
    return env


@contextmanager
def temporary_env(overrides: Mapping[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def print_env(keys: Iterable[str]) -> None:
    print("effective process env:", flush=True)
    for key in keys:
        value = os.environ.get(key)
        if value:
            print(f"  {key}={value}", flush=True)
    overrides = weight_update_nccl_env()
    print("weight-update NCCL overrides applied by this script:", flush=True)
    for key in sorted(overrides):
        print(f"  {key}={overrides[key]}", flush=True)


def post_json(base_url: str, path: str, body: dict, timeout: float) -> dict:
    url = base_url.rstrip("/") + path
    print(f"POST {path} timeout={timeout}s", flush=True)
    response = requests.post(url, json=body, timeout=timeout)
    print(f"RESP {path} status={response.status_code} body={response.text[:500]}", flush=True)
    response.raise_for_status()
    return response.json()


def tokenizer_manager_body(method: str, payload: dict) -> dict:
    io_name = {
        "init_weights_update_group": "io_struct.InitWeightsUpdateGroupReqInput",
        "destroy_weights_update_group": "io_struct.DestroyWeightsUpdateGroupReqInput",
    }[method]
    return {"method": method, "args": [{io_name: payload}]}


def make_tensors(args: argparse.Namespace) -> list[tuple[str, torch.Tensor]]:
    dtype = dtype_from_name(args.dtype)
    if args.preset == "embed":
        specs = [("model.embed_tokens.weight", (154880, 2048))]
    elif args.preset == "small":
        specs = [("model.layers.24.input_layernorm.weight", (2048,))]
    elif args.preset == "custom":
        if not args.name or not args.shape:
            raise ValueError("--preset custom requires --name and --shape")
        specs = [(args.name, parse_shape(args.shape))]
    else:
        raise ValueError(f"Unknown preset {args.preset!r}")

    tensors = []
    for index, (name, shape) in enumerate(specs):
        if args.zeros:
            tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        else:
            torch.manual_seed(args.seed + index)
            tensor = torch.randn(shape, dtype=dtype, device="cuda")
        tensors.append((name, tensor))
    return tensors


def run_once(args: argparse.Namespace, iteration: int) -> None:
    group_name = args.group_name or f"swepro-smoke-{int(time.time())}-{iteration}"
    master_address = args.master_address or local_ip()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]

    print(
        f"engine={args.engine_url} master={master_address}:{master_port} "
        f"group={group_name} world_size={args.world_size} load_format={args.load_format}",
        flush=True,
    )

    init_payload = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": args.world_size,
        "group_name": group_name,
        "backend": "nccl",
    }
    destroy_payload = {"group_name": group_name}

    with ThreadPoolExecutor(max_workers=2) as executor:
        init_future = executor.submit(
            post_json,
            args.engine_url,
            "/engine/call_tokenizer_manager",
            tokenizer_manager_body("init_weights_update_group", init_payload),
            args.http_timeout,
        )
        with temporary_env(weight_update_nccl_env()):
            group = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=args.world_size,
                rank=0,
                group_name=group_name,
                timeout=timedelta(seconds=args.pg_timeout),
            )
        print("trainer process group initialized", flush=True)
        print("init result", init_future.result(timeout=args.http_timeout + 10), flush=True)

        try:
            named_tensors = make_tensors(args)
            names = [name for name, _ in named_tensors]
            dtypes = [dtype_to_wire_name(tensor.dtype) for _, tensor in named_tensors]
            shapes = [list(tensor.shape) for _, tensor in named_tensors]
            total_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)

            update_body = {
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": group_name,
                "flush_cache": args.flush_cache,
                "weight_version": f"{args.weight_version_prefix}-{iteration}",
            }
            if args.load_format != "default":
                update_body["load_format"] = args.load_format

            update_future = executor.submit(
                post_json,
                args.engine_url,
                "/engine/update_weights_from_distributed",
                update_body,
                args.http_timeout,
            )
            time.sleep(args.engine_head_start_seconds)

            with temporary_env(weight_update_nccl_env()):
                start = time.time()
                if args.load_format == "flattened_bucket":
                    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
                    flattened = bucket.get_flattened_tensor()
                    print(
                        "trainer broadcast start "
                        f"format=flattened_bucket tensors={len(named_tensors)} "
                        f"bytes={format_bytes(flattened.numel() * flattened.element_size())} "
                        f"dtype={flattened.dtype} shape={tuple(flattened.shape)}",
                        flush=True,
                    )
                    dist.broadcast(flattened, src=0, group=group)
                else:
                    print(
                        "trainer broadcast start "
                        f"format=default tensors={len(named_tensors)} bytes={format_bytes(total_bytes)}",
                        flush=True,
                    )
                    handles = [
                        dist.broadcast(tensor, src=0, group=group, async_op=True)
                        for _, tensor in named_tensors
                    ]
                    for handle in handles:
                        handle.wait()
                torch.cuda.synchronize()
                elapsed = time.time() - start
            print(f"trainer broadcast done elapsed={elapsed:.3f}s", flush=True)
            print("update result", update_future.result(timeout=args.http_timeout + 10), flush=True)
        finally:
            destroy_future = executor.submit(
                post_json,
                args.engine_url,
                "/engine/call_tokenizer_manager",
                tokenizer_manager_body("destroy_weights_update_group", destroy_payload),
                args.http_timeout,
            )
            dist.destroy_process_group(group)
            print("destroy result", destroy_future.result(timeout=args.http_timeout + 10), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default=os.environ.get("ENGINE_URL", "http://warnold-swepro-frontend:3000"))
    parser.add_argument("--group-name", default=os.environ.get("GROUP_NAME"))
    parser.add_argument("--master-address", default=os.environ.get("MASTER_ADDRESS"))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "3")))
    parser.add_argument("--preset", choices=["small", "embed", "custom"], default=os.environ.get("PRESET", "small"))
    parser.add_argument("--name", default=os.environ.get("WEIGHT_NAME"))
    parser.add_argument("--shape", default=os.environ.get("WEIGHT_SHAPE"))
    parser.add_argument("--dtype", default=os.environ.get("WEIGHT_DTYPE", "bfloat16"))
    parser.add_argument(
        "--load-format",
        choices=["default", "flattened_bucket"],
        default=os.environ.get("LOAD_FORMAT", "flattened_bucket"),
    )
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--zeros", action="store_true")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--http-timeout", type=float, default=float(os.environ.get("HTTP_TIMEOUT", "180")))
    parser.add_argument("--pg-timeout", type=float, default=float(os.environ.get("PG_TIMEOUT", "180")))
    parser.add_argument(
        "--engine-head-start-seconds",
        type=float,
        default=float(os.environ.get("ENGINE_HEAD_START_SECONDS", "1")),
    )
    parser.add_argument("--weight-version-prefix", default=os.environ.get("WEIGHT_VERSION_PREFIX", "smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_env(ENV_KEYS)
    for iteration in range(args.repeat):
        run_once(args, iteration)


if __name__ == "__main__":
    main()
