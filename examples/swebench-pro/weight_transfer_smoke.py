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
    "NCCL_CROSS_NIC",
    "NCCL_IB_MERGE_NICS",
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
    "NCCL_DEBUG_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_HCA",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_GID_INDEX",
    "SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC",
    "SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS",
    "SLIME_WEIGHT_UPDATE_NCCL_P2P_PXN_LEVEL",
    "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_ENABLE",
    "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_HOST_ENABLE",
    "SLIME_WEIGHT_UPDATE_NCCL_GRAPH_MIXING_SUPPORT",
    "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE",
    "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL",
    "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE",
    "SLIME_WEIGHT_UPDATE_NCCL_NET",
    "SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME",
    "SLIME_WEIGHT_UPDATE_NCCL_DEBUG",
    "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS",
    "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_FILE",
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
        "SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC": "NCCL_CROSS_NIC",
        "SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS": "NCCL_IB_MERGE_NICS",
        "SLIME_WEIGHT_UPDATE_NCCL_P2P_PXN_LEVEL": "NCCL_P2P_PXN_LEVEL",
        "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_ENABLE": "NCCL_CUMEM_ENABLE",
        "SLIME_WEIGHT_UPDATE_NCCL_CUMEM_HOST_ENABLE": "NCCL_CUMEM_HOST_ENABLE",
        "SLIME_WEIGHT_UPDATE_NCCL_GRAPH_MIXING_SUPPORT": "NCCL_GRAPH_MIXING_SUPPORT",
        "SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE": "NCCL_MNNVL_ENABLE",
        "SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL": "MC_FORCE_MNNVL",
        "SLIME_WEIGHT_UPDATE_NCCL_NVLS_ENABLE": "NCCL_NVLS_ENABLE",
        "SLIME_WEIGHT_UPDATE_NCCL_NET": "NCCL_NET",
        "SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME": "NCCL_SOCKET_IFNAME",
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG": "NCCL_DEBUG",
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS": "NCCL_DEBUG_SUBSYS",
        "SLIME_WEIGHT_UPDATE_NCCL_DEBUG_FILE": "NCCL_DEBUG_FILE",
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


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in parse_csv(value)]


def engine_urls(args: argparse.Namespace) -> list[str]:
    urls = parse_csv(args.engine_urls or args.engine_url)
    if not urls:
        raise ValueError("At least one engine URL is required")
    return urls


def engine_gpu_counts(args: argparse.Namespace, urls: list[str]) -> list[int]:
    if args.engine_gpu_counts:
        counts = parse_int_csv(args.engine_gpu_counts)
        if len(counts) != len(urls):
            raise ValueError(
                f"--engine-gpu-counts has {len(counts)} entries, but --engine-urls has {len(urls)}"
            )
        return counts
    inferred = args.world_size - 1
    if inferred % len(urls) != 0:
        raise ValueError(
            "Cannot infer engine GPU counts: pass --engine-gpu-counts when "
            f"world_size={args.world_size} and engines={len(urls)}"
        )
    return [inferred // len(urls)] * len(urls)


def tokenizer_manager_body(method: str, payload: dict) -> dict:
    io_name = {
        "init_weights_update_group": "io_struct.InitWeightsUpdateGroupReqInput",
        "destroy_weights_update_group": "io_struct.DestroyWeightsUpdateGroupReqInput",
        "post_process_weights": "io_struct.PostProcessWeightsReqInput",
    }[method]
    return {"method": method, "args": [{io_name: payload}]}


def make_tensors(args: argparse.Namespace) -> list[tuple[str, torch.Tensor]]:
    dtype = dtype_from_name(args.dtype)
    if args.preset == "embed":
        specs = [("model.embed_tokens.weight", (154880, 2048))]
    elif args.preset == "qwen35-embed":
        specs = [("model.language_model.embed_tokens.weight", (248320, 3072))]
    elif args.preset == "qwen35-small":
        specs = [("model.language_model.layers.24.input_layernorm.weight", (3072,))]
    elif args.preset == "qwen35-480m":
        # Five 96 MiB q_proj tensors approximate one of the large flattened
        # buckets in the 122B Qwen3.5 run.
        specs = [
            (f"model.language_model.layers.{layer}.self_attn.q_proj.weight", (16384, 3072))
            for layer in (3, 7, 11, 15, 19)
        ]
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
    urls = engine_urls(args)
    gpu_counts = engine_gpu_counts(args, urls)

    print(
        f"engines={urls} gpu_counts={gpu_counts} master={master_address}:{master_port} "
        f"group={group_name} world_size={args.world_size} load_format={args.load_format}",
        flush=True,
    )

    destroy_payload = {"group_name": group_name}

    pending_update_results = []
    with ThreadPoolExecutor(max_workers=max(args.engine_request_workers, len(urls))) as executor:
        rank_offset = 1
        init_futures = []
        for url, gpu_count in zip(urls, gpu_counts, strict=True):
            init_payload = {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": args.world_size,
                "group_name": group_name,
                "backend": "nccl",
            }
            init_futures.append(
                executor.submit(
                    post_json,
                    url,
                    "/engine/call_tokenizer_manager",
                    tokenizer_manager_body("init_weights_update_group", init_payload),
                    args.http_timeout,
                )
            )
            rank_offset += gpu_count
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
        for init_future in init_futures:
            print("init result", init_future.result(timeout=args.http_timeout + 10), flush=True)

        try:
            if args.restore_before_load:
                print(
                    "restore-before-load result",
                post_json(
                    urls[0],
                    "/engine/call_tokenizer_manager",
                    tokenizer_manager_body(
                        "post_process_weights",
                            {
                                "restore_weights_before_load": True,
                                "post_process_quantization": False,
                            },
                    ),
                    args.http_timeout,
                ),
                    flush=True,
                )

            named_tensors = make_tensors(args)
            update_result: dict | None = None
            for update_index in range(args.updates_per_group):
                update_result = _broadcast_update(
                    args,
                    executor,
                    group_name,
                    group,
                    named_tensors,
                    f"{args.weight_version_prefix}-{iteration}-u{update_index}",
                )
                if args.defer_update_results:
                    pending_update_results.append(update_result)
            for update_future in pending_update_results:
                print("deferred update result", update_future.result(timeout=args.http_timeout + 10), flush=True)
            if (
                args.post_process_after_load
                and not args.defer_update_results
                and update_result is not None
                and update_result.get("success")
            ):
                print(
                    "post-process-after-load result",
                    post_json(
                        urls[0],
                        "/engine/call_tokenizer_manager",
                        tokenizer_manager_body(
                            "post_process_weights",
                            {
                                "restore_weights_before_load": False,
                                "post_process_quantization": True,
                            },
                        ),
                        args.http_timeout,
                    ),
                    flush=True,
                )
        finally:
            destroy_futures = [
                executor.submit(
                    post_json,
                    url,
                    "/engine/call_tokenizer_manager",
                    tokenizer_manager_body("destroy_weights_update_group", destroy_payload),
                    args.http_timeout,
                )
                for url in urls
            ]
            try:
                # Keep the trainer rank alive until the engine has torn down its
                # side of the custom group; dropping rank 0 first can surface as
                # an RDMA retry error during otherwise successful smoke runs.
                for destroy_future in destroy_futures:
                    print("destroy result", destroy_future.result(timeout=args.http_timeout + 10), flush=True)
            finally:
                dist.destroy_process_group(group)


def _make_group_name(args: argparse.Namespace, iteration: int, group_index: int | None = None) -> str:
    base = args.group_name or f"swepro-smoke-{int(time.time())}-{iteration}"
    if group_index is None:
        return base
    return f"{base}-g{group_index}"


def _init_update_group(
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    group_name: str,
    master_address: str,
) -> dist.ProcessGroup:
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
    return group


def _broadcast_update(
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    group_name: str,
    group: dist.ProcessGroup,
    named_tensors: list[tuple[str, torch.Tensor]],
    weight_version: str,
) -> dict:
    urls = engine_urls(args)
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
        "weight_version": weight_version,
    }
    if args.load_format != "default":
        update_body["load_format"] = args.load_format

    update_futures = [
        executor.submit(
            post_json,
            url,
            "/engine/update_weights_from_distributed",
            update_body,
            args.http_timeout,
        )
        for url in urls
    ]
    time.sleep(args.engine_head_start_seconds)

    with temporary_env(weight_update_nccl_env()):
        start = time.time()
        if args.load_format == "flattened_bucket":
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened = bucket.get_flattened_tensor()
            print(
                "trainer broadcast start "
                f"group={group_name} format=flattened_bucket tensors={len(named_tensors)} "
                f"bytes={format_bytes(flattened.numel() * flattened.element_size())} "
                f"dtype={flattened.dtype} shape={tuple(flattened.shape)}",
                flush=True,
            )
            dist.broadcast(flattened, src=0, group=group)
        else:
            print(
                "trainer broadcast start "
                f"group={group_name} format=default tensors={len(named_tensors)} bytes={format_bytes(total_bytes)}",
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
    print(f"trainer broadcast done group={group_name} elapsed={elapsed:.3f}s", flush=True)
    if args.defer_update_results:
        print("update result deferred", flush=True)
        return update_futures
    update_results = [future.result(timeout=args.http_timeout + 10) for future in update_futures]
    for update_result in update_results:
        print("update result", update_result, flush=True)
    return update_results[-1] if update_results else {}


def _destroy_update_group(
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    group_name: str,
    group: dist.ProcessGroup,
) -> None:
    destroy_payload = {"group_name": group_name}
    destroy_future = executor.submit(
        post_json,
        args.engine_url,
        "/engine/call_tokenizer_manager",
        tokenizer_manager_body("destroy_weights_update_group", destroy_payload),
        args.http_timeout,
    )
    try:
        # Keep the trainer rank alive until the engine has torn down its side
        # of the custom group; dropping rank 0 first can surface as an RDMA
        # retry error during otherwise successful smoke runs.
        print("destroy result", destroy_future.result(timeout=args.http_timeout + 10), flush=True)
    finally:
        dist.destroy_process_group(group)


def run_held_groups(args: argparse.Namespace, iteration: int) -> None:
    master_address = args.master_address or local_ip()
    named_tensors = make_tensors(args)
    groups: list[tuple[str, dist.ProcessGroup]] = []

    with ThreadPoolExecutor(max_workers=max(args.engine_request_workers, args.held_groups)) as executor:
        pending_update_results = []
        try:
            for group_index in range(args.held_groups):
                group_name = _make_group_name(args, iteration, group_index)
                group = _init_update_group(args, executor, group_name, master_address)
                groups.append((group_name, group))

            if args.restore_before_load:
                print(
                    "restore-before-load result",
                    post_json(
                        args.engine_url,
                        "/engine/call_tokenizer_manager",
                        tokenizer_manager_body(
                            "post_process_weights",
                            {
                                "restore_weights_before_load": True,
                                "post_process_quantization": False,
                            },
                        ),
                        args.http_timeout,
                    ),
                    flush=True,
                )

            for group_index, (group_name, group) in enumerate(groups):
                for update_index in range(args.updates_per_group):
                    update_result = _broadcast_update(
                        args,
                        executor,
                        group_name,
                        group,
                        named_tensors,
                        f"{args.weight_version_prefix}-{iteration}-g{group_index}-u{update_index}",
                    )
                    if args.defer_update_results:
                        pending_update_results.append(update_result)
            for update_future in pending_update_results:
                print("deferred update result", update_future.result(timeout=args.http_timeout + 10), flush=True)

            if args.post_process_after_load:
                print(
                    "post-process-after-load result",
                    post_json(
                        args.engine_url,
                        "/engine/call_tokenizer_manager",
                        tokenizer_manager_body(
                            "post_process_weights",
                            {
                                "restore_weights_before_load": False,
                                "post_process_quantization": True,
                            },
                        ),
                        args.http_timeout,
                    ),
                    flush=True,
                )
        finally:
            for group_name, group in reversed(groups):
                _destroy_update_group(args, executor, group_name, group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default=os.environ.get("ENGINE_URL", "http://warnold-swepro-frontend:3000"))
    parser.add_argument(
        "--engine-urls",
        default=os.environ.get("ENGINE_URLS", ""),
        help="Comma-separated direct engine URLs. When set, overrides --engine-url.",
    )
    parser.add_argument(
        "--engine-gpu-counts",
        default=os.environ.get("ENGINE_GPU_COUNTS", ""),
        help="Comma-separated engine TP sizes/rank counts used to compute rank offsets.",
    )
    parser.add_argument("--group-name", default=os.environ.get("GROUP_NAME"))
    parser.add_argument("--master-address", default=os.environ.get("MASTER_ADDRESS"))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "3")))
    parser.add_argument(
        "--preset",
        choices=["small", "embed", "qwen35-small", "qwen35-embed", "qwen35-480m", "custom"],
        default=os.environ.get("PRESET", "small"),
    )
    parser.add_argument("--name", default=os.environ.get("WEIGHT_NAME"))
    parser.add_argument("--shape", default=os.environ.get("WEIGHT_SHAPE"))
    parser.add_argument("--dtype", default=os.environ.get("WEIGHT_DTYPE", "bfloat16"))
    parser.add_argument(
        "--load-format",
        choices=["default", "flattened_bucket"],
        default=os.environ.get("LOAD_FORMAT", "flattened_bucket"),
    )
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "1")))
    parser.add_argument(
        "--held-groups",
        type=int,
        default=int(os.environ.get("HELD_GROUPS", "1")),
        help="Create this many update groups and keep them all alive while broadcasting through each.",
    )
    parser.add_argument(
        "--updates-per-group",
        type=int,
        default=int(os.environ.get("UPDATES_PER_GROUP", "1")),
        help="Broadcast this many consecutive updates through each initialized group.",
    )
    parser.add_argument(
        "--engine-request-workers",
        type=int,
        default=int(os.environ.get("ENGINE_REQUEST_WORKERS", "2")),
        help="Number of local threads used to issue engine HTTP update requests.",
    )
    parser.add_argument(
        "--defer-update-results",
        action="store_true",
        help="Do not wait for each engine update response until all broadcasts have been issued.",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--zeros", action="store_true")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--restore-before-load", action="store_true")
    parser.add_argument("--post-process-after-load", action="store_true")
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
        if args.held_groups > 1:
            run_held_groups(args, iteration)
        else:
            run_once(args, iteration)


if __name__ == "__main__":
    main()
