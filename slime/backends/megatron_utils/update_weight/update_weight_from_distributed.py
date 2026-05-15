from __future__ import annotations

import logging
import os
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_to_hf
from ..sglang import DeltaSpec, FlattenedTensorBucket
from .common import all_gather_param, named_params_and_buffers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedWeightUpdateGroup:
    group_name: str
    group: dist.ProcessGroup
    rollout_engines: tuple[ActorHandle, ...]
    engine_gpu_counts: tuple[int, ...]


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def _format_tensor_manifest(named_tensors: Sequence[tuple[str, torch.Tensor]], *, limit: int | None = None) -> str:
    lines = []
    for index, (name, tensor) in enumerate(named_tensors):
        if limit is not None and index >= limit:
            lines.append(f"... {len(named_tensors) - limit} more tensors")
            break
        shape = tuple(int(dim) for dim in tensor.shape)
        bytes_ = tensor.numel() * tensor.element_size()
        lines.append(
            f"{index:04d} name={name} shape={shape} dtype={tensor.dtype} "
            f"device={tensor.device} bytes={_format_bytes(bytes_)}"
        )
    return "\n".join(lines)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid float env %s=%r", name, value)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid int env %s=%r", name, value)
        return default


def _log_slow_tensor_group(
    event: str,
    *,
    group_name: str,
    elapsed: float,
    tensor_group: Sequence[tuple[str, torch.Tensor]],
    total_bytes: int,
) -> None:
    threshold = _env_float("SLIME_WEIGHT_UPDATE_SLOW_MANIFEST_SECONDS", 0.0)
    if threshold <= 0 or elapsed < threshold:
        return

    manifest_limit = _env_int("SLIME_WEIGHT_UPDATE_SLOW_MANIFEST_LIMIT", 24)
    logger.warning(
        "[WEIGHT UPDATE group=%s] slow %s elapsed=%.3fs threshold=%.3fs tensors=%d bytes=%s manifest:\n%s",
        group_name,
        event,
        elapsed,
        threshold,
        len(tensor_group),
        _format_bytes(total_bytes),
        _format_tensor_manifest(tensor_group, limit=manifest_limit),
    )


def _raise_failed_engine_updates(
    responses: Sequence[object],
    *,
    group_name: str,
    tensor_group: Sequence[tuple[str, torch.Tensor]],
) -> None:
    failures = []
    for index, response in enumerate(responses):
        if isinstance(response, dict) and response.get("success") is False:
            failures.append(f"engine_index={index} response={response}")

    if not failures:
        return

    manifest = _format_tensor_manifest(tensor_group)
    raise RuntimeError(
        f"Weight update failed for group={group_name} with {len(failures)} engine failure(s):\n"
        + "\n".join(failures)
        + "\nTensor manifest:\n"
        + manifest
    )


def should_post_process_rollout_weights(quantization_config: dict[str, object] | None) -> bool:
    if os.getenv("SLIME_WEIGHT_UPDATE_POST_PROCESS_WEIGHTS", "0") == "1":
        return True
    return bool(quantization_config and quantization_config.get("quant_method") in ["compressed-tensors"])


@contextmanager
def _temporary_nccl_env(overrides: Mapping[str, str]):
    if not overrides:
        yield
        return

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


def _weight_update_nccl_env() -> dict[str, str]:
    env = {}

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


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    Subclasses override ``_send_weights`` / ``_on_chunk`` to inject per-mode behaviour.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self._expert_model_update_groups = None
        self.update_weight_metrics: dict[str, float] = {}
        self._rank_log_prefix = ""
        self._use_direct_expert_update = os.getenv("SLIME_WEIGHT_UPDATE_DIRECT_EXPERTS", "0") == "1"

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Drained by the actor onto the rollout/step log.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        expert_tp_rank = getattr(mpu, "get_expert_tensor_parallel_rank", lambda: 0)()
        self._is_direct_expert_src_rank = (
            self._use_direct_expert_update
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and expert_tp_rank == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"
        if self._is_direct_expert_src_rank:
            self._expert_group_name = f"slime-pp_{pp_rank}-expert-rank_{dist.get_rank()}"
        self._rank_log_prefix = (
            f"[WEIGHT UPDATE rank={dist.get_rank()} pp={pp_rank} "
            f"tp={mpu.get_tensor_model_parallel_rank()} "
            f"cp={mpu.get_context_parallel_rank()}]"
        )

        if self._is_pp_src_rank:
            lock_t0 = time.time()
            while not ray.get(self.rollout_engine_lock.acquire.remote()):
                time.sleep(0.1)
            lock_elapsed = time.time() - lock_t0
            logger.info(
                "%s connect lock acquired group=%s wait=%.3fs engines=%d engine_gpu_counts=%s",
                self._rank_log_prefix,
                self._group_name,
                lock_elapsed,
                len(rollout_engines),
                engine_gpu_counts,
            )
            connect_t0 = time.time()
            try:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    rollout_engines,
                    engine_gpu_counts=engine_gpu_counts,
                )
            except Exception:
                logger.exception("%s connect failed group=%s", self._rank_log_prefix, self._group_name)
                raise
            finally:
                ray.get(self.rollout_engine_lock.release.remote())
            logger.info(
                "%s connect done group=%s lock_wait=%.3fs connect=%.3fs",
                self._rank_log_prefix,
                self._group_name,
                lock_elapsed,
                time.time() - connect_t0,
            )

        if self._is_direct_expert_src_rank:
            lock_t0 = time.time()
            while not ray.get(self.rollout_engine_lock.acquire.remote()):
                time.sleep(0.1)
            lock_elapsed = time.time() - lock_t0
            logger.info(
                "%s expert direct connect lock acquired group=%s wait=%.3fs engines=%d engine_gpu_counts=%s",
                self._rank_log_prefix,
                self._expert_group_name,
                lock_elapsed,
                len(rollout_engines),
                engine_gpu_counts,
            )
            connect_t0 = time.time()
            try:
                if self._expert_model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args,
                        self._expert_group_name,
                        self._expert_model_update_groups,
                        self.rollout_engines,
                    )
                self._expert_model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._expert_group_name,
                    rollout_engines,
                    engine_gpu_counts=engine_gpu_counts,
                )
            except Exception:
                logger.exception("%s expert direct connect failed group=%s", self._rank_log_prefix, self._expert_group_name)
                raise
            finally:
                ray.get(self.rollout_engine_lock.release.remote())
            logger.info(
                "%s expert direct connect done group=%s lock_wait=%.3fs connect=%.3fs",
                self._rank_log_prefix,
                self._expert_group_name,
                lock_elapsed,
                time.time() - connect_t0,
            )

    def disconnect_rollout_engines(self) -> None:
        if getattr(self, "_is_pp_src_rank", False) and self._model_update_groups is not None:
            disconnect_rollout_engines_from_distributed(
                self.args, self._group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = None
        if getattr(self, "_is_direct_expert_src_rank", False) and self._expert_model_update_groups is not None:
            disconnect_rollout_engines_from_distributed(
                self.args, self._expert_group_name, self._expert_model_update_groups, self.rollout_engines
            )
            self._expert_model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → _send_weights → continue. Progress on PP source.
        """
        import time as _time

        self.weight_version += 1

        if dist.get_rank() == 0:
            _t0 = _time.time()
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            _pause_elapsed = _time.time() - _t0

            _t0 = _time.time()
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            _flush_elapsed = _time.time() - _t0

            logger.info(
                f"[WEIGHT UPDATE] pause_generation={_pause_elapsed:.3f}s | "
                f"flush_cache={_flush_elapsed:.3f}s | "
                f"num_engines={len(self.rollout_engines)}"
            )

            # Restore inference-side packed/quantized layouts before loading
            # trainer-format tensors.
            if should_post_process_rollout_weights(self.quantization_config):
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        _sync_t0 = _time.time()
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        self._send_weights(pbar)

        _sync_elapsed = _time.time() - _sync_t0
        if dist.get_rank() == 0:
            # Re-apply inference-side weight layout transforms after loading.
            if should_post_process_rollout_weights(self.quantization_config):
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            _t0 = _time.time()
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            _continue_elapsed = _time.time() - _t0

            logger.info(
                f"[WEIGHT UPDATE] sync={_sync_elapsed:.3f}s | "
                f"continue_generation={_continue_elapsed:.3f}s | "
                f"total={_sync_elapsed + _continue_elapsed:.3f}s"
            )
        dist.barrier(group=get_gloo_group())

    def _send_weights(self, pbar: tqdm | None) -> None:
        """
        Non-expert (TP) pass → barrier → expert (EP) pass → barrier. Each iterator
        yields broadcast-ready chunks (bucketing happens internally); subclasses
        override ``_on_chunk`` to inject per-chunk behaviour.
        """
        for hf_chunk in self._iter_non_expert_chunks():
            self._on_chunk(hf_chunk)
            self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
        dist.barrier(group=get_gloo_group())

        if self._use_direct_expert_update:
            buffer_size = 0
            converted_named_tensors: list[tuple[str, torch.Tensor]] = []
            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." not in name:
                    continue
                buffer_size = self._update_direct_expert_weight_from_distributed(
                    name, param, converted_named_tensors, buffer_size, pbar=pbar
                )
            if buffer_size:
                self._update_direct_expert_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
        else:
            for hf_chunk in self._iter_expert_chunks():
                self._on_chunk(hf_chunk)
                self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
        dist.barrier(group=get_gloo_group())

    def _on_chunk(self, hf_chunk: list[tuple[str, torch.Tensor]]) -> None:
        """
        Hook for each HF chunk in ``_send_weights`` before its broadcast. No-op by default.
        """

    def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield broadcast-sized HF chunks of non-expert params: TP all-gather +
        HF convert per param, then bucket up to ``--update-weight-buffer-size``.
        Empty on non-PP-src ranks (they still join all_gather_param).
        """
        buffer_size = 0
        buffer: list[tuple[str, torch.Tensor]] = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            convert_t0 = time.time()
            hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            convert_elapsed = time.time() - convert_t0
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
                logger.info(
                    "%s flushing non-expert bucket before %s bytes=%s tensors=%d next_param=%s",
                    self._rank_log_prefix,
                    name,
                    _format_bytes(buffer_size),
                    len(buffer),
                    _format_bytes(chunk_bytes),
                )
                yield buffer
                buffer = []
                buffer_size = 0
            if convert_elapsed > 5:
                logger.info(
                    "%s converted non-expert param %s in %.3fs size=%s outputs=%d",
                    self._rank_log_prefix,
                    name,
                    convert_elapsed,
                    _format_bytes(chunk_bytes),
                    len(hf_chunk),
                )
            buffer.extend(hf_chunk)
            buffer_size += chunk_bytes
        if buffer:
            logger.info(
                "%s flushing final non-expert bucket bytes=%s tensors=%d",
                self._rank_log_prefix,
                _format_bytes(buffer_size),
                len(buffer),
            )
            yield buffer

    def _update_direct_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Gather expert-TP shards and send locally owned experts directly.
        """
        param = all_gather_param(name, param)
        param_size = param.numel() * param.element_size()
        if buffer_size and buffer_size + param_size > self.args.update_weight_buffer_size:
            logger.info(
                "%s flushing direct expert bucket before %s bytes=%s tensors=%d next_param=%s",
                self._rank_log_prefix,
                name,
                _format_bytes(buffer_size),
                len(converted_named_tensors),
                _format_bytes(param_size),
            )
            self._update_direct_expert_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0

        if self._is_direct_expert_src_rank:
            converted_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            )

        return buffer_size + param_size

    def _update_direct_expert_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
    ) -> None:
        if self._is_direct_expert_src_rank and converted_named_tensors:
            self._send_bucket_weights_from_distributed(
                self._expert_group_name,
                self._expert_model_update_groups,
                converted_named_tensors,
                pbar=pbar,
                sync_tensor_model_parallel=False,
            )

        expert_tp_world = getattr(mpu, "get_expert_tensor_parallel_world_size", lambda: 1)()
        if expert_tp_world > 1:
            dist.barrier(group=mpu.get_expert_tensor_parallel_group())

    def _iter_expert_chunks(
        self,
        params: Iterator[tuple[str, torch.Tensor]] | None = None,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield one HF chunk per EP-weighted batch of expert params: TP gather +
        buffer until threshold, then EP gather + HF convert. ``params`` lets
        callers restrict the iter to a subset (used by delta-sync sub-passes);
        defaults to all expert params on this rank.
        """
        if params is None:
            params = ((n, p) for n, p in named_params_and_buffers(self.args, self.model) if ".experts." in n)
        buffer_size = 0
        batch: list[tuple[str, torch.Tensor]] = []
        for name, param in params:
            param = all_gather_param(name, param)
            param_size = param.numel() * param.element_size()
            if (
                batch
                and buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
                logger.info(
                    "%s flushing expert bucket before %s bytes=%s tensors=%d next_param=%s ep_world=%d",
                    self._rank_log_prefix,
                    name,
                    _format_bytes(buffer_size),
                    len(batch),
                    _format_bytes(param_size),
                    mpu.get_expert_model_parallel_world_size(),
                )
                hf_chunk = self._ep_gather_and_convert(batch)
                if hf_chunk:
                    yield hf_chunk
                batch = []
                buffer_size = 0
            batch.append((name, param))
            buffer_size += param_size
        if batch:
            logger.info(
                "%s flushing final expert bucket bytes=%s tensors=%d",
                self._rank_log_prefix,
                _format_bytes(buffer_size),
                len(batch),
            )
            hf_chunk = self._ep_gather_and_convert(batch)
            if hf_chunk:
                yield hf_chunk

    def _ep_gather_and_convert(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """
        EP all-gather a buffered batch + HF convert on PP source. Returns HF tensors on
        PP source, [] elsewhere. Clears ``named_tensors``.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        gather_t0 = time.time()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()
        gather_elapsed = time.time() - gather_t0
        if gather_elapsed > 5:
            logger.info(
                "%s gathered expert bucket in %.3fs tensors=%d ep_world=%d",
                self._rank_log_prefix,
                gather_elapsed,
                len(named_tensors),
                mpu.get_expert_model_parallel_world_size(),
            )

        if os.getenv("SLIME_WEIGHT_UPDATE_EXPERT_PRE_BROADCAST_BARRIER", "0") == "1":
            barrier_t0 = time.time()
            dist.barrier(group=mpu.get_expert_model_parallel_group())
            barrier_elapsed = time.time() - barrier_t0
            if barrier_elapsed > 1:
                logger.info(
                    "%s expert pre-broadcast barrier took %.3fs tensors=%d ep_world=%d",
                    self._rank_log_prefix,
                    barrier_elapsed,
                    len(named_tensors),
                    mpu.get_expert_model_parallel_world_size(),
                )

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return []

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        convert_t0 = time.time()
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        convert_elapsed = time.time() - convert_t0
        if convert_elapsed > 5:
            logger.info(
                "%s converted expert bucket in %.3fs gathered_tensors=%d hf_tensors=%d",
                self._rank_log_prefix,
                convert_elapsed,
                len(all_gathered_params),
                len(converted_hf_tensors),
            )
        return converted_hf_tensors

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        load_format: str | None = None,
        delta: DeltaSpec | None = None,
        *,
        sync_tensor_model_parallel: bool = True,
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        Delta sync passes ``load_format="delta"`` + a ``DeltaSpec`` describing the
        per-param decoding of the (__positions__, __values__) bucket tensors.
        """
        if self._is_pp_src_rank and converted_named_tensors:
            self._send_bucket_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                converted_named_tensors,
                pbar=pbar,
                sync_tensor_model_parallel=sync_tensor_model_parallel,
                load_format=load_format,
                delta=delta,
            )

        if sync_tensor_model_parallel:
            # Non-source TP ranks must not enter the next TP all-gather while
            # the source rank is still broadcasting this bucket on the engine
            # update process group. Keep collective ordering consistent across
            # groups.
            barrier_t0 = time.time()
            dist.barrier(group=mpu.get_tensor_model_parallel_group())
            barrier_elapsed = time.time() - barrier_t0
            if barrier_elapsed > 5:
                logger.info(
                    "%s tensor-model-parallel bucket barrier took %.3fs",
                    self._rank_log_prefix,
                    barrier_elapsed,
                )

    def _send_bucket_weights_from_distributed(
        self,
        group_name: str,
        model_update_groups: Sequence[DistributedWeightUpdateGroup] | None,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        *,
        sync_tensor_model_parallel: bool,
        load_format: str | None = None,
        delta: DeltaSpec | None = None,
    ) -> None:
        if model_update_groups is None:
            raise RuntimeError(f"Weight-update group {group_name} has not been connected")

        bucket_bytes = sum(param.numel() * param.element_size() for _, param in converted_named_tensors)
        logger.info(
            "%s bucket update start group=%s tensors=%d bytes=%s format=%s sync_tp=%s",
            self._rank_log_prefix,
            group_name,
            len(converted_named_tensors),
            _format_bytes(bucket_bytes),
            load_format or "default",
            sync_tensor_model_parallel,
        )
        # Lock the rollout engines to prevent dead lock on broadcast.
        lock_t0 = time.time()
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        lock_elapsed = time.time() - lock_t0
        logger.info(
            "%s bucket update lock acquired group=%s wait=%.3fs",
            self._rank_log_prefix,
            group_name,
            lock_elapsed,
        )

        update_t0 = time.time()
        try:
            refs = update_weights_from_distributed(
                group_name,
                model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                load_format=load_format,
                delta=delta,
            )

            response_t0 = time.time()
            responses = ray.get(refs)
            response_elapsed = time.time() - response_t0
            logger.info(
                "%s bucket engine responses group=%s refs=%d wait=%.3fs",
                self._rank_log_prefix,
                group_name,
                len(refs),
                response_elapsed,
            )
            _log_slow_tensor_group(
                "engine-response-wait",
                group_name=group_name,
                elapsed=response_elapsed,
                tensor_group=converted_named_tensors,
                total_bytes=bucket_bytes,
            )
            _raise_failed_engine_updates(
                responses,
                group_name=group_name,
                tensor_group=converted_named_tensors,
            )
        except Exception:
            logger.exception(
                "%s bucket update failed group=%s tensors=%d manifest:\n%s",
                self._rank_log_prefix,
                group_name,
                len(converted_named_tensors),
                _format_tensor_manifest(converted_named_tensors),
            )
            raise
        finally:
            ray.get(self.rollout_engine_lock.release.remote())

        update_elapsed = time.time() - update_t0
        converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)
        logger.info(
            "%s bucket update done group=%s lock_wait=%.3fs update=%.3fs bytes=%s",
            self._rank_log_prefix,
            group_name,
            lock_elapsed,
            update_elapsed,
            _format_bytes(bucket_bytes),
        )

def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> list[DistributedWeightUpdateGroup]:
    """
    Create NCCL group(s): training rank 0 + engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    if len(engine_gpu_counts) != len(rollout_engines):
        raise ValueError(
            f"engine_gpu_counts length {len(engine_gpu_counts)} does not match "
            f"rollout_engines length {len(rollout_engines)}"
        )

    per_engine_groups = os.getenv("SLIME_WEIGHT_UPDATE_PER_ENGINE_GROUPS", "1") == "1"
    if per_engine_groups and len(rollout_engines) > 1:
        _destroy_stale_weight_update_group(group_name, rollout_engines)
        logger.info(
            "Creating one weight-update NCCL group per engine for %s: engines=%d gpu_counts=%s",
            group_name,
            len(rollout_engines),
            list(engine_gpu_counts),
        )
        return [
            _connect_weight_update_group(args, f"{group_name}-engine_{i}", [engine], [gpu_count])
            for i, (engine, gpu_count) in enumerate(zip(rollout_engines, engine_gpu_counts, strict=True))
        ]

    return [_connect_weight_update_group(args, group_name, rollout_engines, engine_gpu_counts)]


def _connect_weight_update_group(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int],
) -> DistributedWeightUpdateGroup:
    if not rollout_engines:
        raise ValueError("Cannot create weight-update group without rollout engines")

    _destroy_stale_weight_update_group(group_name, rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    # Compute cumulative rank offsets: engine i starts at cumulative[i] + 1.
    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    with _temporary_nccl_env(_weight_update_nccl_env()):
        refs = [
            engine.init_weights_update_group.remote(
                master_address=master_address,
                master_port=master_port,
                rank_offset=cumulative[i] + 1,
                world_size=world_size,
                group_name=group_name,
                backend="nccl",
            )
            for i, engine in enumerate(rollout_engines)
        ]
        model_update_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        ray.get(refs)
    return DistributedWeightUpdateGroup(
        group_name=group_name,
        group=model_update_groups,
        rollout_engines=tuple(rollout_engines),
        engine_gpu_counts=tuple(engine_gpu_counts),
    )


def _destroy_stale_weight_update_group(group_name: str, rollout_engines: Sequence[ActorHandle]) -> None:
    # A stopped or crashed training job can leave an engine-side update group
    # alive. SGLang/Dynamo rejects reusing the same group name, and the training
    # rank can then block waiting for ranks that never join. Clean up first;
    # destroying a missing group is best-effort and intentionally non-fatal.
    stale_group_refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    try:
        ray.get(stale_group_refs, timeout=30)
    except Exception as exc:
        logger.warning("Best-effort cleanup of stale rollout weight group %s failed: %s", group_name, exc)


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    update_groups = _normalize_update_groups(group_name, model_update_groups, rollout_engines)
    for update_group in update_groups:
        refs = [
            engine.destroy_weights_update_group.remote(update_group.group_name)
            for engine in update_group.rollout_engines
        ]
        try:
            # Keep the trainer rank in the custom group until the engines have
            # torn down their side. Destroying rank 0 first can show up as
            # IBV_WC_RETRY_EXC_ERR on otherwise healthy RDMA paths.
            ray.get(refs)
        finally:
            dist.destroy_process_group(update_group.group)


def _normalize_update_groups(
    group_name: str,
    group: dist.ProcessGroup | Sequence[DistributedWeightUpdateGroup],
    rollout_engines: Sequence[ActorHandle],
) -> list[DistributedWeightUpdateGroup]:
    if isinstance(group, Sequence) and not isinstance(group, dist.ProcessGroup):
        return list(group)

    return [
        DistributedWeightUpdateGroup(
            group_name=group_name,
            group=group,
            rollout_engines=tuple(rollout_engines),
            engine_gpu_counts=(),
        )
    ]


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup | Sequence[DistributedWeightUpdateGroup],
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
    delta: DeltaSpec | None = None,
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    Delta sync passes ``load_format="delta"`` + ``delta`` (DeltaSpec).
    """
    use_flattened_bucket = (
        os.getenv("SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET", "1") == "1"
        and load_format is None
        and delta is None
    )
    if use_flattened_bucket:
        if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            tensor_groups = [converted_named_tensors]
        else:
            grouped_by_dtype = {}
            for name, tensor in converted_named_tensors:
                grouped_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))
            tensor_groups = list(grouped_by_dtype.values())
    else:
        tensor_groups = [converted_named_tensors]

    if os.getenv("SLIME_WEIGHT_UPDATE_SINGLE_TENSOR_BUCKETS", "0") == "1":
        tensor_groups = [[tensor] for tensor_group in tensor_groups for tensor in tensor_group]

    update_groups = _normalize_update_groups(group_name, group, rollout_engines)
    all_refs = []
    for tensor_group in tensor_groups:
        names = [name for name, _ in tensor_group]
        dtypes = [param.dtype for _, param in tensor_group]
        shapes = [param.shape for _, param in tensor_group]
        effective_load_format = "flattened_bucket" if use_flattened_bucket else load_format

        for update_group in update_groups:
            request_kwargs = {
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": update_group.group_name,
                "weight_version": str(weight_version),
                "load_format": effective_load_format,
            }
            if delta is not None:
                request_kwargs["delta"] = delta

            dispatch_t0 = time.time()
            refs = [
                engine.update_weights_from_distributed.remote(**request_kwargs)
                for engine in update_group.rollout_engines
            ]
            dispatch_elapsed = time.time() - dispatch_t0
            all_refs.extend(refs)
            logger.info(
                "[WEIGHT UPDATE group=%s] dispatched engine update requests tensors=%d engines=%d format=%s in %.3fs",
                update_group.group_name,
                len(tensor_group),
                len(update_group.rollout_engines),
                effective_load_format or "default",
                dispatch_elapsed,
            )
            if os.getenv("SLIME_WEIGHT_UPDATE_DEBUG_MANIFEST", "0") == "1":
                logger.info(
                    "[WEIGHT UPDATE group=%s] tensor manifest:\n%s",
                    update_group.group_name,
                    _format_tensor_manifest(tensor_group),
                )

            broadcast_t0 = time.time()
            with _temporary_nccl_env(_weight_update_nccl_env()):
                if use_flattened_bucket:
                    flatten_t0 = time.time()
                    bucket = FlattenedTensorBucket(named_tensors=tensor_group)
                    flattened_tensor = bucket.get_flattened_tensor()
                    flatten_elapsed = time.time() - flatten_t0
                    total_bytes = flattened_tensor.numel() * flattened_tensor.element_size()
                    logger.info(
                        "[WEIGHT UPDATE group=%s] flattened tensors=%d bytes=%s dtype=%s device=%s in %.3fs",
                        update_group.group_name,
                        len(tensor_group),
                        _format_bytes(total_bytes),
                        flattened_tensor.dtype,
                        flattened_tensor.device,
                        flatten_elapsed,
                    )
                    dist.broadcast(flattened_tensor, 0, group=update_group.group)
                    torch.cuda.synchronize(flattened_tensor.device)
                else:
                    handles = []
                    total_bytes = sum(param.numel() * param.element_size() for _, param in tensor_group)
                    logger.info(
                        "[WEIGHT UPDATE group=%s] broadcasting tensors=%d bytes=%s first_dtype=%s first_device=%s first_shape=%s format=%s",
                        update_group.group_name,
                        len(tensor_group),
                        _format_bytes(total_bytes),
                        tensor_group[0][1].dtype if tensor_group else None,
                        tensor_group[0][1].device if tensor_group else None,
                        tuple(tensor_group[0][1].shape) if tensor_group else None,
                        effective_load_format or "default",
                    )
                    for _, param in tensor_group:
                        handles.append(dist.broadcast(param.data, 0, group=update_group.group, async_op=True))
                    for handle in handles:
                        handle.wait()
                    torch.cuda.synchronize()

            broadcast_elapsed = time.time() - broadcast_t0
            logger.info(
                "[WEIGHT UPDATE group=%s] broadcasted tensors=%d bytes=%s format=%s in %.3fs",
                update_group.group_name,
                len(tensor_group),
                _format_bytes(total_bytes),
                effective_load_format or "default",
                broadcast_elapsed,
            )
            _log_slow_tensor_group(
                "broadcast",
                group_name=update_group.group_name,
                elapsed=broadcast_elapsed,
                tensor_group=tensor_group,
                total_bytes=total_bytes,
            )

            if os.getenv("SLIME_WEIGHT_UPDATE_VALIDATE_EACH_GROUP", "0") == "1":
                response_t0 = time.time()
                responses = ray.get(refs)
                response_elapsed = time.time() - response_t0
                logger.info(
                    "[WEIGHT UPDATE group=%s] engine responses returned refs=%d in %.3fs",
                    update_group.group_name,
                    len(refs),
                    response_elapsed,
                )
                _log_slow_tensor_group(
                    "per-group-engine-response-wait",
                    group_name=update_group.group_name,
                    elapsed=response_elapsed,
                    tensor_group=tensor_group,
                    total_bytes=total_bytes,
                )
                _raise_failed_engine_updates(
                    responses,
                    group_name=update_group.group_name,
                    tensor_group=tensor_group,
                )

    return all_refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger rollout-engine weight restore/repack hooks.

    SGLang's hook is used for compressed-tensor quantization and for other
    inference-only packed layouts, such as FlashInfer TRT-LLM BF16 MoE weights.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
