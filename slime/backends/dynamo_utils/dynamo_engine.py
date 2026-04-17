"""Dynamo-backed rollout engine for slime.

DynamoEngine is a drop-in replacement for SGLangEngine that launches a
``dynamo.sglang`` worker with ``--enable-rl``.  The worker exposes standard
SGLang HTTP endpoints **plus** the ``/engine/call_tokenizer_manager`` route
(via Dynamo's RLMixin) for RL-specific operations like weight updates,
pause/continue generation, etc.

Architecture
------------
* Each DynamoEngine actor manages **one** ``dynamo.sglang`` worker process.
* Workers self-register with Dynamo's discovery layer, so the Dynamo frontend
  (router) finds them automatically — no explicit ``/add_worker`` calls needed.
* Generation requests flow through the Dynamo frontend (KV-aware routing).
* RL control-plane calls (weight sync, pause, profile, …) go directly to the
  worker's ``/engine/call_tokenizer_manager`` endpoint.
"""

import ipaddress
import logging
import os
import subprocess
import sys
import time

import requests
from sglang.srt.utils import kill_process_tree

from slime.backends.sglang_utils.sglang_engine import get_base_gpu_id, _to_local_gpu_id
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


def _format_ipv6(addr):
    """Wrap bare IPv6 addresses in brackets for use in URLs."""
    if not addr or addr.startswith("["):
        return addr
    try:
        if ipaddress.ip_address(addr).version == 6:
            return f"[{addr}]"
    except ValueError:
        pass
    return addr


class DynamoEngine(RayActor):
    """Ray actor that wraps a single ``dynamo.sglang`` worker.

    Mirrors the public interface of :class:`SGLangEngine` so that it can be
    used as a drop-in replacement inside :class:`ServerGroup`.
    """

    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self.process = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        host = _format_ipv6(host or get_host_info()[1])
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_ipv6(ip_part)}:{port_part}"

        # KV router requires etcd discovery; propagate from CLI args.
        router_mode = getattr(self.args, "dynamo_router_mode", None) or "round-robin"
        self._discovery_backend = "etcd" if router_mode == "kv" else "file"

        self.server_host = host
        self.server_port = port

        gpus_per_engine = self.num_gpus_per_engine or self.args.rollout_num_gpus_per_engine
        nnodes = max(1, gpus_per_engine // self.args.num_gpus_per_node)
        self.node_rank = self.rank % nnodes
        self.tp_size = gpus_per_engine // getattr(self.args, "sglang_pp_size", 1)

        base = self.base_gpu_id if self.base_gpu_id is not None else get_base_gpu_id(self.args, self.rank)
        base = _to_local_gpu_id(base)
        self._base_gpu_id = base

        # Store for building the worker command
        self._dist_init_addr = dist_init_addr
        self._nccl_port = nccl_port
        self._host = host
        self._disaggregation_bootstrap_port = disaggregation_bootstrap_port

        if self.node_rank != 0:
            return

        self._launch_worker()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _launch_worker(self):
        args = self.args
        gpus_per_engine = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine

        cmd = [
            sys.executable, "-m", "dynamo.sglang",
            "--model-path", args.hf_checkpoint,
            "--tp", str(self.tp_size),
            "--port", str(self.server_port),
            "--host", self._host.strip("[]"),
            "--trust-remote-code",
            "--enable-rl",
        ]

        if getattr(args, "sglang_mem_fraction_static", None) is not None:
            cmd.extend(["--mem-fraction-static", str(args.sglang_mem_fraction_static)])

        stream_interval = getattr(args, "rollout_stream_interval", 1)
        if stream_interval != 1:
            cmd.extend(["--stream-interval", str(stream_interval)])

        if getattr(args, "sglang_dp_size", 1) > 1:
            cmd.extend(["--dp-size", str(args.sglang_dp_size)])

        pp_size = getattr(args, "sglang_pp_size", 1)
        if pp_size > 1:
            cmd.extend(["--pp-size", str(pp_size)])

        if getattr(args, "fp16", False):
            cmd.extend(["--dtype", "float16"])

        if getattr(args, "offload_rollout", False):
            cmd.append("--enable-memory-saver")

        if self.worker_type in ("prefill", "decode"):
            cmd.extend(["--disaggregation-mode", self.worker_type])
            if self._disaggregation_bootstrap_port:
                cmd.extend(["--disaggregation-bootstrap-port", str(self._disaggregation_bootstrap_port)])

        # Publish KV cache events when the router is configured to consume
        # them. Each worker binds a unique ZMQ port; the router subscribes
        # via NATS. When --no-dynamo-router-kv-events is set, skip the
        # publisher — the router uses approximate / predict-on-route mode.
        publish_kv_events = (
            self._discovery_backend == "etcd"
            and getattr(args, "dynamo_router_kv_events", True)
        )
        if publish_kv_events:
            import json as _json
            kv_events_cfg = getattr(args, "sglang_kv_events_config", None)
            if not kv_events_cfg:
                zmq_port = self.server_port + 10000  # unique per engine
                kv_events_cfg = _json.dumps({
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "endpoint": f"tcp://*:{zmq_port}",
                    "enable_kv_cache_events": True,
                })
            cmd.extend(["--kv-events-config", kv_events_cfg])

        # Per-group sglang overrides (skip keys already handled above)
        _skip_override_keys = {"model_path", "tp", "port", "host", "enable_rl"}
        for key, value in self.sglang_overrides.items():
            if key in _skip_override_keys:
                continue
            flag = f"--{key.replace('_', '-')}"
            cmd.extend([flag, str(value)])

        env = os.environ.copy()
        # KV router requires etcd discovery; propagate from frontend config.
        discovery = getattr(self, "_discovery_backend", None) or "file"
        env["DYN_DISCOVERY_BACKEND"] = discovery
        # The Dynamo runtime's system status server exposes /health and
        # /engine/* routes (including call_tokenizer_manager).  It only
        # starts when DYN_SYSTEM_PORT >= 0.
        env["DYN_SYSTEM_PORT"] = str(self.server_port)
        # Block hashes in KV events must be deterministic across workers so
        # the router can match predicted inserts with engine-published
        # events.  Without a fixed hash seed Python's randomized hash leaks
        # into SGLang's block-hash derivation and the router logs
        # "block_hash mismatch: sequence hashes should be uniform across
        # workers", preventing any prefix cache hits.
        env.setdefault("PYTHONHASHSEED", "0")
        # Force dynamo.sglang's decode_handler to call SGLang's engine with
        # stream=False. Slime always aggregates full responses, so we get no
        # benefit from the per-chunk scheduler->detokenizer push path — and
        # pay a ~4-5k tok/s aggregate decode-throughput penalty for it. See
        # dynamo commit 724b1a994e1 for the backend-side toggle.
        env.setdefault("DYN_SGL_FORCE_NONSTREAM", "1")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(self._base_gpu_id + i) for i in range(gpus_per_engine)
        )

        logger.info("Launching Dynamo worker (rank=%d): %s", self.rank, " ".join(cmd))
        self.process = subprocess.Popen(cmd, env=env)
        self._wait_healthy(timeout=300)
        self.flush_cache()

    def _wait_healthy(self, timeout=300):
        url = f"http://{self.server_host}:{self.server_port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Dynamo worker healthy at %s:%s", self.server_host, self.server_port)
                    return
            except requests.RequestException:
                pass
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"Dynamo worker exited with code {self.process.returncode} (rank={self.rank})"
                )
            time.sleep(2)
        raise TimeoutError(
            f"Dynamo worker at {self.server_host}:{self.server_port} not healthy after {timeout}s"
        )

    def shutdown(self):
        if self.process is None:
            return
        logger.info("Shutting down Dynamo engine %s:%s", self.server_host, self.server_port)
        kill_process_tree(self.process.pid)

    def simulate_crash(self):
        if self.process:
            logger.info("Simulating crash on Dynamo engine %s:%s", self.server_host, self.server_port)
            self.shutdown()

    # ------------------------------------------------------------------
    # Engine route helpers
    # ------------------------------------------------------------------

    def _call_engine_route(self, route, body=None):
        """Call a dedicated /engine/{route} endpoint on the system status server."""
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/engine/{route}"
        response = requests.post(url, json=body or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def _call_tm(self, method, args=None, kwargs=None):
        """Call a tokenizer_manager method via the RLMixin endpoint."""
        if self.node_rank != 0:
            return
        body = {"method": method}
        if args is not None:
            body["args"] = args
        if kwargs is not None:
            body["kwargs"] = kwargs
        return self._call_engine_route("call_tokenizer_manager", body)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_generate(self, timeout: float = 5.0) -> bool:
        if self.node_rank != 0:
            return True
        resp = requests.get(
            f"http://{self.server_host}:{self.server_port}/health",
            timeout=timeout,
        )
        resp.raise_for_status()
        return True

    def get_url(self) -> str | None:
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    # ------------------------------------------------------------------
    # Weight management (via dedicated /engine/* routes)
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        body = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            body["weight_version"] = weight_version
        return self._call_engine_route("update_weights_from_tensor", body)

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
    ):
        body = {
            "names": names,
            "dtypes": [str(d).replace("torch.", "") for d in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            body["weight_version"] = weight_version
        return self._call_engine_route("update_weights_from_distributed", body)

    def update_weights_from_disk(self, model_path, load_format=None):
        body = {"model_path": model_path}
        if load_format:
            body["load_format"] = load_format
        return self._call_engine_route("update_weights_from_disk", body)

    def init_weights_update_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        return self._call_tm(
            "init_weights_update_group",
            args=[{
                "io_struct.InitWeightsUpdateGroupReqInput": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": backend,
                },
            }],
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._call_tm(
                "destroy_weights_update_group",
                args=[{
                    "io_struct.DestroyWeightsUpdateGroupReqInput": {
                        "group_name": group_name,
                    },
                }],
            )
        except requests.exceptions.RequestException:
            pass

    def get_weight_version(self):
        # No get_weight_version on tokenizer_manager or engine routes.
        return None

    # ------------------------------------------------------------------
    # Cache & generation control
    # ------------------------------------------------------------------

    def flush_cache(self):
        if self.node_rank != 0:
            return
        self._call_tm("flush_cache")

    def pause_generation(self):
        if self.node_rank != 0:
            return
        self._call_tm(
            "pause_generation",
            args=[{"io_struct.PauseGenerationReqInput": {}}],
        )

    def continue_generation(self):
        if self.node_rank != 0:
            return
        self._call_tm(
            "continue_generation",
            args=[{"io_struct.ContinueGenerationReqInput": {}}],
        )

    def release_memory_occupation(self):
        # Dynamo's release_memory_occupation engine route unregisters from
        # discovery and pauses generation, which breaks the frontend's model
        # routing.  For now, just flush the cache without unregistering.
        self.flush_cache()

    def resume_memory_occupation(self, tags: list[str] | None = None):
        # Matching no-op for release_memory_occupation above.
        pass

    # ------------------------------------------------------------------
    # Profiling & misc
    # ------------------------------------------------------------------

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        return self._call_tm(
            "post_process_weights",
            args=[{
                "io_struct.PostProcessWeightsReqInput": {
                    "restore_weights_before_load": restore_weights_before_load,
                    "post_process_quantization": post_process_quantization,
                },
            }],
        )

    def check_weights(self, action: str):
        return self._call_tm(
            "check_weights",
            args=[{"io_struct.CheckWeightsReqInput": {"action": action}}],
        )

    def start_profile(
        self,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        return self._call_engine_route("start_profile", {
            "output_dir": output_dir,
            "start_step": start_step,
            "num_steps": num_steps,
            "activities": activities,
            "profile_by_stage": profile_by_stage,
            "with_stack": with_stack,
            "record_shapes": record_shapes,
        })

    def stop_profile(self):
        return self._call_engine_route("stop_profile")
