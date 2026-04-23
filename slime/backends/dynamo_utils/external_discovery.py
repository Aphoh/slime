"""Discover Dynamo workers behind an externally-managed frontend.

Slime queries the frontend's /health to enumerate instances, parses host
from each instance's ``transport.tcp`` field, and calls
``/engine/call_tokenizer_manager`` with ``method=get_internal_state`` on
each worker's system-status port (DYN_SYSTEM_PORT) to read topology
(tp_size, pp_size, disaggregation_mode, node_rank, ...).

Multi-node TP workers expose one system-port endpoint per node, but only
node_rank=0 carries the tokenizer_manager and is addressable for
weight-update calls. We filter the rest.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Iterable

import requests

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DiscoveredWorker:
    instance_id: int
    host: str
    system_port: int
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    nnodes: int
    node_rank: int
    disaggregation_mode: str  # "null" | "prefill" | "decode"
    served_model_name: str | None

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.system_port}"

    @property
    def worker_type(self) -> str:
        """Map disaggregation_mode to slime's ServerGroup worker_type."""
        if self.disaggregation_mode in ("prefill", "decode"):
            return self.disaggregation_mode
        return "regular"


def _parse_tcp_host(transport_tcp: str) -> str:
    """Extract host from ``host:port/token/endpoint`` form."""
    host_port = transport_tcp.split("/", 1)[0]
    return host_port.rsplit(":", 1)[0]


def _fetch_health(frontend_url: str, timeout: float = 5.0) -> dict | None:
    try:
        resp = requests.get(f"{frontend_url.rstrip('/')}/health", timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def _fetch_internal_state(host: str, system_port: int, timeout: float = 10.0) -> dict | None:
    """POST get_internal_state via /engine/call_tokenizer_manager."""
    url = f"http://{host}:{system_port}/engine/call_tokenizer_manager"
    try:
        resp = requests.post(
            url,
            json={"method": "get_internal_state"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("get_internal_state failed for %s:%s (%s)", host, system_port, e)
        return None
    body = resp.json()
    result = body.get("result")
    if not isinstance(result, list) or not result:
        logger.warning("get_internal_state returned unexpected shape from %s:%s: %r", host, system_port, body)
        return None
    # result is one entry per DP rank; ServerArgs fields are identical across DPs.
    return result[0]


def _instances_by_id(health: dict) -> dict[int, str]:
    """Dedupe /health instances by instance_id; return {instance_id: tcp_host}.

    Only counts the ``generate`` endpoint — KV-routing frontends self-register
    a ``router-discovery`` endpoint on the frontend's own host that has no
    DYN_SYSTEM_PORT and would fail topology probes.
    """
    by_id: dict[int, str] = {}
    for inst in health.get("instances", []):
        if inst.get("endpoint") != "generate":
            continue
        iid = inst.get("instance_id")
        transport = inst.get("transport") or {}
        tcp = transport.get("tcp")
        if iid is None or not tcp:
            continue
        host = _parse_tcp_host(tcp)
        by_id.setdefault(iid, host)
    return by_id


def wait_for_workers(
    frontend_url: str,
    expected_count: int,
    dyn_system_port: int,
    timeout: int = 600,
    poll_interval: float = 2.0,
) -> list[DiscoveredWorker]:
    """Poll frontend /health until ``expected_count`` distinct node_rank=0 workers
    have registered, then discover each one's topology.

    Raises TimeoutError if the count isn't reached in ``timeout`` seconds.
    """
    if expected_count <= 0:
        raise ValueError(f"expected_count must be positive, got {expected_count}")

    deadline = time.time() + timeout
    last_seen = -1
    while time.time() < deadline:
        health = _fetch_health(frontend_url)
        if health is None:
            logger.info("Waiting for Dynamo frontend at %s (not yet healthy)", frontend_url)
            time.sleep(poll_interval)
            continue

        by_id = _instances_by_id(health)
        if len(by_id) != last_seen:
            logger.info("Dynamo frontend reports %d unique instance(s) (want %d)", len(by_id), expected_count)
            last_seen = len(by_id)
        if len(by_id) < expected_count:
            time.sleep(poll_interval)
            continue

        # We have enough instances registered; discover each one's topology.
        discovered: list[DiscoveredWorker] = []
        for iid, host in by_id.items():
            state = _fetch_internal_state(host, dyn_system_port)
            if state is None:
                # Worker not yet responding on system port — keep polling.
                discovered = []
                break
            node_rank = int(state.get("node_rank") or 0)
            if node_rank != 0:
                logger.info("Skipping instance_id=%s (node_rank=%d, not coordinator)", iid, node_rank)
                continue
            discovered.append(
                DiscoveredWorker(
                    instance_id=int(iid),
                    host=host,
                    system_port=dyn_system_port,
                    tp_size=int(state.get("tp_size") or 1),
                    pp_size=int(state.get("pp_size") or 1),
                    dp_size=int(state.get("dp_size") or 1),
                    ep_size=int(state.get("ep_size") or 1),
                    nnodes=int(state.get("nnodes") or 1),
                    node_rank=node_rank,
                    disaggregation_mode=str(state.get("disaggregation_mode") or "null"),
                    served_model_name=state.get("served_model_name"),
                )
            )

        if len(discovered) >= expected_count:
            logger.info(
                "Discovered %d Dynamo workers behind %s: %s",
                len(discovered),
                frontend_url,
                [(w.instance_id, w.host, w.worker_type, w.tp_size) for w in discovered],
            )
            return sorted(discovered, key=lambda w: (w.worker_type, w.instance_id))

        time.sleep(poll_interval)

    raise TimeoutError(
        f"External Dynamo frontend at {frontend_url} did not report {expected_count} "
        f"node_rank=0 workers within {timeout}s"
    )


def topology_fingerprint(workers: Iterable[DiscoveredWorker]) -> str:
    """Deterministic hash of the topology-relevant subset of worker state.

    Used by the weight-update path to detect churn: if this hash changes
    between updates, tear down the NCCL group and reform.
    """
    import hashlib

    key = sorted(
        (w.instance_id, w.host, w.system_port, w.tp_size, w.pp_size, w.dp_size, w.ep_size, w.worker_type)
        for w in workers
    )
    return hashlib.sha256(repr(key).encode()).hexdigest()
