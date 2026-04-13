"""Start a Dynamo frontend as a replacement for sglang_router.

The Dynamo frontend uses file-based discovery by default — no NATS/etcd
needed.  Workers (launched by DynamoEngine) register themselves automatically.
"""

import logging
import os
import subprocess
import sys
import time

import requests

from slime.utils.http_utils import find_available_port, get_host_info, _wrap_ipv6

logger = logging.getLogger(__name__)

_etcd_nats_started = False


def _ensure_etcd_nats():
    """Start etcd and NATS if not already running (needed for KV router)."""
    global _etcd_nats_started
    if _etcd_nats_started:
        return

    import shutil

    for name, cmd in [
        ("etcd", ["etcd", "--listen-client-urls", "http://0.0.0.0:2379",
                   "--advertise-client-urls", "http://127.0.0.1:2379"]),
        ("nats-server", ["nats-server", "-a", "0.0.0.0", "-p", "4222"]),
    ]:
        if shutil.which(name) is None:
            raise RuntimeError(f"{name} not found on PATH; install it for KV router support")
        logger.info("Starting %s for KV router discovery", name)
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Give etcd/nats a moment to bind
    time.sleep(2)
    _etcd_nats_started = True


def start_dynamo_frontend(args, *, has_pd_disaggregation: bool = False, force_new: bool = False):
    """Launch a Dynamo frontend and return ``(ip, port)``.

    Signature matches ``_start_router()`` so it can be swapped in directly.
    """
    if not force_new and getattr(args, "sglang_router_ip", None) is not None:
        return args.sglang_router_ip, args.sglang_router_port

    frontend_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        frontend_port = find_available_port(3000)
    else:
        frontend_port = getattr(args, "sglang_router_port", None) or find_available_port(3000)

    router_mode = getattr(args, "dynamo_router_mode", None) or ("kv" if has_pd_disaggregation else "round-robin")
    discovery_backend = getattr(args, "dynamo_discovery_backend", "file")

    # KV router requires etcd + NATS for event streaming between workers and router.
    if router_mode == "kv" and discovery_backend == "file":
        discovery_backend = "etcd"
        logger.info("KV router mode requires etcd discovery; switching from file to etcd")

    if discovery_backend == "etcd":
        _ensure_etcd_nats()

    cmd = [
        sys.executable, "-m", "dynamo.frontend",
        "--http-port", str(frontend_port),
        "--router-mode", router_mode,
        "--discovery-backend", discovery_backend,
    ]

    env = os.environ.copy()
    env["DYN_DISCOVERY_BACKEND"] = discovery_backend

    logger.info("Launching Dynamo frontend: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, env=env)

    # Wait for frontend health
    deadline = time.time() + 60
    url = f"http://{frontend_ip}:{frontend_port}/health"
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("Dynamo frontend healthy at %s:%s", frontend_ip, frontend_port)
                return frontend_ip, frontend_port
        except requests.RequestException:
            pass
        if process.poll() is not None:
            raise RuntimeError(f"Dynamo frontend exited with code {process.returncode}")
        time.sleep(2)

    raise TimeoutError(f"Dynamo frontend at {frontend_ip}:{frontend_port} not healthy after 60s")
