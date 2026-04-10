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

    router_mode = "kv" if has_pd_disaggregation else "round-robin"
    discovery_backend = getattr(args, "dynamo_discovery_backend", "file")

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
