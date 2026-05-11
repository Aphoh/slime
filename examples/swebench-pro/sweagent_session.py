"""Client helpers for SWE-bench Pro sessionful SWE-agent tool workers."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from nats.aio.client import Client as NATS


def _json_default(value: Any) -> str:
    return str(value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass
class SweAgentSessionClient:
    nats_url: str
    timeout: float = 1800.0
    session_workers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, args: Any | None = None) -> "SweAgentSessionClient":
        nats_url = os.getenv("SWEPRO_NATS_URL")
        if not nats_url and args is not None:
            nats_url = getattr(args, "swepro_nats_url", None)
        return cls(
            nats_url=nats_url or "nats://warnold-swepro-nats:4222",
            timeout=float(os.getenv("SWEPRO_SESSION_TIMEOUT", "1800")),
        )

    async def _request(self, subject: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        nc = NATS()
        await nc.connect(servers=[self.nats_url])
        try:
            msg = await nc.request(
                subject,
                json.dumps(payload, default=_json_default).encode("utf-8"),
                timeout=timeout or self.timeout,
            )
            response = json.loads(msg.data.decode("utf-8"))
        finally:
            await nc.drain()
        if response.get("error"):
            raise RuntimeError(f"{subject} failed: {response['error']}")
        return response

    def _subject(self, method: str, session_id: str | None = None) -> str:
        if session_id:
            worker_id = self.session_workers.get(session_id)
            if worker_id:
                return f"swepro.sessions.{worker_id}.{method}"
        return f"swepro.sessions.{method}"

    async def start(
        self,
        *,
        instance_id: str,
        image_name: str,
        base_commit: str,
        repo_name: str = "app",
        session_id: str | None = None,
        sample: dict[str, Any] | None = None,
        agent_context: dict[str, Any] | None = None,
        tool_events_zmq_endpoint: str | None = None,
    ) -> dict[str, Any]:
        start_timeout = float(os.getenv("SWEPRO_SESSION_START_TIMEOUT", "2400"))
        retry_delay = float(os.getenv("SWEPRO_SESSION_CAPACITY_RETRY_DELAY", "10"))
        deadline = time.monotonic() + start_timeout
        payload = {
            "session_id": session_id or str(uuid.uuid4()),
            "instance_id": instance_id,
            "image_name": image_name,
            "base_commit": base_commit,
            "repo_name": repo_name,
            "sample": sample or {},
        }
        if agent_context:
            payload["agent_context"] = dict(agent_context)
        if tool_events_zmq_endpoint:
            payload["tool_events_zmq_endpoint"] = tool_events_zmq_endpoint
        while True:
            try:
                response = await self._request("swepro.sessions.start", payload, timeout=start_timeout)
                worker_id = response.get("worker_id")
                if worker_id:
                    self.session_workers[response["session_id"]] = str(worker_id)
                return response
            except RuntimeError as exc:
                if "session capacity reached" not in str(exc) or time.monotonic() + retry_delay >= deadline:
                    raise
                await asyncio.sleep(retry_delay)

    async def step(self, session_id: str, tool_call: dict[str, Any], thought: str = "") -> dict[str, Any]:
        timeout = _env_float("SWEPRO_SESSION_STEP_REQUEST_TIMEOUT", min(self.timeout, 300.0))
        return await self._request(
            self._subject("step", session_id),
            {"session_id": session_id, "tool_call": tool_call, "thought": thought},
            timeout=timeout,
        )

    async def submit(self, session_id: str) -> dict[str, Any]:
        timeout = _env_float("SWEPRO_SESSION_SUBMIT_REQUEST_TIMEOUT", min(self.timeout, 300.0))
        return await self._request(self._subject("submit", session_id), {"session_id": session_id}, timeout=timeout)

    async def close(self, session_id: str) -> dict[str, Any]:
        try:
            timeout = _env_float("SWEPRO_SESSION_CLOSE_TIMEOUT", 120.0)
            return await self._request(self._subject("close", session_id), {"session_id": session_id}, timeout=timeout)
        finally:
            self.session_workers.pop(session_id, None)

    async def health(self) -> dict[str, Any]:
        timeout = _env_float("SWEPRO_SESSION_HEALTH_TIMEOUT", 30.0)
        return await self._request("swepro.sessions.health", {}, timeout=timeout)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
