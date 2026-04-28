#!/usr/bin/env python3
"""NATS worker for sessionful SWE-bench Pro SWE-agent tool execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nats.aio.client import Client as NATS

CONTROL_PLANE_METHODS = {"close", "health"}


def _json_default(value: Any) -> str:
    return str(value)


def _safe_worker_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "worker"


def _extract_submission(observation: str) -> str | None:
    marker = "<<SWE_AGENT_SUBMISSION>>"
    parts = observation.split(marker)
    if len(parts) >= 3:
        return parts[1].strip() + ("\n" if parts[1].strip() else "")
    return None


def _tool_error_observation(exc: Exception) -> str:
    extra = getattr(exc, "extra_info", None)
    if isinstance(extra, dict) and ("bash_stdout" in extra or "bash_stderr" in extra):
        return (
            "Your bash command contained syntax errors and was NOT executed. "
            "Please fix the syntax errors and try again. This can be the result "
            "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
            f"{extra.get('bash_stdout', '')}\n{extra.get('bash_stderr', '')}"
        )
    message = getattr(exc, "message", None) or str(exc)
    return f"{type(exc).__name__}: {message}"


def _load_sweagent(eval_root: Path):
    sweagent_root = eval_root / "SWE-agent"
    sys.path.insert(0, str(sweagent_root))
    os.environ.setdefault("SWE_AGENT_CONFIG_ROOT", str(sweagent_root))

    import yaml  # type: ignore
    from swerex.deployment.config import DockerDeploymentConfig  # type: ignore
    import swerex.deployment.docker as swerex_docker  # type: ignore
    from sweagent.environment.repo import PreExistingRepoConfig  # type: ignore
    from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv  # type: ignore
    from sweagent.tools.tools import ToolConfig, ToolHandler  # type: ignore
    from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError  # type: ignore

    original_get_swerex_start_cmd = swerex_docker.DockerDeployment._get_swerex_start_cmd

    def _build_image_with_host_network(self):
        if self._config.python_standalone_dir == "/__system__":
            dockerfile = (
                "ARG BASE_IMAGE\n"
                "FROM ghcr.io/astral-sh/uv:latest AS uv\n"
                "FROM $BASE_IMAGE\n"
                "COPY --from=uv /uv /uvx /bin/\n"
                "ENV UV_CACHE_DIR=/opt/uv-cache\n"
                "ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python\n"
                "RUN /bin/uv venv /opt/swerex-venv --managed-python --python 3.11\n"
                "RUN /bin/uv pip install --python /opt/swerex-venv/bin/python "
                "--index-url https://pypi.org/simple swe-rex==1.2.0\n"
                "RUN /opt/swerex-venv/bin/swerex-remote --version\n"
            )
            build_cmd = [
                "docker",
                "build",
                "-q",
                "--network",
                "host",
                "--build-arg",
                f"BASE_IMAGE={self._config.image}",
                "-",
            ]
            image_id = subprocess.check_output(build_cmd, input=dockerfile.encode()).decode().strip()
            if not image_id.startswith("sha256:"):
                raise RuntimeError(f"Failed to build image. Image ID is not a SHA256: {image_id}")
            return image_id

        dockerfile = self.glibc_dockerfile
        platform_arg = []
        if self._config.platform:
            platform_arg = ["--platform", self._config.platform]
        build_cmd = [
            "docker",
            "build",
            "-q",
            "--network",
            "host",
            *platform_arg,
            "--build-arg",
            f"BASE_IMAGE={self._config.image}",
            "-",
        ]
        image_id = subprocess.check_output(build_cmd, input=dockerfile.encode()).decode().strip()
        if not image_id.startswith("sha256:"):
            raise RuntimeError(f"Failed to build image. Image ID is not a SHA256: {image_id}")
        return image_id

    def _get_swerex_start_cmd(self, token: str):
        if self._config.python_standalone_dir == "/__system__":
            return ["/opt/swerex-venv/bin/swerex-remote", "--auth-token", token]
        return original_get_swerex_start_cmd(self, token)

    swerex_docker.DockerDeployment._build_image = _build_image_with_host_network
    swerex_docker.DockerDeployment._get_swerex_start_cmd = _get_swerex_start_cmd

    return {
        "yaml": yaml,
        "DockerDeploymentConfig": DockerDeploymentConfig,
        "PreExistingRepoConfig": PreExistingRepoConfig,
        "EnvironmentConfig": EnvironmentConfig,
        "SWEEnv": SWEEnv,
        "ToolConfig": ToolConfig,
        "ToolHandler": ToolHandler,
        "BashIncorrectSyntaxError": BashIncorrectSyntaxError,
        "CommandTimeoutError": CommandTimeoutError,
        "sweagent_root": sweagent_root,
    }


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        name = ((tool or {}).get("function") or {}).get("name")
        if isinstance(name, str):
            names.add(name)
    return names


@dataclass
class Session:
    session_id: str
    instance_id: str
    env: Any
    tools: Any
    started_at: float
    last_used_at: float


class SessionManager:
    def __init__(
        self,
        *,
        eval_root: Path,
        max_sessions: int,
        cpus: str,
        memory: str,
        startup_timeout: float,
        worker_id: str,
    ) -> None:
        self.eval_root = eval_root
        self.max_sessions = max_sessions
        self.cpus = cpus
        self.memory = memory
        self.startup_timeout = startup_timeout
        self.worker_id = worker_id
        self.sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._starting_sessions = 0
        self._imports = _load_sweagent(eval_root)
        self._tools_config = self._load_tools_config()
        self._tool_schema = self._tools_config.tools
        self._valid_tools = _tool_names(self._tool_schema)

    def _load_tools_config(self):
        yaml = self._imports["yaml"]
        ToolConfig = self._imports["ToolConfig"]
        config_path = self._imports["sweagent_root"] / "config" / "tool_use.yaml"
        agent_cfg = yaml.safe_load(config_path.read_text())["agent"]
        return ToolConfig.model_validate(agent_cfg["tools"])

    def start(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id") or str(uuid.uuid4())
        instance_id = request["instance_id"]
        image_name = request["image_name"]
        base_commit = request["base_commit"]
        repo_name = request.get("repo_name") or "app"

        with self._lock:
            active_or_starting = len(self.sessions) + self._starting_sessions
        if active_or_starting >= self.max_sessions:
            self._prune_unhealthy_sessions()

        with self._lock:
            active_or_starting = len(self.sessions) + self._starting_sessions
            if active_or_starting >= self.max_sessions:
                raise RuntimeError(f"session capacity reached: {active_or_starting} >= {self.max_sessions}")
            if session_id in self.sessions:
                raise RuntimeError(f"duplicate session_id={session_id}")
            self._starting_sessions += 1

        DockerDeploymentConfig = self._imports["DockerDeploymentConfig"]
        EnvironmentConfig = self._imports["EnvironmentConfig"]
        PreExistingRepoConfig = self._imports["PreExistingRepoConfig"]
        SWEEnv = self._imports["SWEEnv"]
        ToolHandler = self._imports["ToolHandler"]

        docker_args = [
            "--label",
            "owner=warnold",
            "--label",
            "app=swepro-session",
            "--label",
            f"swepro.session={session_id}",
            "--entrypoint=",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
        ]
        docker_args.extend(request.get("docker_args") or [])

        deployment = DockerDeploymentConfig(
            image=image_name,
            startup_timeout=self.startup_timeout,
            pull="missing",
            python_standalone_dir="/__system__",
            docker_args=docker_args,
        )
        env_cfg = EnvironmentConfig(
            deployment=deployment,
            repo=PreExistingRepoConfig(repo_name=repo_name, base_commit=base_commit),
        )
        env = None
        try:
            env = SWEEnv.from_config(env_cfg)
            tools = ToolHandler(self._tools_config)
            env.start()
            tools.install(env)
            env.set_env_variables(
                {
                    "PAGER": "cat",
                    "GIT_PAGER": "cat",
                    "GH_PAGER": "cat",
                    "LESS": "-F -X",
                    "TERM": "dumb",
                }
            )
        except Exception:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    traceback.print_exc()
            with self._lock:
                self._starting_sessions -= 1
            raise

        now = time.time()
        with self._lock:
            self._starting_sessions -= 1
            self.sessions[session_id] = Session(
                session_id=session_id,
                instance_id=instance_id,
                env=env,
                tools=tools,
                started_at=now,
                last_used_at=now,
            )
        state = tools.get_state(env)
        return {
            "session_id": session_id,
            "worker_id": self.worker_id,
            "instance_id": instance_id,
            "tools": self._tool_schema,
            "state": state,
            "error": None,
        }

    def _execution_timeout(self) -> float:
        raw = os.getenv("SWEPRO_SESSION_STEP_TIMEOUT")
        if raw is None or raw == "":
            return float(self._tools_config.execution_timeout)
        return float(raw)

    def _drop_session(self, session_id: str, *, reason: str | None = None) -> str | None:
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return None
        close_error = None
        try:
            session.env.close()
        except Exception as exc:
            close_error = repr(exc)
            print(
                f"swepro-session: best-effort close failed for {session_id}"
                f"{f' ({reason})' if reason else ''}: {close_error}",
                flush=True,
            )
            traceback.print_exc()
        return close_error

    def _prune_unhealthy_sessions(self) -> None:
        for session_id, session in list(self.sessions.items()):
            try:
                session.tools.get_state(session.env)
            except Exception:
                print(f"swepro-session: pruning unhealthy session {session_id}", flush=True)
                self._drop_session(session_id, reason="health check failed")

    def _get(self, session_id: str) -> Session:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(f"unknown session_id={session_id}")
            session.last_used_at = time.time()
        return session

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._get(request["session_id"])
        tool_call = request["tool_call"]
        name = tool_call.get("function", {}).get("name")
        if name not in self._valid_tools:
            observation = f"Invalid tool {name!r}; valid tools: {sorted(self._valid_tools)}"
            state = session.tools.get_state(session.env)
            return {
                "session_id": session.session_id,
                "worker_id": self.worker_id,
                "instance_id": session.instance_id,
                "thought": request.get("thought", ""),
                "action": "",
                "run_action": "",
                "observation": observation,
                "state": state,
                "submitted": False,
                "done": False,
                "submission": "",
                "tool_error": "invalid_tool",
                "error": None,
            }
        output = {"message": request.get("thought", ""), "tool_calls": [tool_call]}
        try:
            thought, action = session.tools.parse_actions(output)
            run_action = session.tools.guard_multiline_input(action).strip()
        except Exception as exc:
            state = session.tools.get_state(session.env)
            return {
                "session_id": session.session_id,
                "worker_id": self.worker_id,
                "instance_id": session.instance_id,
                "thought": request.get("thought", ""),
                "action": "",
                "run_action": "",
                "observation": _tool_error_observation(exc),
                "state": state,
                "submitted": False,
                "done": False,
                "submission": "",
                "tool_error": type(exc).__name__,
                "error": None,
            }
        timeout = self._execution_timeout()
        try:
            observation = session.env.communicate(
                input=run_action,
                timeout=timeout,
                check="ignore",
            )
        except self._imports["BashIncorrectSyntaxError"] as exc:
            observation = _tool_error_observation(exc)
        except self._imports["CommandTimeoutError"]:
            try:
                session.env.interrupt_session()
            except Exception:
                traceback.print_exc()
            observation = (
                f"The command was cancelled because it took more than {timeout:g} seconds. "
                "Please inspect less broadly or run a narrower command."
            )
        state = session.tools.get_state(session.env)
        submission = _extract_submission(observation)
        submitted = submission is not None or session.tools.check_for_submission_cmd(observation)
        return {
            "session_id": session.session_id,
            "worker_id": self.worker_id,
            "instance_id": session.instance_id,
            "thought": thought,
            "action": action,
            "run_action": run_action,
            "observation": observation,
            "state": state,
            "submitted": submitted,
            "done": submitted,
            "submission": submission,
            "error": None,
        }

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._get(request["session_id"])
        output = {"message": "", "tool_calls": [{"id": "call_submit", "type": "function", "function": {"name": "submit", "arguments": "{}"}}]}
        _, action = session.tools.parse_actions(output)
        observation = session.env.communicate(
            input=session.tools.guard_multiline_input(action).strip(),
            timeout=session.tools.config.execution_timeout,
            check="ignore",
        )
        state = session.tools.get_state(session.env)
        return {
            "session_id": session.session_id,
            "worker_id": self.worker_id,
            "observation": observation,
            "submission": _extract_submission(observation) or "",
            "state": state,
            "error": None,
        }

    def close(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request["session_id"]
        close_error = self._drop_session(session_id, reason="explicit close")
        return {
            "session_id": session_id,
            "worker_id": self.worker_id,
            "closed": True,
            "close_error": close_error,
            "error": None,
        }

    def health(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        if request.get("prune"):
            self._prune_unhealthy_sessions()
        with self._lock:
            sessions = list(self.sessions)
            num_sessions = len(self.sessions)
            starting_sessions = self._starting_sessions
        return {
            "sessions": sessions,
            "num_sessions": num_sessions,
            "starting_sessions": starting_sessions,
            "max_sessions": self.max_sessions,
            "worker_id": self.worker_id,
            "tools": sorted(self._valid_tools),
            "error": None,
        }

    def close_idle(self, idle_seconds: float) -> None:
        now = time.time()
        with self._lock:
            stale = [sid for sid, session in self.sessions.items() if now - session.last_used_at > idle_seconds]
        for sid in stale:
            try:
                self.close({"session_id": sid})
            except Exception:
                traceback.print_exc()


async def handle_request(
    manager: Any,
    method_name: str,
    msg: Any,
    data_semaphore: asyncio.Semaphore,
    control_semaphore: asyncio.Semaphore,
) -> None:
    selected_semaphore = control_semaphore if method_name in CONTROL_PLANE_METHODS else data_semaphore
    async with selected_semaphore:
        try:
            request = json.loads(msg.data.decode("utf-8") or "{}")
            result = await asyncio.to_thread(getattr(manager, method_name), request)
        except Exception as exc:
            traceback.print_exc()
            result = {"error": repr(exc), "traceback": traceback.format_exc()}
        await msg.respond(json.dumps(result, default=_json_default).encode("utf-8"))


def complete_background_task(background_tasks: set[asyncio.Task], task: asyncio.Task) -> None:
    background_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        raise
    except Exception:
        traceback.print_exc()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nats-url", default=os.getenv("SWEPRO_NATS_URL", "nats://warnold-swepro-nats:4222"))
    parser.add_argument("--eval-root", type=Path, default=Path(os.getenv("SWEPRO_EVAL_ROOT", "/code/SWE-bench_Pro-os")))
    parser.add_argument("--max-sessions", type=int, default=int(os.getenv("SWEPRO_MAX_SESSIONS", "2")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("SWEPRO_SESSION_CONCURRENCY", "2")))
    parser.add_argument(
        "--control-concurrency",
        type=int,
        default=int(os.getenv("SWEPRO_SESSION_CONTROL_CONCURRENCY", "4")),
        help="Concurrency for control-plane requests such as health and close.",
    )
    parser.add_argument("--cpus", default=os.getenv("SWEPRO_SESSION_CHILD_CPUS", "4"))
    parser.add_argument("--memory", default=os.getenv("SWEPRO_SESSION_CHILD_MEMORY", "16g"))
    parser.add_argument("--startup-timeout", type=float, default=float(os.getenv("SWEPRO_SESSION_STARTUP_TIMEOUT", "1800")))
    parser.add_argument("--idle-seconds", type=float, default=float(os.getenv("SWEPRO_SESSION_IDLE_SECONDS", "3600")))
    parser.add_argument(
        "--worker-id",
        default=_safe_worker_id(os.getenv("SWEPRO_SESSION_WORKER_ID") or socket.gethostname()),
    )
    parser.add_argument("--start-queue", default=os.getenv("SWEPRO_SESSION_START_QUEUE", "swepro-session-start"))
    parser.add_argument(
        "--legacy-subjects",
        action="store_true",
        default=os.getenv("SWEPRO_SESSION_LEGACY_SUBJECTS", "0").lower() in {"1", "true", "yes"},
    )
    args = parser.parse_args()
    args.worker_id = _safe_worker_id(args.worker_id)

    manager = SessionManager(
        eval_root=args.eval_root,
        max_sessions=args.max_sessions,
        cpus=args.cpus,
        memory=args.memory,
        startup_timeout=args.startup_timeout,
        worker_id=args.worker_id,
    )
    print(f"swepro-session: manager initialized worker_id={args.worker_id}", flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    control_semaphore = asyncio.Semaphore(args.control_concurrency)
    nc = NATS()
    print(f"swepro-session: connecting to {args.nats_url}", flush=True)
    await nc.connect(servers=[args.nats_url])
    print("swepro-session: connected to NATS", flush=True)

    background_tasks: set[asyncio.Task] = set()

    def schedule(method_name: str):
        async def callback(msg):
            task = asyncio.create_task(handle_request(manager, method_name, msg, semaphore, control_semaphore))
            background_tasks.add(task)
            task.add_done_callback(lambda task: complete_background_task(background_tasks, task))

        return callback

    await nc.subscribe("swepro.sessions.start", queue=args.start_queue, cb=schedule("start"))
    worker_subject_prefix = f"swepro.sessions.{args.worker_id}"
    await nc.subscribe(f"{worker_subject_prefix}.step", cb=schedule("step"))
    await nc.subscribe(f"{worker_subject_prefix}.submit", cb=schedule("submit"))
    await nc.subscribe(f"{worker_subject_prefix}.close", cb=schedule("close"))
    await nc.subscribe(f"{worker_subject_prefix}.health", cb=schedule("health"))
    if args.legacy_subjects:
        await nc.subscribe("swepro.sessions.step", cb=schedule("step"))
        await nc.subscribe("swepro.sessions.submit", cb=schedule("submit"))
        await nc.subscribe("swepro.sessions.close", cb=schedule("close"))
    await nc.subscribe("swepro.sessions.health", cb=schedule("health"))
    await nc.flush()
    print(
        "swepro-session: subscriptions registered "
        f"start_queue={args.start_queue} worker_subject_prefix={worker_subject_prefix}",
        flush=True,
    )

    while True:
        await asyncio.sleep(60)
        manager.close_idle(args.idle_seconds)


if __name__ == "__main__":
    asyncio.run(main())
