#!/usr/bin/env python3
"""NATS worker for SWE-bench Pro Docker evaluations."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import docker
from nats.aio.client import Client as NATS

EVAL_TRACE_LOG_PREFIX = "SWEPRO_EVAL_TRACE"
TEST_PREVIEW_LIMIT = 50


def _json_default(value):
    try:
        return value.item()
    except Exception:
        return str(value)


def _ensure_eval_import(eval_root: Path):
    sys.path.insert(0, str(eval_root))
    import swe_bench_pro_eval as eval_mod  # type: ignore
    from helper_code.image_uri import get_dockerhub_image_uri  # type: ignore

    # The upstream helpers read dockerfiles via process-relative paths. Avoid
    # os.chdir here because eval requests run concurrently in worker threads.
    def load_base_docker(iid):
        return (eval_root / "dockerfiles" / "base_dockerfile" / iid / "Dockerfile").read_text()

    def instance_docker(iid):
        return (eval_root / "dockerfiles" / "instance_dockerfile" / iid / "Dockerfile").read_text()

    def load_local_script(scripts_dir, instance_id, script_name):
        scripts_path = Path(scripts_dir)
        if not scripts_path.is_absolute():
            scripts_path = eval_root / scripts_path
        script_path = scripts_path / instance_id / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        return script_path.read_text()

    eval_mod.load_base_docker = load_base_docker
    eval_mod.instance_docker = instance_docker
    eval_mod.load_local_script = load_local_script

    return eval_mod, get_dockerhub_image_uri


def _normalize_sample(sample: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(sample)
    normalized["instance_id"] = request["instance_id"]
    normalized["repo"] = request.get("repo") or normalized.get("repo")
    for key in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run"):
        value = request.get(key, normalized.get(key))
        if isinstance(value, list):
            normalized[key] = json.dumps(value)
        elif value is None:
            normalized[key] = "[]"
        else:
            normalized[key] = value
    normalized["FAIL_TO_PASS"] = normalized.get("FAIL_TO_PASS", normalized["fail_to_pass"])
    normalized["PASS_TO_PASS"] = normalized.get("PASS_TO_PASS", normalized["pass_to_pass"])
    return normalized


def _tail(path: Path, limit: int = 8000) -> str:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    return text[-limit:]


def _eval_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"tests", "stdout_tail", "stderr_tail"}}


def evaluate_request(request: dict[str, Any], *, eval_root: Path, work_root: Path, output_root: Path, timeout: int) -> dict[str, Any]:
    eval_mod, get_dockerhub_image_uri = _ensure_eval_import(eval_root)

    request_id = request.get("request_id") or f"req-{int(time.time())}"
    instance_id = request["instance_id"]
    sample = _normalize_sample(request.get("sample") or {}, request)
    patch = request.get("patch") or ""
    dockerhub_username = os.getenv("SWEPRO_DOCKERHUB_USERNAME", "jefzda")
    scripts_dir = Path(os.getenv("SWEPRO_RUN_SCRIPTS_DIR", str(eval_root / "run_scripts")))

    workspace_dir = work_root / request_id / "workspace"
    run_output_dir = output_root / instance_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    files, entryscript_content = eval_mod.assemble_workspace_files(instance_id, str(scripts_dir), patch, sample)
    eval_mod.write_files_local(str(workspace_dir), files)
    (run_output_dir / f"{request_id}_entryscript.sh").write_text(entryscript_content or "")
    (run_output_dir / f"{request_id}_patch.diff").write_text(patch)

    image = get_dockerhub_image_uri(instance_id, dockerhub_username, sample.get("repo", ""))
    client = docker.from_env()
    container = None
    status_code = 1
    error = None
    try:
        try:
            client.images.pull(image)
        except Exception:
            client.images.get(image)

        container = client.containers.run(
            image,
            detach=True,
            remove=False,
            entrypoint="/bin/bash",
            command=["-c", "bash /workspace/entryscript.sh"],
            volumes={str(workspace_dir.resolve()): {"bind": "/workspace", "mode": "rw"}},
            labels={"owner": "warnold", "app": "swepro-eval", "request_id": request_id},
        )
        result = container.wait(timeout=timeout)
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
    except Exception as exc:
        error = repr(exc)
        if container is not None:
            try:
                container.kill()
            except Exception:
                pass
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass

    output_path = workspace_dir / "output.json"
    if output_path.exists():
        output = json.loads(output_path.read_text())
    else:
        output = {"tests": []}
        error = error or "output.json not found"

    shutil.copyfile(workspace_dir / "stdout.log", run_output_dir / f"{request_id}_stdout.log") if (workspace_dir / "stdout.log").exists() else None
    shutil.copyfile(workspace_dir / "stderr.log", run_output_dir / f"{request_id}_stderr.log") if (workspace_dir / "stderr.log").exists() else None
    shutil.copyfile(output_path, run_output_dir / f"{request_id}_output.json") if output_path.exists() else None

    tests = output.get("tests", [])
    status_counts = collections.Counter(test.get("status") or "UNKNOWN" for test in tests)
    passed_tests = {test.get("name") for test in tests if test.get("status") == "PASSED"}
    failed_tests = [test.get("name") for test in tests if test.get("status") != "PASSED"]
    required = set(request.get("fail_to_pass") or []) | set(request.get("pass_to_pass") or [])
    missing_required_tests = sorted(test for test in required - passed_tests if test)
    passed = bool(required) and not missing_required_tests and status_code == 0 and error is None

    result = {
        "request_id": request_id,
        "instance_id": instance_id,
        "passed": passed,
        "status_code": status_code,
        "tests": tests,
        "test_status_counts": dict(status_counts),
        "required_test_count": len(required),
        "missing_required_count": len(missing_required_tests),
        "missing_required_tests": missing_required_tests[:TEST_PREVIEW_LIMIT],
        "failed_tests_preview": failed_tests[:TEST_PREVIEW_LIMIT],
        "patch_chars": len(patch),
        "stdout_tail": _tail(workspace_dir / "stdout.log"),
        "stderr_tail": _tail(workspace_dir / "stderr.log"),
        "error": error,
    }
    summary = _eval_result_summary(result)
    (run_output_dir / f"{request_id}_result.json").write_text(
        json.dumps(summary, default=_json_default, indent=2, sort_keys=True)
    )
    print(f"{EVAL_TRACE_LOG_PREFIX} {json.dumps(summary, default=_json_default, sort_keys=True)}", flush=True)
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nats-url", default=os.getenv("SWEPRO_NATS_URL", "nats://warnold-swepro-nats:4222"))
    parser.add_argument("--subject", default=os.getenv("SWEPRO_NATS_SUBJECT", "swepro.evals"))
    parser.add_argument("--queue", default=os.getenv("SWEPRO_EVAL_QUEUE_GROUP", "swepro-eval-workers"))
    parser.add_argument("--eval-root", type=Path, default=Path(os.getenv("SWEPRO_EVAL_ROOT", "/opt/SWE-bench_Pro-os")))
    parser.add_argument("--work-root", type=Path, default=Path(os.getenv("SWEPRO_WORK_ROOT", "/swepro-workspaces")))
    parser.add_argument("--output-root", type=Path, default=Path(os.getenv("SWEPRO_OUTPUT_ROOT", "/swepro-workspaces/results")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("SWEPRO_EVAL_CONCURRENCY", "1")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("SWEPRO_EVAL_TIMEOUT", "3600")))
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)
    nc = NATS()
    await nc.connect(servers=[args.nats_url])

    async def handle(msg):
        async with semaphore:
            try:
                request = json.loads(msg.data.decode("utf-8"))
                result = await asyncio.to_thread(
                    evaluate_request,
                    request,
                    eval_root=args.eval_root,
                    work_root=args.work_root,
                    output_root=args.output_root,
                    timeout=args.timeout,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                print(tb, flush=True)
                result = {"passed": False, "error": repr(exc), "traceback": tb}
            await msg.respond(json.dumps(result, default=_json_default).encode("utf-8"))

    background_tasks: set[asyncio.Task] = set()

    async def schedule(msg):
        task = asyncio.create_task(handle(msg))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    await nc.subscribe(args.subject, queue=args.queue, cb=schedule)
    await nc.flush()
    print(f"swepro-eval: subscribed to {args.subject} queue={args.queue}", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
