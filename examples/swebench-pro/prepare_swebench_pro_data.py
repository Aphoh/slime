#!/usr/bin/env python3
"""Normalize SWE-bench Pro rows into slime prompt-data JSONL."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("~/proj/SWE-bench_Pro-os/helper_code/sweap_eval_full_v2.jsonl").expanduser()
DEFAULT_OUTPUT = Path("/data/swebench-pro/swebench_pro_train.jsonl")


def _load_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except Exception:
                pass
        return [text]
    return value


def _as_list(value: Any) -> list[str]:
    loaded = _load_jsonish(value)
    if loaded is None:
        return []
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]


def _prompt_from_row(row: dict[str, Any]) -> str:
    problem = row.get("problem_statement") or row.get("prompt") or ""
    requirements = row.get("requirements")
    interface = row.get("interface") or row.get("new_interfaces") or row.get("New interfaces introduced")

    parts = [str(problem).strip()]
    if requirements:
        parts.append("Requirements:\n" + str(requirements).strip())
    if interface:
        parts.append("New interfaces introduced:\n" + str(interface).strip())
    return "\n\n".join(part for part in parts if part)


def _sweagent_problem_statement(row: dict[str, Any], source_root: Path | None) -> str:
    if source_root is not None:
        helper_dir = source_root / "helper_code"
        if helper_dir.exists():
            sys.path.insert(0, str(helper_dir))
            try:
                from create_problem_statement import create_problem_statement  # type: ignore

                return create_problem_statement(row)
            except Exception:
                pass
    return _prompt_from_row(row)


def _sweagent_image_name(row: dict[str, Any], source_root: Path | None, dockerhub_username: str) -> str | None:
    if source_root is None or not row.get("repo"):
        return row.get("image_name")
    helper_dir = source_root / "helper_code"
    if helper_dir.exists():
        sys.path.insert(0, str(helper_dir))
        try:
            from image_uri import get_dockerhub_image_uri  # type: ignore

            return get_dockerhub_image_uri(row["instance_id"], dockerhub_username, row["repo"])
        except Exception:
            return row.get("image_name")
    return row.get("image_name")


def normalize_row(
    row: dict[str, Any], *, source_root: Path | None = None, dockerhub_username: str = "jefzda"
) -> dict[str, Any]:
    instance_id = row["instance_id"]
    fail_to_pass = _as_list(row.get("fail_to_pass", row.get("FAIL_TO_PASS")))
    pass_to_pass = _as_list(row.get("pass_to_pass", row.get("PASS_TO_PASS")))
    selected_test_files = _as_list(row.get("selected_test_files_to_run"))
    sweagent_problem = _sweagent_problem_statement(row, source_root)
    image_name = _sweagent_image_name(row, source_root, dockerhub_username)

    raw_row = dict(row)
    raw_row["fail_to_pass"] = fail_to_pass
    raw_row["pass_to_pass"] = pass_to_pass
    raw_row["selected_test_files_to_run"] = selected_test_files

    run_scripts_dir = None
    if source_root is not None:
        candidate = source_root / "run_scripts" / instance_id
        if candidate.exists():
            run_scripts_dir = str(candidate)

    metadata = {
        "instance_id": instance_id,
        "repo": row.get("repo"),
        "base_commit": row.get("base_commit"),
        "selected_test_files_to_run": selected_test_files,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "image_name": image_name,
        "repo_name": row.get("repo_name"),
        "problem_statement": sweagent_problem,
        "sweagent": {
            "repo_name": "app",
            "image_name": image_name,
        },
        "before_repo_set_cmd": row.get("before_repo_set_cmd"),
        "base_dockerfile": row.get("base_dockerfile"),
        "instance_dockerfile": row.get("instance_dockerfile"),
        "run_script": row.get("run_script"),
        "parsing_script": row.get("parsing_script"),
        "run_scripts_dir": run_scripts_dir,
        "source_root": str(source_root) if source_root else None,
        "raw_row": raw_row,
    }
    if "patch" in row:
        metadata["gold_patch"] = row["patch"]
    if "test_patch" in row:
        metadata["test_patch"] = row["test_patch"]

    return {
        "prompt": _prompt_from_row(row),
        "instance_id": instance_id,
        "metadata": metadata,
    }


def convert(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    source_root: Path | None = None,
    dockerhub_username: str = "jefzda",
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(
                json.dumps(
                    normalize_row(row, source_root=source_root, dockerhub_username=dockerhub_username),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=Path("~/proj/SWE-bench_Pro-os").expanduser())
    parser.add_argument("--dockerhub-username", default="jefzda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count = convert(
        args.input.expanduser(),
        args.output,
        limit=args.limit,
        source_root=args.source_root.expanduser(),
        dockerhub_username=args.dockerhub_username,
    )
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
