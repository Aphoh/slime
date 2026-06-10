"""SWE-bench Pro custom generation and reward hooks for slime."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import json
import logging
import os
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import yaml

from completions_direct_model import (
    GLM_TOOL_STOPS,
    GLM_TOOL_STOP_TOKEN_IDS,
    DirectCompletionsConfig,
    DirectCompletionsModel,
    TOOL_CALL_END,
    encode_qwen_tool_observation_delta,
    parse_glm_tool_call_from_completion,
    stop_reason_token_ids,
)
from dynamo_agent_trace import build_agent_context_for_sample, derive_tool_events_zmq_endpoint, llm_request_id
from sweagent_session import SweAgentSessionClient
from trace_replay import TraceReplaySessionClient, TraceReplayStore

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
AGENT_TRACE_LOG_PREFIX = "SWEPRO_AGENT_TRACE"
PATCH_FILE_PREVIEW_LIMIT = 20

_MODEL: DirectCompletionsModel | None = None
_MASK_GENERATOR: MultiTurnLossMaskGenerator | None = None
_SWEAGENT_TEMPLATES: dict[str, str] | None = None
_TRACE_REPLAY_STORE: TraceReplayStore | None = None
_TRACE_REPLAY_STORE_PATH: str | None = None


SYSTEM_PROMPT = """You are an autonomous software engineer. Produce a minimal git patch that satisfies the issue.
Return the final answer as a unified diff only when you are ready to submit."""


def _json_log_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_log_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_log_value(item) for key, item in value.items()}
    return str(value)


def _agent_trace_log(event: str, **fields: Any) -> None:
    payload = {"event": event}
    payload.update({key: _json_log_value(value) for key, value in fields.items() if value is not None})
    logger.info("%s %s", AGENT_TRACE_LOG_PREFIX, json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _trajectory_id(agent_context: dict[str, Any] | None) -> str | None:
    if not agent_context:
        return None
    value = agent_context.get("trajectory_id")
    return str(value) if value is not None else None


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return "unknown"


def _tool_call_id(tool_call: dict[str, Any]) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    value = tool_call.get("id")
    return str(value) if value is not None else None


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %s", name, value, default)
        return default
    return parsed


def _optional_positive_int_env(name: str, default: int) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if value.lower() in {"0", "none", "unlimited"}:
        return None
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %s", name, value, default)
        return default
    return parsed


def _nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Ignoring negative %s=%r; using %s", name, value, default)
        return default
    return parsed


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _positive_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %s", name, value, default)
        return default
    return parsed


def _nonnegative_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Ignoring negative %s=%r; using %s", name, value, default)
        return default
    return parsed


def _turn_max_tokens(args, sampling_params: dict[str, Any] | None = None) -> int:
    rollout_max = int(getattr(args, "rollout_max_response_len", 4096))
    default_turn_max = min(8192, rollout_max)
    turn_max = _optional_positive_int_env("SWEPRO_TURN_MAX_TOKENS", default_turn_max)
    requested = rollout_max
    if sampling_params is not None:
        requested = int(sampling_params.get("max_new_tokens") or rollout_max)
    limits = [requested, rollout_max]
    if turn_max is not None:
        limits.append(turn_max)
    return max(1, min(limits))


def _episode_wall_timeout_s() -> float | None:
    raw = os.getenv("SWEPRO_EPISODE_WALL_TIMEOUT")
    if raw is None or raw == "" or raw.lower() in {"0", "none", "unlimited"}:
        return None
    timeout = float(raw)
    if timeout <= 0:
        return None
    return timeout


def _model_call_timeout_s() -> float:
    return _positive_float_env("SWEPRO_MODEL_CALL_TIMEOUT", 600.0)


def _session_call_timeout_s(kind: str, default: float) -> float:
    kind_upper = kind.upper()
    candidates = [
        f"SWEPRO_SESSION_{kind_upper}_CALL_TIMEOUT",
        f"SWEPRO_SESSION_{kind_upper}_REQUEST_TIMEOUT",
        f"SWEPRO_SESSION_{kind_upper}_TIMEOUT",
    ]
    for specific in candidates:
        if os.getenv(specific):
            return _positive_float_env(specific, default)
    return _positive_float_env("SWEPRO_SESSION_CALL_TIMEOUT", default)


async def _await_with_timeout(awaitable, *, timeout: float, label: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout:g}s") from exc


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _dynamo_frontend_url(args) -> str | None:
    base_url = os.getenv("SWEPRO_DYNAMO_FRONTEND_URL") or os.getenv("DYNAMO_FRONTEND_URL")
    if base_url:
        return base_url.rstrip("/")
    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if router_ip and router_port:
        return f"http://{router_ip}:{router_port}"
    return None


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except Exception:
            pass
        return [value]
    return [str(value)]


def _get_model(args) -> DirectCompletionsModel:
    global _MODEL
    if _MODEL is None:
        base_url = _dynamo_frontend_url(args)
        if not base_url:
            raise ValueError("Set SWEPRO_DYNAMO_FRONTEND_URL or DYNAMO_FRONTEND_URL")
        config = DirectCompletionsConfig(
            base_url=base_url.rstrip("/"),
            tokenizer_path=args.hf_checkpoint,
            model=os.getenv("SWEPRO_MODEL", getattr(args, "hf_checkpoint", None) or "default"),
            max_tokens=_turn_max_tokens(args),
            temperature=float(getattr(args, "rollout_temperature", 1.0)),
            top_p=float(getattr(args, "rollout_top_p", 1.0)),
            top_k=getattr(args, "rollout_top_k", None),
            timeout=float(os.getenv("SWEPRO_REQUEST_TIMEOUT", "1800")),
            retries=int(os.getenv("SWEPRO_REQUEST_RETRIES", "5")),
            metadata_upload_url=os.getenv("SWEPRO_DYNAMO_METADATA_UPLOAD_URL"),
            metadata_upload_format=os.getenv("SWEPRO_DYNAMO_METADATA_UPLOAD_FORMAT", "msgpack"),
        )
        _MODEL = DirectCompletionsModel(config)
    return _MODEL


def _get_trace_replay_store() -> TraceReplayStore | None:
    path = os.getenv("SWEPRO_TRACE_REPLAY_PATH") or os.getenv("SWEPRO_MOCK_ENV_TRACE_PATH")
    if not path:
        return None
    global _TRACE_REPLAY_STORE, _TRACE_REPLAY_STORE_PATH
    if _TRACE_REPLAY_STORE is None or _TRACE_REPLAY_STORE_PATH != path:
        sleep_scale = _nonnegative_float_env("SWEPRO_MOCK_ENV_SCALE", 1.0)
        _TRACE_REPLAY_STORE = TraceReplayStore.from_path(path, sleep_scale=sleep_scale)
        _TRACE_REPLAY_STORE_PATH = path
        logger.info("Loaded %s trace replay plans from %s", len(_TRACE_REPLAY_STORE.plans), path)
    return _TRACE_REPLAY_STORE


def _mock_sleep_scale() -> float:
    return _nonnegative_float_env("SWEPRO_MOCK_ENV_SCALE", 1.0)


def _trace_replay_force_fixed_decode() -> bool:
    return _bool_env("SWEPRO_TRACE_REPLAY_FORCE_FIXED_DECODE", False)


def _mock_reward_value() -> float:
    raw = os.getenv("SWEPRO_MOCK_ENV_REWARD") or os.getenv("SWEPRO_TRACE_REPLAY_REWARD") or "0.0"
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid mock reward value %r; using 0.0", raw)
        return 0.0


def _uses_mock_reward(metadata: dict[str, Any]) -> bool:
    if metadata.get("trace_replay") or metadata.get("mock_trace_replay"):
        return True
    return _bool_env("SWEPRO_MOCK_REWARD", False)


def _get_mask_generator(args) -> MultiTurnLossMaskGenerator:
    global _MASK_GENERATOR
    if _MASK_GENERATOR is None:
        state = GenerateState(args)
        _MASK_GENERATOR = MultiTurnLossMaskGenerator(state.tokenizer, tokenizer_type=args.loss_mask_type)
    return _MASK_GENERATOR


def _extract_patch(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(diff --git .*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    idx = text.find("diff --git ")
    if idx >= 0:
        return text[idx:].strip() + "\n"
    return text.strip()


def _find_sublist(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    width = len(needle)
    for idx in range(0, len(haystack) - width + 1):
        if haystack[idx : idx + width] == needle:
            return idx
    return -1


def _align_logprobs(
    response_tokens: list[int],
    response_loss_mask: list[int],
    generated_token_ids: list[int],
    generated_logprobs: list[float],
) -> list[float]:
    aligned = [0.0] * len(response_tokens)
    if not generated_token_ids:
        return aligned

    start = _find_sublist(response_tokens, generated_token_ids)
    if start < 0:
        masked_positions = [idx for idx, value in enumerate(response_loss_mask) if value]
        for idx, logprob in zip(masked_positions, generated_logprobs, strict=False):
            aligned[idx] = float(logprob)
        return aligned

    for offset, logprob in enumerate(generated_logprobs[: len(generated_token_ids)]):
        pos = start + offset
        if 0 <= pos < len(aligned):
            aligned[pos] = float(logprob)
    return aligned


def _parse_tool_call_from_generated_tokens(
    model: DirectCompletionsModel,
    content: str,
    generated_ids: list[int],
    matched_stop_token_ids: list[int],
) -> tuple[str, list[dict[str, Any]], bool]:
    return parse_glm_tool_call_from_completion(model.tokenizer, content, generated_ids, matched_stop_token_ids)


def _assistant_content_for_history(content: str, tool_calls: list[dict[str, Any]]) -> str:
    if tool_calls and TOOL_CALL_END not in content:
        return content + TOOL_CALL_END
    return content


def _local_patch_trace_fields(submission: str) -> dict[str, Any]:
    patch_bytes = submission.encode("utf-8", errors="backslashreplace")
    files = [
        match.group(2)
        for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", submission, flags=re.MULTILINE)
    ]
    return {
        "patch_chars": len(submission),
        "patch_sha256": hashlib.sha256(patch_bytes).hexdigest()[:16] if submission else "",
        "patch_starts_with_diff": submission.lstrip().startswith("diff --git"),
        "patch_file_count": len(files),
        "patch_files_preview": files[:PATCH_FILE_PREVIEW_LIMIT],
    }


def _submission_trace_fields(submission: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = _local_patch_trace_fields(submission)
    if isinstance(result, dict) and isinstance(result.get("patch_diagnostics"), dict):
        fields.update(result["patch_diagnostics"])
        fields.setdefault("patch_chars", len(submission))
    return fields


def _tool_observation_delta_ids(model: DirectCompletionsModel, observation: str) -> list[int]:
    return encode_qwen_tool_observation_delta(model.tokenizer, observation)


def _padding_token_id(model: DirectCompletionsModel) -> int:
    tokenizer = model.tokenizer
    return int(
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )


def _resize_prompt_token_ids(prompt_token_ids: list[int], target_length: int, pad_token_id: int) -> None:
    if target_length <= 0:
        return
    if len(prompt_token_ids) > target_length:
        del prompt_token_ids[target_length:]
    elif len(prompt_token_ids) < target_length:
        prompt_token_ids.extend([pad_token_id] * (target_length - len(prompt_token_ids)))


def _align_current_token_length(
    *,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float],
    target_length: int,
    pad_token_id: int,
) -> None:
    if target_length <= 0:
        return
    current_length = len(prompt_token_ids) + len(response_token_ids)
    if current_length == target_length:
        return
    if current_length < target_length:
        missing = target_length - current_length
        response_token_ids.extend([pad_token_id] * missing)
        loss_mask.extend([0] * missing)
        rollout_log_probs.extend([0.0] * missing)
        return

    overflow = current_length - target_length
    if overflow <= len(response_token_ids):
        del response_token_ids[-overflow:]
        del loss_mask[-overflow:]
        del rollout_log_probs[-overflow:]
    else:
        del response_token_ids[:]
        del loss_mask[:]
        del rollout_log_probs[:]
        _resize_prompt_token_ids(prompt_token_ids, target_length, pad_token_id)


def _append_untrainable_tokens(
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float],
    *,
    token_id: int,
    count: int,
) -> None:
    if count <= 0:
        return
    response_token_ids.extend([token_id] * count)
    loss_mask.extend([0] * count)
    rollout_log_probs.extend([0.0] * count)


def _build_messages(sample: Sample) -> list[dict[str, str]]:
    metadata = _metadata(sample)
    prompt = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt, ensure_ascii=False)
    repo = metadata.get("repo") or "unknown repository"
    instance_id = metadata.get("instance_id") or sample.label or "unknown"
    user = f"Instance: {instance_id}\nRepository: {repo}\n\n{prompt}\n\nReturn a unified diff patch."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _load_sweagent_templates() -> dict[str, str]:
    global _SWEAGENT_TEMPLATES
    if _SWEAGENT_TEMPLATES is None:
        root = Path(os.getenv("SWE_AGENT_CONFIG_ROOT", "/code/SWE-bench_Pro-os/SWE-agent"))
        local_root = Path("~/proj/SWE-bench_Pro-os/SWE-agent").expanduser()
        if not (root / "config/tool_use.yaml").exists() and (local_root / "config/tool_use.yaml").exists():
            root = local_root
        data = yaml.safe_load((root / "config/tool_use.yaml").read_text())
        _SWEAGENT_TEMPLATES = dict(data["agent"]["templates"])
    return _SWEAGENT_TEMPLATES


def _render_template(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def _build_sweagent_messages(sample: Sample) -> list[dict[str, Any]]:
    metadata = _metadata(sample)
    templates = _load_sweagent_templates()
    problem_statement = metadata.get("problem_statement") or sample.prompt
    if not isinstance(problem_statement, str):
        problem_statement = json.dumps(problem_statement, ensure_ascii=False)
    working_dir = metadata.get("working_dir") or "/app"
    return [
        {
            "role": "system",
            "content": _render_template(templates["system_template"], working_dir=working_dir),
        },
        {
            "role": "user",
            "content": _render_template(
                templates["instance_template"],
                working_dir=working_dir,
                problem_statement=problem_statement,
            ),
        },
    ]


def _format_observation(observation: str) -> str:
    templates = _load_sweagent_templates()
    if observation.strip():
        return _render_template(templates["next_step_template"], observation=observation)
    return templates["next_step_no_output_template"]


def _metadata_image_name(metadata: dict[str, Any]) -> str:
    sweagent = metadata.get("sweagent") if isinstance(metadata.get("sweagent"), dict) else {}
    image_name = sweagent.get("image_name") or metadata.get("image_name")
    if image_name:
        return str(image_name)
    source_root = Path(metadata.get("source_root") or os.getenv("SWEPRO_EVAL_ROOT", "/code/SWE-bench_Pro-os"))
    helper_dir = source_root / "helper_code"
    if helper_dir.exists():
        import sys

        sys.path.insert(0, str(helper_dir))
        from image_uri import get_dockerhub_image_uri  # type: ignore

        return get_dockerhub_image_uri(
            metadata.get("instance_id"),
            os.getenv("SWEPRO_DOCKERHUB_USERNAME", "jefzda"),
            metadata.get("repo") or "",
        )
    raise ValueError("SWE-bench Pro sample is missing image_name and image_uri helper is unavailable")


def _max_context_len(args) -> int | None:
    value = getattr(args, "rollout_max_context_len", None)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _trim_prompt_for_untrainable(prompt_token_ids: list[int], max_context_len: int | None) -> list[int]:
    if max_context_len is None:
        return prompt_token_ids
    return prompt_token_ids[: max(0, max_context_len - 1)]


def _remaining_context(
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    max_context_len: int | None,
) -> int | None:
    if max_context_len is None:
        return None
    # SGLang rejects requests at the exact context boundary, so keep one token
    # of headroom while still allowing effectively unbounded turns.
    return max_context_len - len(prompt_token_ids) - len(response_token_ids) - 1


def _trim_response_to_context(
    *,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float],
    max_context_len: int | None,
    metadata: dict[str, Any],
) -> bool:
    if max_context_len is None:
        return False

    response_budget = max(0, max_context_len - len(prompt_token_ids))
    if len(response_token_ids) <= response_budget:
        return False

    before_response_len = len(response_token_ids)
    before_total_len = len(prompt_token_ids) + before_response_len
    del response_token_ids[response_budget:]
    del loss_mask[response_budget:]
    del rollout_log_probs[response_budget:]

    info = {
        "max_context_len": max_context_len,
        "pre_truncate_total_length": before_total_len,
        "post_truncate_total_length": len(prompt_token_ids) + len(response_token_ids),
        "dropped_response_tokens": before_response_len - len(response_token_ids),
    }
    metadata["max_context_truncated"] = True
    metadata["max_context_truncation"] = info
    return True


def _mark_untrainable_sample(
    sample: Sample,
    *,
    prompt_token_ids: list[int],
    model: DirectCompletionsModel | None,
    metadata: dict[str, Any],
    error: Exception,
    max_context_len: int | None = None,
) -> Sample:
    """Return a structurally valid sample that contributes zero policy loss."""
    if not prompt_token_ids:
        prompt_token_ids = list(sample.tokens or [])
    if not prompt_token_ids and model is not None:
        try:
            prompt_token_ids = model.tokenizer(str(sample.prompt), add_special_tokens=False)["input_ids"]
        except Exception:
            prompt_token_ids = []

    fallback_token = 0
    if model is not None:
        fallback_token = (
            model.tokenizer.eos_token_id
            if model.tokenizer.eos_token_id is not None
            else model.tokenizer.pad_token_id or 0
        )

    prompt_token_ids = _trim_prompt_for_untrainable(prompt_token_ids, max_context_len)
    metadata["session_error"] = repr(error)
    sample.tokens = prompt_token_ids + [int(fallback_token)]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.status = Sample.Status.ABORTED
    sample.reward = 0.0
    sample.remove_sample = True
    sample.metadata = metadata
    return sample


def _finalize_sweagent_session_sample(
    sample: Sample,
    *,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    response_text_parts: list[str],
    loss_mask: list[int],
    rollout_log_probs: list[float],
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
    session_id: str | None,
    submission: str,
    finish_reason: str,
    started_at: float,
    status: Sample.Status | None = None,
    max_context_len: int | None = None,
) -> Sample:
    if len(loss_mask) != len(response_token_ids):
        raise ValueError(f"loss_mask length mismatch: {len(loss_mask)} != {len(response_token_ids)}")
    if len(rollout_log_probs) != len(response_token_ids):
        raise ValueError(f"rollout_log_probs length mismatch: {len(rollout_log_probs)} != {len(response_token_ids)}")

    if _trim_response_to_context(
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        max_context_len=max_context_len,
        metadata=metadata,
    ):
        finish_reason = "length"

    metadata.update(
        {
            "instance_id": metadata.get("instance_id") or sample.label,
            "patch": submission,
            "agent_mode": "sweagent_session",
            "messages": messages,
            "session_id": session_id,
            "session_wall_time_s": time.time() - started_at,
            "repo": metadata.get("repo"),
            "fail_to_pass": _list(metadata.get("fail_to_pass")),
            "pass_to_pass": _list(metadata.get("pass_to_pass")),
            "selected_test_files_to_run": _list(metadata.get("selected_test_files_to_run")),
        }
    )

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = "".join(response_text_parts)
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.metadata = metadata
    sample.status = status or (Sample.Status.TRUNCATED if finish_reason == "length" else Sample.Status.COMPLETED)
    return sample


async def _generate_sweagent_session(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for SWE-bench Pro."
    metadata = _metadata(sample)
    instance_id = metadata.get("instance_id") or sample.label
    if not instance_id:
        raise ValueError("SWE-bench Pro sample is missing instance_id")
    agent_context = build_agent_context_for_sample(instance_id, sample)
    tool_events_zmq_endpoint = derive_tool_events_zmq_endpoint(_dynamo_frontend_url(args))
    metadata["dynamo_agent_context"] = agent_context
    if tool_events_zmq_endpoint:
        metadata["dynamo_tool_events_zmq_endpoint"] = tool_events_zmq_endpoint
    base_commit = metadata.get("base_commit")
    if not base_commit:
        raise ValueError(f"SWE-bench Pro sample {instance_id} is missing base_commit")

    model = _get_model(args)
    trace_replay_store = _get_trace_replay_store()
    trace_replay_plan = trace_replay_store.claim(str(instance_id)) if trace_replay_store else None
    client = (
        TraceReplaySessionClient(trace_replay_plan, sleep_scale=_mock_sleep_scale())
        if trace_replay_plan is not None
        else SweAgentSessionClient.from_env(args)
    )
    if trace_replay_plan is not None:
        metadata["trace_replay"] = True
        metadata["trace_replay_trajectory_id"] = trace_replay_plan.trajectory_id
        metadata["trace_replay_source_instance_id"] = trace_replay_plan.instance_id
        metadata["trace_replay_turns"] = len(trace_replay_plan.turns)
    messages = _build_sweagent_messages(sample)
    image_name = _metadata_image_name(metadata)
    session_id = None
    prompt_token_ids: list[int] = []
    response_token_ids: list[int] = []
    response_text_parts: list[str] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    submission = ""
    finish_reason = "stop"

    max_tool_calls_raw = os.getenv("SWEPRO_MAX_TOOL_CALLS", str(getattr(args, "swepro_max_tool_calls", 20)))
    max_tool_calls = None if max_tool_calls_raw.lower() in {"", "0", "none", "unlimited"} else int(max_tool_calls_raw)
    episode_wall_timeout = _episode_wall_timeout_s()
    max_context = _max_context_len(args)
    turn_iter = itertools.count() if max_tool_calls is None else range(max_tool_calls)
    started_at = time.time()
    try:
        started = await _await_with_timeout(
            client.start(
                instance_id=str(instance_id),
                image_name=image_name,
                base_commit=str(base_commit),
                repo_name=str((metadata.get("sweagent") or {}).get("repo_name") or metadata.get("repo_name") or "app"),
                sample=metadata.get("raw_row") or metadata,
                agent_context=agent_context,
                tool_events_zmq_endpoint=tool_events_zmq_endpoint,
            ),
            timeout=_session_call_timeout_s("start", 900.0),
            label=f"SWE-agent session start for {instance_id}",
        )
        session_id = started["session_id"]
        tools = started["tools"]
        prompt_token_ids = model.encode_prompt(messages, tools=tools)
        pad_token_id = _padding_token_id(model)
        if trace_replay_plan is not None:
            _resize_prompt_token_ids(prompt_token_ids, trace_replay_plan.initial_prompt_tokens, pad_token_id)
        _agent_trace_log(
            "session_started",
            trajectory_id=_trajectory_id(agent_context),
            instance_id=instance_id,
            sample_index=getattr(sample, "index", None),
            sample_session_id=getattr(sample, "session_id", None),
            session_id=session_id,
            tool_count=len(tools or []),
            prompt_tokens=len(prompt_token_ids),
            tool_events_zmq_endpoint=tool_events_zmq_endpoint,
            trace_replay=trace_replay_plan is not None,
            trace_replay_source_trajectory_id=trace_replay_plan.trajectory_id if trace_replay_plan else None,
        )
        if trace_replay_plan is not None:
            replay_generated_tokens = [turn.generated_tokens for turn in trace_replay_plan.turns]
            replay_prompt_tokens = [turn.prompt_tokens for turn in trace_replay_plan.turns]
            _agent_trace_log(
                "trace_replay_plan_claimed",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                trace_replay_source_trajectory_id=trace_replay_plan.trajectory_id,
                trace_replay_source_instance_id=trace_replay_plan.instance_id,
                trace_replay_turns=len(trace_replay_plan.turns),
                trace_replay_initial_prompt_tokens=trace_replay_plan.initial_prompt_tokens,
                trace_replay_max_prompt_tokens=max(replay_prompt_tokens, default=0),
                trace_replay_total_generated_tokens=sum(replay_generated_tokens),
                trace_replay_max_generated_tokens=max(replay_generated_tokens, default=0),
                trace_replay_total_observation_tokens=sum(turn.observation_tokens for turn in trace_replay_plan.turns),
                trace_replay_total_tool_duration_s=round(
                    sum(turn.tool_duration_s for turn in trace_replay_plan.turns),
                    3,
                ),
                trace_replay_duration_s=round(trace_replay_plan.duration_s, 3),
            )
        if max_context is not None and len(prompt_token_ids) >= max_context:
            return _mark_untrainable_sample(
                sample,
                prompt_token_ids=prompt_token_ids,
                model=model,
                metadata=metadata,
                error=RuntimeError(
                    f"initial SWE-agent prompt length {len(prompt_token_ids)} exceeds rollout_max_context_len={max_context}"
                ),
                max_context_len=max_context,
            )

        if trace_replay_plan is not None:
            replay_turn_count = len(trace_replay_plan.turns)
            if max_tool_calls is not None:
                replay_turn_count = min(replay_turn_count, max_tool_calls)
            turn_iter = range(replay_turn_count)

        for turn in turn_iter:
            if episode_wall_timeout is not None and time.time() - started_at >= episode_wall_timeout:
                finish_reason = "length"
                break

            replay_turn = trace_replay_plan.turn_for(turn) if trace_replay_plan is not None else None
            if replay_turn is not None:
                _align_current_token_length(
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    loss_mask=loss_mask,
                    rollout_log_probs=rollout_log_probs,
                    target_length=replay_turn.prompt_tokens,
                    pad_token_id=pad_token_id,
                )
            current_ids = prompt_token_ids + response_token_ids
            remaining_context = _remaining_context(prompt_token_ids, response_token_ids, max_context)
            if remaining_context is not None and remaining_context <= 0:
                finish_reason = "length"
                break
            max_turn_tokens = _turn_max_tokens(args, sampling_params)
            if remaining_context is not None:
                max_turn_tokens = min(max_turn_tokens, remaining_context)
            request_stop = sampling_params.get("stop") or GLM_TOOL_STOPS
            request_stop_token_ids = sampling_params.get("stop_token_ids") or GLM_TOOL_STOP_TOKEN_IDS
            request_ignore_eos = False
            request_min_tokens = None
            if replay_turn is not None:
                if _trace_replay_force_fixed_decode():
                    replay_decode_tokens = replay_turn.generated_tokens or replay_turn.backend_generated_tokens or 0
                    max_turn_tokens = max(1, min(max_turn_tokens, max(1, replay_decode_tokens)))
                    request_stop = None
                    request_stop_token_ids = None
                    request_ignore_eos = True
                    request_min_tokens = max_turn_tokens

            request_id = llm_request_id(agent_context, turn=turn)
            _agent_trace_log(
                "model_request",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                turn=turn,
                x_request_id=request_id,
                prompt_tokens=len(current_ids),
                max_tokens=max_turn_tokens,
                min_tokens=request_min_tokens,
                ignore_eos=request_ignore_eos,
                stop_count=len(request_stop) if isinstance(request_stop, list) else (1 if request_stop else 0),
                stop_token_id_count=(
                    len(request_stop_token_ids)
                    if isinstance(request_stop_token_ids, list)
                    else (1 if request_stop_token_ids else 0)
                ),
                stop_token_ids=request_stop_token_ids,
                response_tokens_so_far=len(response_token_ids),
                remaining_context=remaining_context,
                trace_replay=trace_replay_plan is not None,
                trace_replay_prompt_tokens=replay_turn.prompt_tokens if replay_turn else None,
                trace_replay_generated_tokens=replay_turn.generated_tokens if replay_turn else None,
                trace_replay_backend_generated_tokens=replay_turn.backend_generated_tokens if replay_turn else None,
                trace_replay_force_fixed_decode=_trace_replay_force_fixed_decode() if replay_turn else False,
            )
            result = await _await_with_timeout(
                asyncio.to_thread(
                    model.complete_prompt_ids,
                    current_ids,
                    trace_messages=messages,
                    agent_context=agent_context,
                    x_request_id=request_id,
                    max_tokens=max_turn_tokens,
                    temperature=sampling_params.get("temperature", getattr(args, "rollout_temperature", 1.0)),
                    top_p=sampling_params.get("top_p", getattr(args, "rollout_top_p", 1.0)),
                    top_k=sampling_params.get("top_k", getattr(args, "rollout_top_k", None)),
                    stop=request_stop,
                    stop_token_ids=request_stop_token_ids,
                    ignore_eos=request_ignore_eos,
                    min_tokens=request_min_tokens,
                ),
                timeout=_model_call_timeout_s(),
                label=f"model completion for {instance_id} turn {turn}",
            )
            content = result["content"]
            extra = result["extra"]
            generated_ids = list(extra.get("generated_token_ids") or [])
            generated_logprobs = list(extra.get("token_logprobs") or [])
            if len(generated_logprobs) < len(generated_ids):
                generated_logprobs.extend([0.0] * (len(generated_ids) - len(generated_logprobs)))
            generated_logprobs = generated_logprobs[: len(generated_ids)]
            generated_loss_mask = [1] * len(generated_ids)
            backend_generated_len = len(generated_ids)
            if replay_turn is not None:
                target_generated_len = max(0, replay_turn.generated_tokens)
                if len(generated_ids) > target_generated_len:
                    if _trace_replay_force_fixed_decode():
                        raise RuntimeError(
                            "Trace replay fixed-decode request over-generated: "
                            f"got {len(generated_ids)} tokens, expected {target_generated_len} "
                            f"for {instance_id} turn {turn}"
                        )
                    generated_ids = generated_ids[:target_generated_len]
                    generated_logprobs = generated_logprobs[:target_generated_len]
                    generated_loss_mask = generated_loss_mask[:target_generated_len]
                    content = model.tokenizer.decode(generated_ids, skip_special_tokens=False)
                elif len(generated_ids) < target_generated_len:
                    missing = target_generated_len - len(generated_ids)
                    if _trace_replay_force_fixed_decode():
                        raise RuntimeError(
                            "Trace replay fixed-decode request under-generated: "
                            f"got {len(generated_ids)} tokens, expected {target_generated_len} "
                            f"for {instance_id} turn {turn}"
                        )
                    generated_ids.extend([pad_token_id] * missing)
                    generated_logprobs.extend([0.0] * missing)
                    generated_loss_mask.extend([0] * missing)
            finish_reason = extra.get("finish_reason") or "stop"
            stop_reason = extra.get("stop_reason")
            matched_stop_token_ids = stop_reason_token_ids(stop_reason, model.tokenizer)

            normal_text, tool_calls, needs_tool_close = _parse_tool_call_from_generated_tokens(
                model,
                content,
                generated_ids,
                matched_stop_token_ids,
            )
            if replay_turn is not None:
                finish_reason = replay_turn.finish_reason or finish_reason
                stop_reason = replay_turn.stop_reason or stop_reason
                matched_stop_token_ids = stop_reason_token_ids(stop_reason, model.tokenizer)
                if replay_turn.has_tool_call:
                    normal_text = ""
                    tool_calls = [replay_turn.tool_call()]
                    needs_tool_close = False
                else:
                    tool_calls = []
                    needs_tool_close = False
            _agent_trace_log(
                "model_response",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                turn=turn,
                x_request_id=request_id,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                matched_stop_token_ids=matched_stop_token_ids,
                generated_tokens=len(generated_ids),
                backend_generated_tokens=extra.get("backend_generated_tokens"),
                requested_max_tokens=extra.get("requested_max_tokens"),
                locally_truncated_to_max_tokens=extra.get("locally_truncated_to_max_tokens"),
                content_chars=len(content),
                contains_tool_call_text=TOOL_CALL_END in content or "<tool_call>" in content,
                parsed_tool_count=len(tool_calls),
                parsed_tool_names=[_tool_call_name(tool_call) for tool_call in tool_calls],
                needs_tool_close=needs_tool_close,
                trace_replay=trace_replay_plan is not None,
                trace_replay_backend_generated_tokens=backend_generated_len,
                trace_replay_generated_tokens=replay_turn.generated_tokens if replay_turn else None,
                trace_replay_backend_target_tokens=replay_turn.backend_generated_tokens if replay_turn else None,
                trace_replay_tool_name=replay_turn.tool_name if replay_turn else None,
                trace_replay_tool_duration_s=replay_turn.tool_duration_s if replay_turn else None,
                trace_replay_observation_tokens=replay_turn.observation_tokens if replay_turn else None,
            )
            content_for_history = _assistant_content_for_history(content, tool_calls)
            if needs_tool_close:
                stop_closer_ids = [model.tool_token_ids.tool_close]
            else:
                stop_closer_ids = []

            messages.append({"role": "assistant", "content": content_for_history})
            response_text_parts.append(content_for_history)
            response_token_ids.extend(generated_ids)
            loss_mask.extend(generated_loss_mask)
            rollout_log_probs.extend(float(x) for x in generated_logprobs)
            if stop_closer_ids:
                response_token_ids.extend(stop_closer_ids)
                loss_mask.extend([0] * len(stop_closer_ids))
                rollout_log_probs.extend([0.0] * len(stop_closer_ids))

            if _trim_response_to_context(
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                loss_mask=loss_mask,
                rollout_log_probs=rollout_log_probs,
                max_context_len=max_context,
                metadata=metadata,
            ):
                finish_reason = "length"
                break

            if finish_reason == "length":
                break
            if finish_reason == "stop" and not tool_calls:
                _agent_trace_log(
                    "model_stop_without_tool",
                    trajectory_id=_trajectory_id(agent_context),
                    instance_id=instance_id,
                    session_id=session_id,
                    turn=turn,
                    x_request_id=request_id,
                    stop_reason=stop_reason,
                    generated_tokens=len(generated_ids),
                    content_chars=len(content),
                )
                break
            if len(tool_calls) != 1:
                observation = f"Expected exactly one tool call, received {len(tool_calls)}. Please call exactly one tool."
                _agent_trace_log(
                    "format_error",
                    trajectory_id=_trajectory_id(agent_context),
                    instance_id=instance_id,
                    session_id=session_id,
                    turn=turn,
                    x_request_id=request_id,
                    parsed_tool_count=len(tool_calls),
                )
                observation_content = _format_observation(observation)
                messages.append({"role": "tool", "name": "format_error", "content": observation_content})
                observation_ids = _tool_observation_delta_ids(model, observation_content)
                response_token_ids.extend(observation_ids)
                loss_mask.extend([0] * len(observation_ids))
                rollout_log_probs.extend([0.0] * len(observation_ids))
                if _trim_response_to_context(
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    loss_mask=loss_mask,
                    rollout_log_probs=rollout_log_probs,
                    max_context_len=max_context,
                    metadata=metadata,
                ):
                    finish_reason = "length"
                    break
                continue

            tool_call = tool_calls[0]
            _agent_trace_log(
                "tool_step_request",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                turn=turn,
                x_request_id=request_id,
                tool_call_id=_tool_call_id(tool_call),
                tool_name=_tool_call_name(tool_call),
                thought_chars=len(normal_text),
            )
            step = await _await_with_timeout(
                client.step(session_id, tool_call, thought=normal_text),
                timeout=_session_call_timeout_s("step", 180.0),
                label=f"SWE-agent session step for {instance_id} turn {turn}",
            )
            observation = step.get("observation") or ""
            submitted = bool(step.get("submitted"))
            _agent_trace_log(
                "tool_step_response",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                turn=turn,
                x_request_id=request_id,
                tool_call_id=_tool_call_id(tool_call),
                tool_name=_tool_call_name(tool_call),
                observation_bytes=len(observation.encode("utf-8")),
                submitted=submitted,
                tool_error=step.get("tool_error"),
                session_dropped=step.get("session_dropped"),
                trace_replay=trace_replay_plan is not None,
                trace_replay_tool_duration_s=replay_turn.tool_duration_s if replay_turn else None,
                trace_replay_observation_tokens=replay_turn.observation_tokens if replay_turn else None,
                **_submission_trace_fields(step.get("submission") or "", step),
            )
            if step.get("session_dropped"):
                finish_reason = "session_dropped"
                metadata["retryable_session_error"] = step.get("tool_error") or "session_dropped"
                session_id = None
                break
            if submitted:
                submission = step.get("submission") or ""
                if not submission:
                    _agent_trace_log(
                        "submit_request",
                        trajectory_id=_trajectory_id(agent_context),
                        instance_id=instance_id,
                        session_id=session_id,
                        turn=turn,
                        reason="tool_reported_submitted_without_patch",
                    )
                    submit_result = await _await_with_timeout(
                        client.submit(session_id),
                        timeout=_session_call_timeout_s("submit", 300.0),
                        label=f"SWE-agent session submit for {instance_id}",
                    )
                    submission = submit_result.get("submission") or ""
                    _agent_trace_log(
                        "submit_response",
                        trajectory_id=_trajectory_id(agent_context),
                        instance_id=instance_id,
                        session_id=session_id,
                        turn=turn,
                        reason="tool_reported_submitted_without_patch",
                        **_submission_trace_fields(submission, submit_result),
                    )
                break

            observation_content = _format_observation(observation)
            messages.append(
                {
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "content": observation_content,
                }
            )
            if replay_turn is not None:
                observation_token_count = max(0, replay_turn.observation_tokens)
                _append_untrainable_tokens(
                    response_token_ids,
                    loss_mask,
                    rollout_log_probs,
                    token_id=pad_token_id,
                    count=observation_token_count,
                )
            else:
                observation_ids = _tool_observation_delta_ids(model, observation_content)
                response_token_ids.extend(observation_ids)
                loss_mask.extend([0] * len(observation_ids))
                rollout_log_probs.extend([0.0] * len(observation_ids))
            if _trim_response_to_context(
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                loss_mask=loss_mask,
                rollout_log_probs=rollout_log_probs,
                max_context_len=max_context,
                metadata=metadata,
            ):
                finish_reason = "length"
                break
            if episode_wall_timeout is not None and time.time() - started_at >= episode_wall_timeout:
                finish_reason = "length"
                break
        else:
            finish_reason = "length"

        if not submission and session_id:
            _agent_trace_log(
                "submit_request",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                reason="fallback_no_submission",
                finish_reason=finish_reason,
            )
            submit_result = await _await_with_timeout(
                client.submit(session_id),
                timeout=_session_call_timeout_s("submit", 300.0),
                label=f"SWE-agent session submit for {instance_id}",
            )
            submission = submit_result.get("submission") or ""
            _agent_trace_log(
                "submit_response",
                trajectory_id=_trajectory_id(agent_context),
                instance_id=instance_id,
                session_id=session_id,
                reason="fallback_no_submission",
                **_submission_trace_fields(submission, submit_result),
            )

    except Exception as exc:
        logger.exception("SWE-agent session rollout failed for %s", instance_id)
        _agent_trace_log(
            "rollout_error",
            trajectory_id=_trajectory_id(agent_context),
            instance_id=instance_id,
            session_id=session_id,
            finish_reason=finish_reason,
            error_type=type(exc).__name__,
            error=repr(exc),
            response_tokens=len(response_token_ids),
            trainable_tokens=sum(loss_mask),
        )
        if isinstance(exc, TimeoutError):
            metadata["rollout_timeout_error"] = repr(exc)
            if str(exc).startswith("SWE-agent session "):
                metadata["retryable_session_error"] = repr(exc)
        if response_token_ids and sum(loss_mask) > 0:
            metadata["session_error"] = repr(exc)
            metadata["session_error_traceback"] = traceback.format_exc()
            return _finalize_sweagent_session_sample(
                sample,
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                response_text_parts=response_text_parts,
                loss_mask=loss_mask,
                rollout_log_probs=rollout_log_probs,
                metadata=metadata,
                messages=messages,
                session_id=session_id,
                submission=submission,
                finish_reason=finish_reason,
                started_at=started_at,
                status=Sample.Status.FAILED,
                max_context_len=max_context,
            )
        return _mark_untrainable_sample(
            sample,
            prompt_token_ids=prompt_token_ids,
            model=model,
            metadata=metadata,
            error=exc,
            max_context_len=max_context,
        )
    finally:
        if session_id:
            try:
                _agent_trace_log(
                    "session_close_request",
                    trajectory_id=_trajectory_id(agent_context),
                    instance_id=instance_id,
                    session_id=session_id,
                )
                await _await_with_timeout(
                    client.close(session_id),
                    timeout=_session_call_timeout_s("close", 60.0),
                    label=f"SWE-agent session close for {instance_id}",
                )
                _agent_trace_log(
                    "session_close_response",
                    trajectory_id=_trajectory_id(agent_context),
                    instance_id=instance_id,
                    session_id=session_id,
                    status="closed",
                )
            except Exception as close_exc:
                _agent_trace_log(
                    "session_close_response",
                    trajectory_id=_trajectory_id(agent_context),
                    instance_id=instance_id,
                    session_id=session_id,
                    status="error",
                    error_type=type(close_exc).__name__,
                    error=repr(close_exc),
                )
                logger.exception("failed to close SWE-agent session %s", session_id)

    if not response_token_ids or sum(loss_mask) <= 0:
        return _mark_untrainable_sample(
            sample,
            prompt_token_ids=prompt_token_ids,
            model=model,
            metadata=metadata,
            error=RuntimeError("SWE-agent session rollout produced no trainable response tokens"),
            max_context_len=max_context,
        )

    metadata["instance_id"] = instance_id
    final_status = Sample.Status.FAILED if metadata.get("retryable_session_error") else None
    _agent_trace_log(
        "rollout_finished",
        trajectory_id=_trajectory_id(agent_context),
        instance_id=instance_id,
        session_id=session_id,
        finish_reason=finish_reason,
        status=final_status or (Sample.Status.TRUNCATED if finish_reason == "length" else Sample.Status.COMPLETED),
        response_tokens=len(response_token_ids),
        trainable_tokens=sum(loss_mask),
        **_submission_trace_fields(submission),
    )
    return _finalize_sweagent_session_sample(
        sample,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        response_text_parts=response_text_parts,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        metadata=metadata,
        messages=messages,
        session_id=session_id,
        submission=submission,
        finish_reason=finish_reason,
        started_at=started_at,
        status=final_status,
        max_context_len=max_context,
    )


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for SWE-bench Pro."
    if evaluation:
        raise NotImplementedError("SWE-bench Pro custom eval generation is not implemented yet.")

    mode = os.getenv("SWEPRO_AGENT_MODE", "direct_patch")
    metadata = _metadata(sample)
    messages = _build_messages(sample)
    model: DirectCompletionsModel | None = None
    max_context = _max_context_len(args)

    if mode == "sweagent_session":
        retries = _nonnegative_int_env("SWEPRO_SESSION_ROLLOUT_RETRIES", 0)
        original_sample = copy.deepcopy(sample)
        last_result = sample
        for attempt in range(retries + 1):
            attempt_sample = sample if attempt == 0 else copy.deepcopy(original_sample)
            result = await _generate_sweagent_session(args, attempt_sample, sampling_params)
            result.metadata["session_attempt"] = attempt + 1
            last_result = result
            retryable_error = _metadata(result).get("retryable_session_error")
            if not retryable_error or attempt >= retries:
                return result
            logger.warning(
                "Retrying SWE-agent session rollout for %s after %s (attempt %s/%s)",
                _metadata(result).get("instance_id") or result.label,
                retryable_error,
                attempt + 1,
                retries + 1,
            )
        return last_result

    if mode == "gold_patch":
        content = metadata.get("gold_patch", "")
        extra = {"generated_token_ids": [], "token_logprobs": [], "finish_reason": "stop", "mode": mode}
    elif mode == "empty_patch":
        content = ""
        extra = {"generated_token_ids": [], "token_logprobs": [], "finish_reason": "stop", "mode": mode}
    elif mode == "direct_patch":
        model = _get_model(args)
        instance_id = metadata.get("instance_id") or sample.label or "unknown"
        agent_context = build_agent_context_for_sample(instance_id, sample)
        metadata["dynamo_agent_context"] = agent_context
        tool_events_zmq_endpoint = derive_tool_events_zmq_endpoint(_dynamo_frontend_url(args))
        if tool_events_zmq_endpoint:
            metadata["dynamo_tool_events_zmq_endpoint"] = tool_events_zmq_endpoint
        result = await asyncio.to_thread(
            model.query,
            messages,
            agent_context=agent_context,
            x_request_id=llm_request_id(agent_context, turn=0),
            max_tokens=_turn_max_tokens(args, sampling_params),
            temperature=sampling_params.get("temperature", getattr(args, "rollout_temperature", 1.0)),
            top_p=sampling_params.get("top_p", getattr(args, "rollout_top_p", 1.0)),
            top_k=sampling_params.get("top_k", getattr(args, "rollout_top_k", None)),
            stop=sampling_params.get("stop"),
        )
        content = result["content"]
        extra = result["extra"]
    else:
        raise ValueError(f"Unsupported SWEPRO_AGENT_MODE={mode!r}")

    messages_with_response = messages + [{"role": "assistant", "content": content}]
    mask_generator = _get_mask_generator(args)
    token_ids, loss_mask = mask_generator.get_loss_mask(messages_with_response)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    context_truncated = False
    if max_context is not None and len(token_ids) > max_context:
        metadata["max_context_truncated"] = True
        metadata["max_context_truncation"] = {
            "max_context_len": max_context,
            "pre_truncate_total_length": len(token_ids),
            "post_truncate_total_length": max_context,
            "dropped_tokens": len(token_ids) - max_context,
        }
        token_ids = token_ids[:max_context]
        loss_mask = loss_mask[:max_context]
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        context_truncated = True
    if response_length <= 0:
        return _mark_untrainable_sample(
            sample,
            prompt_token_ids=_trim_prompt_for_untrainable(token_ids, max_context),
            model=model,
            metadata=metadata,
            error=RuntimeError("SWE-bench Pro rollout produced no trainable response tokens within max context"),
            max_context_len=max_context,
        )

    response_tokens = token_ids[-response_length:]
    response_loss_mask = loss_mask[-response_length:]
    rollout_log_probs = _align_logprobs(
        response_tokens,
        response_loss_mask,
        list(extra.get("generated_token_ids") or []),
        list(extra.get("token_logprobs") or []),
    )
    if len(rollout_log_probs) != response_length:
        raise ValueError(f"rollout_log_probs length mismatch: {len(rollout_log_probs)} != {response_length}")

    patch = _extract_patch(content)
    metadata.update(
        {
            "instance_id": metadata.get("instance_id") or sample.label,
            "patch": patch,
            "agent_mode": mode,
            "messages": messages_with_response,
            "model_extra": extra,
            "repo": metadata.get("repo"),
            "fail_to_pass": _list(metadata.get("fail_to_pass")),
            "pass_to_pass": _list(metadata.get("pass_to_pass")),
            "selected_test_files_to_run": _list(metadata.get("selected_test_files_to_run")),
        }
    )

    sample.tokens = token_ids
    sample.response = content
    sample.response_length = response_length
    sample.loss_mask = response_loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.metadata = metadata
    sample.status = (
        Sample.Status.TRUNCATED if context_truncated or extra.get("finish_reason") == "length" else Sample.Status.COMPLETED
    )
    return sample


async def _reward_one(args, sample: Sample) -> float:
    metadata = _metadata(sample)
    if _uses_mock_reward(metadata):
        reward = _mock_reward_value()
        result = {
            "passed": reward > 0,
            "reward": reward,
            "mock_trace_replay": True,
            "reason": "trace replay reward bypass",
        }
        metadata["eval"] = result
        metadata["raw_reward"] = reward
        _agent_trace_log(
            "mock_reward",
            trajectory_id=_trajectory_id(metadata.get("dynamo_agent_context")),
            trace_replay_source_trajectory_id=metadata.get("trace_replay_trajectory_id"),
            instance_id=metadata.get("instance_id") or sample.label,
            reward=reward,
        )
        return reward

    patch = metadata.get("patch", sample.response)
    if not patch:
        metadata["eval"] = {"passed": False, "error": "empty patch"}
        return 0.0

    try:
        from nats.aio.client import Client as NATS
    except Exception as exc:
        raise ImportError("nats-py is required for SWE-bench Pro reward evaluation") from exc

    nats_url = os.getenv("SWEPRO_NATS_URL", getattr(args, "swepro_nats_url", "nats://warnold-swepro-nats:4222"))
    timeout = float(os.getenv("SWEPRO_EVAL_TIMEOUT", getattr(args, "swepro_eval_timeout", 3600)))
    request = {
        "request_id": str(uuid.uuid4()),
        "instance_id": metadata.get("instance_id") or sample.label,
        "patch": patch,
        "repo": metadata.get("repo"),
        "fail_to_pass": _list(metadata.get("fail_to_pass")),
        "pass_to_pass": _list(metadata.get("pass_to_pass")),
        "selected_test_files_to_run": _list(metadata.get("selected_test_files_to_run")),
        "sample": metadata.get("raw_row") or metadata,
    }

    nc = NATS()
    await nc.connect(servers=[nats_url])
    try:
        msg = await nc.request("swepro.evals", json.dumps(request).encode("utf-8"), timeout=timeout)
        result = json.loads(msg.data.decode("utf-8"))
    finally:
        await nc.drain()

    metadata["eval"] = result
    return 1.0 if result.get("passed") else 0.0


async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(*[_reward_one(args, item) for item in sample])
    if not isinstance(sample, Sample):
        raise TypeError("SWE-bench Pro reward expects a slime Sample or list[Sample].")
    return await _reward_one(args, sample)
