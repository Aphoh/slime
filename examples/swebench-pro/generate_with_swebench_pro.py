"""SWE-bench Pro custom generation and reward hooks for slime."""

from __future__ import annotations

import asyncio
import copy
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

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
AGENT_TRACE_LOG_PREFIX = "SWEPRO_AGENT_TRACE"

_MODEL: DirectCompletionsModel | None = None
_MASK_GENERATOR: MultiTurnLossMaskGenerator | None = None
_SWEAGENT_TEMPLATES: dict[str, str] | None = None


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


def _turn_max_tokens(args, sampling_params: dict[str, Any] | None = None) -> int:
    rollout_max = int(getattr(args, "rollout_max_response_len", 4096))
    default_turn_max = min(8192, rollout_max)
    turn_max = _positive_int_env("SWEPRO_TURN_MAX_TOKENS", default_turn_max)
    requested = rollout_max
    if sampling_params is not None:
        requested = int(sampling_params.get("max_new_tokens") or rollout_max)
    return max(1, min(turn_max, requested, rollout_max))


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
        )
        _MODEL = DirectCompletionsModel(config)
    return _MODEL


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


def _tool_observation_delta_ids(model: DirectCompletionsModel, observation: str) -> list[int]:
    return encode_qwen_tool_observation_delta(model.tokenizer, observation)


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
    return max_context_len - len(prompt_token_ids) - len(response_token_ids)


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
    client = SweAgentSessionClient.from_env(args)
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

        for turn in turn_iter:
            if episode_wall_timeout is not None and time.time() - started_at >= episode_wall_timeout:
                finish_reason = "length"
                break

            current_ids = prompt_token_ids + response_token_ids
            remaining_context = _remaining_context(prompt_token_ids, response_token_ids, max_context)
            if remaining_context is not None and remaining_context <= 0:
                finish_reason = "length"
                break
            max_turn_tokens = _turn_max_tokens(args, sampling_params)
            if remaining_context is not None:
                max_turn_tokens = min(max_turn_tokens, remaining_context)

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
                response_tokens_so_far=len(response_token_ids),
                remaining_context=remaining_context,
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
                    stop=sampling_params.get("stop") or GLM_TOOL_STOPS,
                    stop_token_ids=sampling_params.get("stop_token_ids") or GLM_TOOL_STOP_TOKEN_IDS,
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
            finish_reason = extra.get("finish_reason") or "stop"
            stop_reason = extra.get("stop_reason")
            matched_stop_token_ids = stop_reason_token_ids(stop_reason, model.tokenizer)

            content_for_history = content
            normal_text, tool_calls, needs_tool_close = _parse_tool_call_from_generated_tokens(
                model,
                content,
                generated_ids,
                matched_stop_token_ids,
            )
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
            )
            if tool_calls and TOOL_CALL_END not in content_for_history:
                content_for_history = content_for_history + TOOL_CALL_END
            if needs_tool_close:
                stop_closer_ids = [model.tool_token_ids.tool_close]
            else:
                stop_closer_ids = []

            messages.append({"role": "assistant", "content": content_for_history})
            response_text_parts.append(content)
            response_token_ids.extend(generated_ids)
            loss_mask.extend([1] * len(generated_ids))
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
                        patch_chars=len(submission),
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
                patch_chars=len(submission),
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
        patch_chars=len(submission),
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
