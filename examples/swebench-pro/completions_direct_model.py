"""Direct token-ID Dynamo adapter for SWE-bench Pro rollouts.

This module intentionally does not import SWE-agent or LiteLLM. It formats chat
prompts locally with the HF tokenizer and sends exact token IDs through Dynamo native SGLang `/generate`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from slime.rollout.dynamo_client import DynamoGeneration, build_dynamo_generate_payload

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - parser-only local tooling may not have transformers.
    AutoTokenizer = None  # type: ignore


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default

logger = logging.getLogger(__name__)
COMPLETIONS_DEBUG_LOG_PREFIX = "SWEPRO_COMPLETIONS_DEBUG"


def _env_flag(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_log_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_log_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_log_value(item) for key, item in value.items()}
    return str(value)


def _round_s(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _payload_debug_fields(payload: dict[str, Any]) -> dict[str, Any]:
    sampling_params = payload.get("sampling_params") or {}
    stop = sampling_params.get("stop")
    stop_list = stop if isinstance(stop, list) else ([stop] if stop else [])
    return {
        "api": "sglang_native",
        "prompt_tokens": len(payload.get("input_ids") or []),
        "max_new_tokens": sampling_params.get("max_new_tokens"),
        "min_new_tokens": sampling_params.get("min_new_tokens"),
        "ignore_eos": sampling_params.get("ignore_eos"),
        "temperature": sampling_params.get("temperature"),
        "top_p": sampling_params.get("top_p"),
        "top_k": sampling_params.get("top_k"),
        "stream": payload.get("stream"),
        "stop_count": len(stop_list),
        "stop": stop_list,
        "return_logprob": payload.get("return_logprob"),
        "return_routed_experts": payload.get("return_routed_experts"),
    }


def _completion_debug_log(event: str, **fields: Any) -> None:
    if not _env_flag("SWEPRO_COMPLETIONS_DEBUG"):
        return
    payload = {"event": event}
    payload.update({key: _json_log_value(value) for key, value in fields.items() if value is not None})
    logger.info("%s %s", COMPLETIONS_DEBUG_LOG_PREFIX, json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    return ""

GLM_TOOL_CALL_TOKEN_ID = 154843
GLM_TOOL_CLOSE_TOKEN_ID = 154844
GLM_TOOL_RESPONSE_START_TOKEN_ID = 154845
GLM_TOOL_RESPONSE_END_TOKEN_ID = 154846
GLM_ASSISTANT_TOKEN_ID = 154828
GLM_OBSERVATION_TOKEN_ID = 154829
GLM_THINK_START_TOKEN_ID = 154841
GLM_EOS_TOKEN_ID = 154820

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
EOS_TEXT = "<|endoftext|>"

GLM_TOOL_STOPS = [
    TOOL_CALL_END,
    EOS_TEXT,
]

GLM_TOOL_STOP_TOKEN_IDS = [
    GLM_TOOL_CLOSE_TOKEN_ID,
    GLM_EOS_TOKEN_ID,
]


def token_ids_for_text(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None:
        return []
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return list(encoded.get("input_ids") or [])
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    return []


def single_token_id_for_text(tokenizer: Any, text: str, fallback: int) -> int:
    token_ids = token_ids_for_text(tokenizer, text)
    return token_ids[0] if len(token_ids) == 1 else fallback


@dataclass(frozen=True)
class QwenToolTokenIds:
    tool_call: int
    tool_close: int
    eos: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> QwenToolTokenIds:
        return cls(
            tool_call=single_token_id_for_text(tokenizer, TOOL_CALL_START, GLM_TOOL_CALL_TOKEN_ID),
            tool_close=single_token_id_for_text(tokenizer, TOOL_CALL_END, GLM_TOOL_CLOSE_TOKEN_ID),
            eos=single_token_id_for_text(tokenizer, EOS_TEXT, GLM_EOS_TOKEN_ID),
        )


def _tool_token_ids(tokenizer: Any | None = None) -> QwenToolTokenIds:
    return (
        QwenToolTokenIds.from_tokenizer(tokenizer)
        if tokenizer is not None
        else QwenToolTokenIds(
            tool_call=GLM_TOOL_CALL_TOKEN_ID,
            tool_close=GLM_TOOL_CLOSE_TOKEN_ID,
            eos=GLM_EOS_TOKEN_ID,
        )
    )


@dataclass
class DirectCompletionsConfig:
    base_url: str
    tokenizer_path: str
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    timeout: float = 600.0
    retries: int = 5
    return_routed_experts: bool = False

    @classmethod
    def from_env(cls) -> DirectCompletionsConfig:
        base_url = _env("SWEPRO_DYNAMO_FRONTEND_URL", _env("DYNAMO_FRONTEND_URL"))
        tokenizer_path = _env("SWEPRO_TOKENIZER_PATH", _env("HF_CHECKPOINT", _env("MODEL_PATH")))
        if not base_url:
            raise ValueError("Set SWEPRO_DYNAMO_FRONTEND_URL or DYNAMO_FRONTEND_URL")
        if not tokenizer_path:
            raise ValueError("Set SWEPRO_TOKENIZER_PATH, HF_CHECKPOINT, or MODEL_PATH")
        return cls(
            base_url=base_url.rstrip("/"),
            tokenizer_path=tokenizer_path,
            max_tokens=int(_env("SWEPRO_MAX_TOKENS", "4096") or "4096"),
            temperature=float(_env("SWEPRO_TEMPERATURE", "1.0") or "1.0"),
            top_p=float(_env("SWEPRO_TOP_P", "1.0") or "1.0"),
            top_k=int(value) if (value := _env("SWEPRO_TOP_K")) else None,
            timeout=float(_env("SWEPRO_REQUEST_TIMEOUT", "600") or "600"),
            retries=int(_env("SWEPRO_REQUEST_RETRIES", "5") or "5"),
        )


class DirectCompletionsModel:
    def __init__(self, config: DirectCompletionsConfig | None = None):
        if AutoTokenizer is None:
            raise ImportError("transformers is required to instantiate DirectCompletionsModel")
        self.config = config or DirectCompletionsConfig.from_env()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
        self.tool_token_ids = QwenToolTokenIds.from_tokenizer(self.tokenizer)

    def get_template_vars(self) -> dict[str, Any]:
        return {}

    def render_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
        rendered = self.render_prompt(messages, tools=tools)
        return self.tokenizer(rendered, add_special_tokens=False)["input_ids"]

    def _build_payload(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        prompt_ids = self.encode_prompt(messages, tools=kwargs.get("tools"))
        return self._build_payload_from_ids(prompt_ids, **kwargs)

    def _build_payload_from_ids(self, prompt_ids: list[int], **kwargs: Any) -> dict[str, Any]:
        sampling_params = {
            "max_new_tokens": int(kwargs.get("max_tokens", self.config.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.config.temperature)),
            "top_p": float(kwargs.get("top_p", self.config.top_p)),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "stop": kwargs.get("stop"),
            "stop_token_ids": kwargs.get("stop_token_ids"),
            "min_new_tokens": kwargs.get("min_tokens"),
            "ignore_eos": kwargs.get("ignore_eos"),
            "sampling_seed": kwargs.get("seed"),
            "skip_special_tokens": kwargs.get("skip_special_tokens"),
            "no_stop_trim": kwargs.get("no_stop_trim"),
            "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens"),
        }
        return build_dynamo_generate_payload(
            prompt_token_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=str(kwargs.get("request_id") or "swepro"),
            return_routed_experts=self.config.return_routed_experts,
        )

    def complete_prompt_ids(
        self,
        prompt_ids: list[int],
        *,
        x_request_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = self._build_payload_from_ids(prompt_ids, request_id=x_request_id, **kwargs)
        return self._post_payload(payload, x_request_id=x_request_id)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        payload = self._build_payload(messages, request_id=kwargs.get("x_request_id"), **kwargs)
        return self._post_payload(payload, x_request_id=kwargs.get("x_request_id"))

    def _post_payload(
        self,
        payload: dict[str, Any],
        *,
        x_request_id: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}/generate"
        last_error: Exception | None = None
        generation: DynamoGeneration | None = None
        raw_events: list[dict[str, Any]] = []
        success_attempt: int | None = None
        success_elapsed_s: float | None = None
        response_status_code: int | None = None

        for attempt in range(self.config.retries):
            attempt_started_at = time.monotonic()
            attempt_status_code: int | None = None
            attempt_generation = DynamoGeneration()
            attempt_events: list[dict[str, Any]] = []
            request_id = f"{x_request_id}:try:{attempt}" if x_request_id else f"swepro:try:{attempt}"
            attempt_payload = dict(payload)
            attempt_payload["rid"] = request_id
            try:
                _completion_debug_log(
                    "completion_request",
                    url=url,
                    attempt=attempt,
                    retries=self.config.retries,
                    x_request_id=x_request_id,
                    request_id=request_id,
                    timeout_s=self.config.timeout,
                    **_payload_debug_fields(attempt_payload),
                )
                request = urllib.request.Request(
                    url,
                    data=json.dumps(attempt_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "x-request-id": request_id},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    attempt_status_code = getattr(response, "status", None)
                    response_status_code = attempt_status_code
                    for raw_line in response:
                        if isinstance(raw_line, bytes):
                            raw_line = raw_line.decode("utf-8")
                        if not raw_line or not raw_line.startswith("data:"):
                            continue
                        data_str = raw_line[len("data:") :].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        event = json.loads(data_str)
                        attempt_generation.consume_sse(event)
                        attempt_events.append(event)
                if not attempt_generation.terminal_event_received:
                    raise RuntimeError("Dynamo native Generate stream ended without a finish reason")
                generation = attempt_generation
                raw_events = attempt_events
                success_attempt = attempt
                success_elapsed_s = time.monotonic() - attempt_started_at
                break
            except Exception as exc:
                last_error = exc
                _completion_debug_log(
                    "completion_request_error",
                    url=url,
                    attempt=attempt,
                    retries=self.config.retries,
                    x_request_id=x_request_id,
                    request_id=request_id,
                    status_code=attempt_status_code,
                    elapsed_s=_round_s(time.monotonic() - attempt_started_at),
                    error_type=type(exc).__name__,
                    error=repr(exc),
                    will_retry=attempt + 1 < self.config.retries and not attempt_generation.token_ids,
                    **_payload_debug_fields(attempt_payload),
                )
                if attempt_generation.token_ids or attempt + 1 >= self.config.retries:
                    raise
                time.sleep(min(2**attempt, 30))
        if generation is None:
            raise RuntimeError(f"completion request failed: {last_error}")

        generated_token_ids = generation.token_ids
        token_logprobs = generation.token_logprobs
        if len(token_logprobs) != len(generated_token_ids):
            raise RuntimeError("Dynamo native Generate returned token IDs without one selected-token logprob per token")
        content = _decode_token_ids(self.tokenizer, generated_token_ids)
        requested_max_tokens = int(payload["sampling_params"]["max_new_tokens"])
        usage = dict(generation.usage)
        usage.setdefault("prompt_tokens", len(payload["input_ids"]))
        usage.setdefault("completion_tokens", len(generated_token_ids))
        usage.setdefault("total_tokens", int(usage["prompt_tokens"]) + int(usage["completion_tokens"]))

        extra = {
            "response": {"events": raw_events},
            "response_tool_calls": [],
            "prompt_token_ids": list(payload["input_ids"]),
            "generated_token_ids": generated_token_ids,
            "token_logprobs": token_logprobs,
            "finish_reason": generation.finish_reason,
            "stop_reason": generation.stop_reason,
            "requested_max_tokens": requested_max_tokens,
            "backend_generated_tokens": len(generated_token_ids),
            "usage_completion_tokens": usage.get("completion_tokens"),
            "parsed_generated_token_ids": len(generated_token_ids),
            "raw_token_logprob_count": len(token_logprobs),
            "generated_token_source": "output_ids",
            "dynamo_metadata": generation.metadata_sequence,
        }
        _completion_debug_log(
            "completion_response",
            url=url,
            attempt=success_attempt,
            x_request_id=x_request_id,
            status_code=response_status_code,
            elapsed_s=_round_s(success_elapsed_s),
            requested_max_tokens=requested_max_tokens,
            finish_reason=generation.finish_reason,
            stop_reason=generation.stop_reason,
            usage_prompt_tokens=usage.get("prompt_tokens"),
            usage_completion_tokens=usage.get("completion_tokens"),
            usage_total_tokens=usage.get("total_tokens"),
            generated_tokens=len(generated_token_ids),
            token_logprobs=len(token_logprobs),
            response_text_chars=len(content),
            **_payload_debug_fields(payload),
        )
        return {"content": content, "message": content, "extra": extra}


def query(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    return DirectCompletionsModel().query(messages, **kwargs)


def stop_reason_token_ids(stop_reason: Any, tokenizer: Any | None = None) -> list[int]:
    """Return token IDs from Dynamo native Generate stop_reason values, without tokenizing."""

    ids = _tool_token_ids(tokenizer)
    if stop_reason is None or isinstance(stop_reason, bool):
        return []
    if isinstance(stop_reason, int):
        return [stop_reason]
    if isinstance(stop_reason, str):
        stop_reason = stop_reason.strip()
        if stop_reason == TOOL_CALL_END:
            return [ids.tool_close]
        if stop_reason == EOS_TEXT:
            return [ids.eos]
        if stop_reason.startswith("token_id:"):
            stop_reason = stop_reason[len("token_id:") :]
        try:
            return [int(stop_reason)]
        except ValueError:
            return []
    if isinstance(stop_reason, list):
        token_ids: list[int] = []
        for item in stop_reason:
            token_ids.extend(stop_reason_token_ids(item, tokenizer))
        return token_ids
    return []


def _rfind_token(token_ids: list[int], token_id: int) -> int:
    for idx in range(len(token_ids) - 1, -1, -1):
        if token_ids[idx] == token_id:
            return idx
    return -1


def parse_glm_tool_call_from_completion(
    tokenizer: Any,
    content: str,
    generated_ids: list[int],
    matched_stop_token_ids: list[int],
) -> tuple[str, list[dict[str, Any]], bool]:
    """Parse a GLM tool call only when the matched stop was `</tool_call>`."""

    ids = _tool_token_ids(tokenizer)
    if ids.tool_close not in matched_stop_token_ids and GLM_TOOL_CLOSE_TOKEN_ID not in matched_stop_token_ids:
        return content, [], False

    tool_start = _rfind_token(generated_ids, ids.tool_call)
    if tool_start < 0 and ids.tool_call != GLM_TOOL_CALL_TOKEN_ID:
        tool_start = _rfind_token(generated_ids, GLM_TOOL_CALL_TOKEN_ID)
    if tool_start < 0:
        if TOOL_CALL_START not in content:
            return content, [], False
        normal_text, tool_calls = parse_glm_tool_calls(content)
        return normal_text, tool_calls, bool(tool_calls and TOOL_CALL_END not in content)

    tool_text = tokenizer.decode(generated_ids[tool_start:], skip_special_tokens=False)
    has_tool_close = ids.tool_close in generated_ids[tool_start:] or GLM_TOOL_CLOSE_TOKEN_ID in generated_ids[tool_start:]
    if not has_tool_close:
        tool_text = tool_text + TOOL_CALL_END

    marker_idx = content.rfind(TOOL_CALL_START)
    if marker_idx >= 0:
        normal_text = content[:marker_idx]
    else:
        normal_text = tokenizer.decode(generated_ids[:tool_start], skip_special_tokens=False)

    _, tool_calls = parse_glm_tool_calls(tool_text)
    return normal_text, tool_calls, not has_tool_close


def encode_qwen_tool_observation_delta(tokenizer: Any, observation: str) -> list[int]:
    """Encode the Qwen chat-template continuation after an assistant tool call."""

    return token_ids_for_text(
        tokenizer,
        (f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"),
    )


def _extract_tag_value(text: str, tag: str) -> str | None:
    pattern = rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def _parse_json_tool_call(body: str) -> tuple[str, dict[str, Any]] | None:
    try:
        data = json.loads(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("tool_name") or data.get("function")
    arguments = data.get("arguments") or data.get("parameters") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"command": arguments}
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments


def _parse_glm_arg_tags(body: str) -> tuple[str, dict[str, Any]] | None:
    name = _extract_tag_value(body, "tool_name") or _extract_tag_value(body, "name")
    if not name:
        function_match = re.search(r"<function=([A-Za-z_][A-Za-z0-9_-]*)>", body)
        if function_match:
            name = function_match.group(1)
    if not name:
        leading = re.split(r"<arg_key>|<parameter=", body, maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", leading):
            name = leading
    if not name:
        return None

    args: dict[str, Any] = {}
    for key, value in re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", body, re.DOTALL):
        key = key.strip()
        raw_value = value.strip()
        try:
            args[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            args[key] = raw_value

    for key, value in re.findall(r"<parameter=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
        raw_value = value.strip()
        try:
            args[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            args[key] = raw_value

    return name.strip(), args


def parse_glm_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse GLM tool-call text into SWE-agent/OpenAI-compatible tool calls."""
    parse_text = text
    if "<tool_call>" in parse_text and "</tool_call>" not in parse_text:
        parse_text = parse_text + "</tool_call>"

    calls: list[dict[str, Any]] = []
    normal_parts: list[str] = []
    cursor = 0
    for idx, match in enumerate(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", parse_text, re.DOTALL)):
        normal_parts.append(parse_text[cursor : match.start()])
        cursor = match.end()
        body = match.group(1).strip()
        parsed = _parse_json_tool_call(body) or _parse_glm_arg_tags(body)
        if parsed is None:
            continue
        name, arguments = parsed
        calls.append(
            {
                "id": f"call_{idx}_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
    normal_parts.append(parse_text[cursor:])
    normal_text = "".join(normal_parts).strip()
    return normal_text, calls
