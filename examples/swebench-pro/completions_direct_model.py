"""Direct token-ID Dynamo adapter for SWE-bench Pro rollouts.

This module intentionally does not import SWE-agent or LiteLLM. It formats chat
prompts locally with the HF tokenizer and sends exact token IDs through either
`/v1/completions` or stateful `/v1/responses`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from slime.rollout.dynamo_client import (
    build_dynamo_payload,
    build_metadata_upload_url,
    read_uploaded_metadata as _read_uploaded_metadata,
)

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
    nvext = payload.get("nvext") if isinstance(payload.get("nvext"), dict) else {}
    prompt = payload.get("prompt") or nvext.get("token_data") or []
    stop = payload.get("stop")
    stop_list = stop if isinstance(stop, list) else ([stop] if stop else [])
    agent_context = nvext.get("agent_context") if isinstance(nvext, dict) else None
    metadata_upload = nvext.get("metadata_upload") if isinstance(nvext, dict) else None
    return {
        "api_mode": "responses" if "input" in payload else "completions",
        "model": payload.get("model"),
        "prompt_tokens": len(prompt),
        "max_tokens": payload.get("max_tokens", payload.get("max_output_tokens")),
        "min_tokens": payload.get("min_tokens"),
        "ignore_eos": payload.get("ignore_eos"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "stream": payload.get("stream"),
        "detokenize": payload.get("detokenize"),
        "return_tokens_as_token_ids": payload.get("return_tokens_as_token_ids"),
        "logprobs": payload.get("logprobs"),
        "stop_count": len(stop_list),
        "stop": stop_list,
        "nvext_extra_fields": nvext.get("extra_fields") if isinstance(nvext, dict) else None,
        "metadata_upload_url": metadata_upload.get("url") if isinstance(metadata_upload, dict) else None,
        "metadata_upload_format": metadata_upload.get("format") if isinstance(metadata_upload, dict) else None,
        "previous_response_id": payload.get("previous_response_id"),
        "agent_trajectory_id": agent_context.get("trajectory_id") if isinstance(agent_context, dict) else None,
        "agent_session_id": agent_context.get("session_id") if isinstance(agent_context, dict) else None,
    }


def _completion_debug_log(event: str, **fields: Any) -> None:
    if not _env_flag("SWEPRO_COMPLETIONS_DEBUG"):
        return
    payload = {"event": event}
    payload.update({key: _json_log_value(value) for key, value in fields.items() if value is not None})
    logger.info("%s %s", COMPLETIONS_DEBUG_LOG_PREFIX, json.dumps(payload, sort_keys=True, ensure_ascii=False))


def token_ids_from_logprob_tokens(tokens: list[Any], tokenizer) -> list[int]:
    ids: list[int] = []
    for token in tokens or []:
        if isinstance(token, str) and token.startswith("token_id:"):
            ids.append(int(token[len("token_id:") :]))
        elif isinstance(token, int):
            ids.append(token)
        elif isinstance(token, str):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            ids.extend(encoded)
    return ids


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


def glm_stop_strings_from_token_ids(token_ids: list[int], tokenizer: Any | None = None) -> list[str]:
    """Return Qwen/GLM stop strings for tool-close and EOS token IDs."""
    ids = _tool_token_ids(tokenizer)
    stops: list[str] = []
    for token_id in token_ids:
        if token_id in {GLM_TOOL_CLOSE_TOKEN_ID, ids.tool_close}:
            stops.append(TOOL_CALL_END)
        elif token_id in {GLM_EOS_TOKEN_ID, ids.eos}:
            stops.append(EOS_TEXT)
        else:
            raise ValueError(f"/v1/completions does not accept stop_token_ids; no GLM stop string mapping for token_id:{token_id}")
    return stops


def _response_output_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _response_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        calls.append(
            {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
        )
    return calls


def _response_output_logprobs(data: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict) or content.get("type") != "output_text":
                continue
            values.extend(float(entry["logprob"]) for entry in content.get("logprobs") or [])
    return values


@dataclass
class DirectCompletionsConfig:
    base_url: str
    tokenizer_path: str
    api_mode: str = "completions"
    model: str = "default"
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    timeout: float = 600.0
    retries: int = 5
    return_logprobs: bool = True
    detokenize: bool = True
    metadata_upload_url: str | None = None
    metadata_upload_format: str = "msgpack"

    @classmethod
    def from_env(cls) -> DirectCompletionsConfig:
        base_url = _env("SWEPRO_DYNAMO_FRONTEND_URL", _env("DYNAMO_FRONTEND_URL", _env("OPENAI_BASE_URL")))
        tokenizer_path = _env("SWEPRO_TOKENIZER_PATH", _env("HF_CHECKPOINT", _env("MODEL_PATH")))
        if not base_url:
            raise ValueError("Set SWEPRO_DYNAMO_FRONTEND_URL or DYNAMO_FRONTEND_URL")
        if not tokenizer_path:
            raise ValueError("Set SWEPRO_TOKENIZER_PATH, HF_CHECKPOINT, or MODEL_PATH")
        return cls(
            base_url=base_url.rstrip("/"),
            tokenizer_path=tokenizer_path,
            api_mode=(_env("DYNAMO_API_MODE", _env("SWEPRO_DYNAMO_API_MODE", "completions")) or "completions").strip().lower(),
            model=_env("SWEPRO_MODEL", tokenizer_path) or "default",
            max_tokens=int(_env("SWEPRO_MAX_TOKENS", "4096") or "4096"),
            temperature=float(_env("SWEPRO_TEMPERATURE", "1.0") or "1.0"),
            top_p=float(_env("SWEPRO_TOP_P", "1.0") or "1.0"),
            top_k=int(value) if (value := _env("SWEPRO_TOP_K")) else None,
            timeout=float(_env("SWEPRO_REQUEST_TIMEOUT", "600") or "600"),
            retries=int(_env("SWEPRO_REQUEST_RETRIES", "5") or "5"),
            return_logprobs=_env_flag("SWEPRO_COMPLETIONS_RETURN_LOGPROBS", True),
            detokenize=_env_flag("SWEPRO_COMPLETIONS_DETOKENIZE", True),
            metadata_upload_url=_env(
                "DYNAMO_METADATA_UPLOAD_URL",
                _env("SWEPRO_DYNAMO_METADATA_UPLOAD_URL"),
            ),
            metadata_upload_format=(
                _env(
                    "DYNAMO_METADATA_UPLOAD_FORMAT",
                    _env("SWEPRO_DYNAMO_METADATA_UPLOAD_FORMAT", "msgpack"),
                )
                or "msgpack"
            ),
        )


class DirectCompletionsModel:
    def __init__(self, config: DirectCompletionsConfig | None = None):
        if AutoTokenizer is None:
            raise ImportError("transformers is required to instantiate DirectCompletionsModel")
        self.config = config or DirectCompletionsConfig.from_env()
        if self.config.api_mode not in {"completions", "responses"}:
            raise ValueError("SWEPRO_DYNAMO_API_MODE must be 'completions' or 'responses'")
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

    def _build_payload(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        tools = kwargs.get("tools")
        rendered = self.render_prompt(messages, tools=tools)
        prompt_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if self.config.api_mode == "responses":
            kwargs.setdefault("response_input", messages)
        return self._build_payload_from_ids(prompt_ids, **kwargs)

    def _build_payload_from_ids(self, prompt_ids: list[int], **kwargs) -> dict[str, Any]:
        stop_token_ids = kwargs.get("stop_token_ids")
        stop = kwargs.get("stop")
        if stop_token_ids:
            mapped_stops = glm_stop_strings_from_token_ids(
                [int(token_id) for token_id in stop_token_ids],
                getattr(self, "tokenizer", None),
            )
            stop_list = list(stop or []) if isinstance(stop, list) else ([stop] if stop else [])
            for mapped_stop in mapped_stops:
                if mapped_stop not in stop_list:
                    stop_list.append(mapped_stop)
            stop = stop_list

        response_input = kwargs.get("response_input")
        if self.config.api_mode == "responses" and not response_input:
            raise ValueError("stateful /v1/responses requests require response_input items")
        payload = build_dynamo_payload(
            api_mode=self.config.api_mode,
            model=kwargs.get("model", self.config.model),
            prompt_token_ids=prompt_ids,
            response_input=response_input,
            previous_response_id=kwargs.get("previous_response_id"),
            store=self.config.api_mode == "responses",
            max_tokens=int(kwargs.get("max_tokens", self.config.max_tokens)),
            temperature=float(kwargs.get("temperature", self.config.temperature)),
            top_p=float(kwargs.get("top_p", self.config.top_p)),
            top_k=kwargs.get("top_k", self.config.top_k),
            stop=stop,
            min_tokens=kwargs.get("min_tokens"),
            ignore_eos=kwargs.get("ignore_eos"),
            seed=kwargs.get("seed"),
            skip_special_tokens=kwargs.get("skip_special_tokens"),
            no_stop_trim=kwargs.get("no_stop_trim"),
            spaces_between_special_tokens=kwargs.get("spaces_between_special_tokens"),
            stream=False,
            return_logprobs=self.config.return_logprobs,
            agent_context=kwargs.get("agent_context"),
        )
        if self.config.api_mode == "completions" and not self.config.detokenize:
            payload["detokenize"] = False
        return payload

    def complete_prompt_ids(
        self,
        prompt_ids: list[int],
        *,
        trace_messages: list[dict[str, Any]] | None = None,
        response_input: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        x_request_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if self.config.api_mode == "responses":
            kwargs["response_input"] = response_input or trace_messages
            kwargs["previous_response_id"] = previous_response_id
        payload = self._build_payload_from_ids(prompt_ids, **kwargs)
        return self._post_payload(payload, trace_messages or [], x_request_id=x_request_id)

    def delete_response(self, response_id: str) -> None:
        response = requests.delete(f"{self.config.base_url}/v1/responses/{response_id}", timeout=self.config.timeout)
        if response.status_code not in {200, 404}:
            response.raise_for_status()

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        payload = self._build_payload(messages, **kwargs)
        return self._post_payload(payload, messages, x_request_id=kwargs.get("x_request_id"))

    def _post_payload(
        self,
        payload: dict[str, Any],
        _messages: list[dict[str, Any]],
        *,
        x_request_id: str | None = None,
    ) -> dict[str, Any]:
        endpoint = "responses" if self.config.api_mode == "responses" else "completions"
        url = f"{self.config.base_url}/v1/{endpoint}"
        last_error: Exception | None = None
        success_attempt: int | None = None
        success_elapsed_s: float | None = None
        success_metadata_upload_url: str | None = None
        response_status_code: int | None = None
        for attempt in range(self.config.retries):
            attempt_started_at = time.monotonic()
            attempt_status_code: int | None = None
            try:
                request_id = f"{x_request_id}:try:{attempt}" if x_request_id else None
                metadata_upload_url = None
                if self.config.metadata_upload_url:
                    metadata_upload_url = build_metadata_upload_url(
                        self.config.metadata_upload_url,
                        request_id,
                    )
                    payload["nvext"]["metadata_upload"] = {
                        "url": metadata_upload_url,
                        "format": self.config.metadata_upload_format,
                    }
                headers = {"x-request-id": request_id} if request_id else None
                _completion_debug_log(
                    "completion_request",
                    url=url,
                    attempt=attempt,
                    retries=self.config.retries,
                    x_request_id=x_request_id,
                    request_id=request_id,
                    timeout_s=self.config.timeout,
                    **_payload_debug_fields(payload),
                )
                response = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout)
                attempt_status_code = getattr(response, "status_code", None)
                response_status_code = attempt_status_code
                response.raise_for_status()
                data = response.json()
                success_attempt = attempt
                success_elapsed_s = time.monotonic() - attempt_started_at
                success_metadata_upload_url = metadata_upload_url
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
                    will_retry=attempt + 1 < self.config.retries,
                    **_payload_debug_fields(payload),
                )
                if attempt + 1 >= self.config.retries:
                    raise
                time.sleep(min(2**attempt, 30))
        else:
            raise RuntimeError(f"completion request failed: {last_error}")

        response_nvext = {}
        if isinstance(data.get("nvext"), dict):
            response_nvext.update(data["nvext"])
        response_tool_calls: list[dict[str, Any]] = []
        prompt_token_ids = list(payload.get("prompt") or payload["nvext"].get("token_data") or [])
        requested_max_tokens = int(payload.get("max_tokens", payload.get("max_output_tokens")))

        if self.config.api_mode == "responses":
            completion_token_ids = response_nvext.get("completion_token_ids")
            generated_token_ids = [int(token_id) for token_id in completion_token_ids or []]
            generated_token_source = "completion_token_ids" if generated_token_ids else None
            token_logprobs = [float(value) for value in response_nvext.get("completion_token_logprobs") or []]
            if not token_logprobs:
                token_logprobs = _response_output_logprobs(data)
            content = _response_output_text(data)
            response_tool_calls = _response_tool_calls(data)
            response_status = data.get("status")
            incomplete_reason = (data.get("incomplete_details") or {}).get("reason")
            if response_status == "incomplete":
                finish_reason = "content_filter" if incomplete_reason == "content_filter" else "length"
            elif response_status == "completed":
                finish_reason = "stop"
            else:
                finish_reason = response_status
            stop_reason = response_nvext.get("stop_reason") or incomplete_reason
            response_usage = data.get("usage") or {}
            usage = {
                "prompt_tokens": response_usage.get("input_tokens"),
                "completion_tokens": response_usage.get("output_tokens"),
                "total_tokens": response_usage.get("total_tokens"),
            }
        else:
            choice = data["choices"][0]
            if isinstance(choice.get("nvext"), dict):
                response_nvext.update(choice["nvext"])
            logprobs = choice.get("logprobs") or {}
            completion_token_ids = response_nvext.get("completion_token_ids")
            generated_token_ids = [int(token_id) for token_id in completion_token_ids or []]
            generated_token_source = "completion_token_ids" if generated_token_ids else None
            if not generated_token_ids:
                generated_token_ids = token_ids_from_logprob_tokens(logprobs.get("tokens") or [], self.tokenizer)
                generated_token_source = "logprobs_tokens" if generated_token_ids else None
            token_logprobs = list(logprobs.get("token_logprobs") or [])
            content = choice.get("text", "")
            finish_reason = choice.get("finish_reason")
            stop_reason = response_nvext.get("stop_reason", choice.get("stop_reason"))
            usage = data.get("usage") or {}

        uploaded_metadata = None
        if success_metadata_upload_url:
            uploaded_metadata = _read_uploaded_metadata(
                success_metadata_upload_url,
                self.config.metadata_upload_format,
            )
            uploaded_logprobs = uploaded_metadata.get("output_token_logprobs") or []
            uploaded_token_ids = [int(entry[1]) for entry in uploaded_logprobs]
            if generated_token_ids and uploaded_token_ids:
                streamed_token_count = len(generated_token_ids)
                if uploaded_token_ids[:streamed_token_count] != generated_token_ids:
                    raise RuntimeError("Dynamo metadata output_token_logprobs token IDs do not match completion_token_ids")
                uploaded_logprobs = uploaded_logprobs[:streamed_token_count]
                uploaded_token_ids = uploaded_token_ids[:streamed_token_count]
            if uploaded_token_ids:
                if not generated_token_ids:
                    generated_token_ids = uploaded_token_ids
                    generated_token_source = "metadata_upload"
                token_logprobs = [float(entry[0]) for entry in uploaded_logprobs]
            uploaded_finish_reason = uploaded_metadata.get("finish_reason")
            if isinstance(uploaded_finish_reason, dict):
                uploaded_finish_reason = uploaded_finish_reason.get("type")
            if uploaded_finish_reason and not finish_reason:
                finish_reason = str(uploaded_finish_reason)

        if not generated_token_ids:
            generated_token_ids = token_ids_for_text(self.tokenizer, content)
            generated_token_source = "text" if generated_token_ids else None
            if not self.config.return_logprobs and payload.get("ignore_eos") and payload.get("min_tokens") == requested_max_tokens:
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = self.tool_token_ids.eos
                if len(generated_token_ids) < requested_max_tokens:
                    generated_token_ids.extend([pad_token_id] * (requested_max_tokens - len(generated_token_ids)))
                elif len(generated_token_ids) > requested_max_tokens:
                    generated_token_ids = generated_token_ids[:requested_max_tokens]
        if generated_token_ids and self.config.api_mode == "responses":
            content = _decode_token_ids(self.tokenizer, generated_token_ids)
            if finish_reason == "stop" and stop_reason is None and len(generated_token_ids) >= requested_max_tokens:
                finish_reason = "length"

        raw_parsed_generated_tokens = len(generated_token_ids)
        raw_token_logprob_count = len(token_logprobs)
        usage_completion_tokens = usage.get("completion_tokens")
        backend_generated_tokens = len(generated_token_ids) if generated_token_source == "completion_token_ids" else raw_token_logprob_count or len(generated_token_ids)

        # Dynamo/SGLang can return an inflated logprobs.tokens array on
        # completions requests even when token_logprobs has the true generated
        # length. Keep token IDs aligned to the authoritative logprob cardinality.
        tokens_array_overcount = False
        if generated_token_source == "logprobs_tokens" and raw_token_logprob_count and len(generated_token_ids) > raw_token_logprob_count:
            generated_token_ids = generated_token_ids[:raw_token_logprob_count]
            content = _decode_token_ids(self.tokenizer, generated_token_ids)
            tokens_array_overcount = True
        locally_truncated_to_max_tokens = False
        if len(generated_token_ids) > requested_max_tokens:
            generated_token_ids = generated_token_ids[:requested_max_tokens]
            token_logprobs = token_logprobs[:requested_max_tokens]
            content = _decode_token_ids(self.tokenizer, generated_token_ids)
            finish_reason = "length"
            response_nvext["stop_reason"] = None
            stop_reason = None
            locally_truncated_to_max_tokens = True

        if self.config.api_mode == "responses" and self.config.return_logprobs and generated_token_ids and not token_logprobs:
            raise RuntimeError("Dynamo /v1/responses returned completion token IDs without token logprobs")
        if len(token_logprobs) > len(generated_token_ids):
            token_logprobs = token_logprobs[: len(generated_token_ids)]
        elif len(token_logprobs) < len(generated_token_ids):
            token_logprobs.extend([0.0] * (len(generated_token_ids) - len(token_logprobs)))

        extra = {
            "response": data,
            "response_id": data.get("id") if self.config.api_mode == "responses" else None,
            "response_tool_calls": response_tool_calls,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "token_logprobs": token_logprobs,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "requested_max_tokens": requested_max_tokens,
            "backend_generated_tokens": backend_generated_tokens,
            "usage_completion_tokens": usage_completion_tokens,
            "parsed_generated_token_ids": raw_parsed_generated_tokens,
            "raw_token_logprob_count": raw_token_logprob_count,
            "generated_token_source": generated_token_source,
            "tokens_array_overcount": tokens_array_overcount,
            "locally_truncated_to_max_tokens": locally_truncated_to_max_tokens,
            "metadata_upload_url": success_metadata_upload_url,
            "uploaded_metadata": uploaded_metadata,
        }
        if locally_truncated_to_max_tokens:
            usage["completion_tokens"] = min(int(usage.get("completion_tokens", 0)), requested_max_tokens)
            usage["total_tokens"] = int(usage.get("prompt_tokens") or len(prompt_token_ids)) + usage["completion_tokens"]
            if self.config.api_mode == "completions" and data.get("usage"):
                data["usage"].update(usage)
        _completion_debug_log(
            "completion_response",
            url=url,
            attempt=success_attempt,
            x_request_id=x_request_id,
            status_code=response_status_code,
            elapsed_s=_round_s(success_elapsed_s),
            requested_max_tokens=requested_max_tokens,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            usage_prompt_tokens=usage.get("prompt_tokens"),
            usage_completion_tokens=usage.get("completion_tokens"),
            usage_total_tokens=usage.get("total_tokens"),
            backend_generated_tokens=backend_generated_tokens,
            raw_parsed_generated_tokens=raw_parsed_generated_tokens,
            raw_token_logprob_count=raw_token_logprob_count,
            generated_token_source=generated_token_source,
            generated_tokens=len(generated_token_ids),
            token_logprobs=len(token_logprobs),
            response_text_chars=len(content),
            tokens_array_overcount=tokens_array_overcount,
            locally_truncated_to_max_tokens=locally_truncated_to_max_tokens,
            **_payload_debug_fields(payload),
        )
        result = {"content": content, "message": content, "extra": extra}
        return result


def query(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    return DirectCompletionsModel().query(messages, **kwargs)


def stop_reason_token_ids(stop_reason: Any, tokenizer: Any | None = None) -> list[int]:
    """Return token IDs from Dynamo/OpenAI stop_reason values, without tokenizing."""

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
