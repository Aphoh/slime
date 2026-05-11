"""Direct token-ID `/v1/completions` adapter for SWE-bench Pro rollouts.

This module intentionally does not import SWE-agent or LiteLLM. It formats chat
prompts locally with the HF tokenizer and sends token IDs to an OpenAI-compatible
completions endpoint.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - parser-only local tooling may not have transformers.
    AutoTokenizer = None  # type: ignore


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


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


def glm_stop_strings_from_token_ids(token_ids: list[int]) -> list[str]:
    """Return GLM stop strings for the GLM special stop token IDs we use."""
    stops: list[str] = []
    for token_id in token_ids:
        if token_id == GLM_TOOL_CLOSE_TOKEN_ID:
            stops.append("</tool_call>")
        elif token_id == GLM_EOS_TOKEN_ID:
            stops.append("<|endoftext|>")
        else:
            raise ValueError(
                f"/v1/completions does not accept stop_token_ids; "
                f"no GLM stop string mapping for token_id:{token_id}"
            )
    return stops


@dataclass
class DirectCompletionsConfig:
    base_url: str
    tokenizer_path: str
    model: str = "default"
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    timeout: float = 600.0
    retries: int = 5

    @classmethod
    def from_env(cls) -> "DirectCompletionsConfig":
        base_url = _env("SWEPRO_DYNAMO_FRONTEND_URL", _env("DYNAMO_FRONTEND_URL", _env("OPENAI_BASE_URL")))
        tokenizer_path = _env("SWEPRO_TOKENIZER_PATH", _env("HF_CHECKPOINT", _env("MODEL_PATH")))
        if not base_url:
            raise ValueError("Set SWEPRO_DYNAMO_FRONTEND_URL or DYNAMO_FRONTEND_URL")
        if not tokenizer_path:
            raise ValueError("Set SWEPRO_TOKENIZER_PATH, HF_CHECKPOINT, or MODEL_PATH")
        return cls(
            base_url=base_url.rstrip("/"),
            tokenizer_path=tokenizer_path,
            model=_env("SWEPRO_MODEL", tokenizer_path) or "default",
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
        return self._build_payload_from_ids(prompt_ids, **kwargs)

    def _build_payload_from_ids(self, prompt_ids: list[int], **kwargs) -> dict[str, Any]:
        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt_ids,
            "max_tokens": int(kwargs.get("max_tokens", self.config.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.config.temperature)),
            "top_p": float(kwargs.get("top_p", self.config.top_p)),
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "stream": False,
            "nvext": {"extra_fields": ["stop_reason"]},
        }
        top_k = kwargs.get("top_k", self.config.top_k)
        if top_k is not None and int(top_k) > 0:
            payload["top_k"] = int(top_k)
        stop_token_ids = kwargs.get("stop_token_ids")
        stop = kwargs.get("stop")
        if stop_token_ids:
            mapped_stops = glm_stop_strings_from_token_ids([int(token_id) for token_id in stop_token_ids])
            stop = list(stop or []) + mapped_stops if isinstance(stop, list) else ([stop] if stop else []) + mapped_stops
        if stop:
            payload["stop"] = stop
        agent_context = kwargs.get("agent_context")
        if agent_context:
            payload["nvext"]["agent_context"] = dict(agent_context)
        # The updated Dynamo OpenAI frontend rejects stop_token_ids as an unsupported
        # top-level parameter, so we send GLM special stops as strings instead.
        return payload

    def complete_prompt_ids(
        self,
        prompt_ids: list[int],
        *,
        trace_messages: list[dict[str, Any]] | None = None,
        x_request_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        payload = self._build_payload_from_ids(prompt_ids, **kwargs)
        return self._post_payload(payload, trace_messages or [], x_request_id=x_request_id)

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
        url = f"{self.config.base_url}/v1/completions"
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                request_id = f"{x_request_id}:try:{attempt}" if x_request_id else None
                headers = {"x-request-id": request_id} if request_id else None
                response = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.config.retries:
                    raise
                time.sleep(min(2**attempt, 30))
        else:
            raise RuntimeError(f"completion request failed: {last_error}")

        choice = data["choices"][0]
        response_nvext = data.get("nvext") or {}
        logprobs = choice.get("logprobs") or {}
        generated_token_ids = token_ids_from_logprob_tokens(logprobs.get("tokens") or [], self.tokenizer)
        token_logprobs = list(logprobs.get("token_logprobs") or [])
        if len(token_logprobs) > len(generated_token_ids):
            token_logprobs = token_logprobs[: len(generated_token_ids)]
        elif len(token_logprobs) < len(generated_token_ids):
            token_logprobs.extend([0.0] * (len(generated_token_ids) - len(token_logprobs)))

        extra = {
            "response": data,
            "prompt_token_ids": payload["prompt"],
            "generated_token_ids": generated_token_ids,
            "token_logprobs": token_logprobs,
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": response_nvext.get("stop_reason", choice.get("stop_reason")),
        }
        result = {"content": choice.get("text", ""), "message": choice.get("text", ""), "extra": extra}
        return result


def query(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    return DirectCompletionsModel().query(messages, **kwargs)


GLM_TOOL_CALL_TOKEN_ID = 154843
GLM_TOOL_CLOSE_TOKEN_ID = 154844
GLM_TOOL_RESPONSE_START_TOKEN_ID = 154845
GLM_TOOL_RESPONSE_END_TOKEN_ID = 154846
GLM_ASSISTANT_TOKEN_ID = 154828
GLM_OBSERVATION_TOKEN_ID = 154829
GLM_THINK_START_TOKEN_ID = 154841
GLM_EOS_TOKEN_ID = 154820

GLM_TOOL_STOPS = [
    "</tool_call>",
    "<|endoftext|>",
]

GLM_TOOL_STOP_TOKEN_IDS = [
    GLM_TOOL_CLOSE_TOKEN_ID,
    GLM_EOS_TOKEN_ID,
]


def stop_reason_token_ids(stop_reason: Any) -> list[int]:
    """Return token IDs from Dynamo/OpenAI stop_reason values, without tokenizing."""

    if stop_reason is None or isinstance(stop_reason, bool):
        return []
    if isinstance(stop_reason, int):
        return [stop_reason]
    if isinstance(stop_reason, str):
        stop_reason = stop_reason.strip()
        if stop_reason == "</tool_call>":
            return [GLM_TOOL_CLOSE_TOKEN_ID]
        if stop_reason == "<|endoftext|>":
            return [GLM_EOS_TOKEN_ID]
        if stop_reason.startswith("token_id:"):
            stop_reason = stop_reason[len("token_id:") :]
        try:
            return [int(stop_reason)]
        except ValueError:
            return []
    if isinstance(stop_reason, list):
        token_ids: list[int] = []
        for item in stop_reason:
            token_ids.extend(stop_reason_token_ids(item))
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

    if GLM_TOOL_CLOSE_TOKEN_ID not in matched_stop_token_ids:
        return content, [], False

    tool_start = _rfind_token(generated_ids, GLM_TOOL_CALL_TOKEN_ID)
    if tool_start < 0:
        return content, [], False

    tool_text = tokenizer.decode(generated_ids[tool_start:], skip_special_tokens=False)
    has_tool_close = GLM_TOOL_CLOSE_TOKEN_ID in generated_ids[tool_start:]
    if not has_tool_close:
        tool_text = tool_text + "</tool_call>"

    marker_idx = content.rfind("<tool_call>")
    if marker_idx >= 0:
        normal_text = content[:marker_idx]
    else:
        normal_text = tokenizer.decode(generated_ids[:tool_start], skip_special_tokens=False)

    _, tool_calls = parse_glm_tool_calls(tool_text)
    return normal_text, tool_calls, not has_tool_close


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
