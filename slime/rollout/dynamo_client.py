import asyncio
import base64
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


DYNAMO_API_MODES = {"completions", "responses"}


def build_metadata_upload_url(root_url: str, request_id: str | None) -> str:
    metadata_request_id = request_id or f"anonymous:{uuid.uuid4().hex}"
    readable_id = re.sub(r"[^A-Za-z0-9._-]+", "-", metadata_request_id).strip("-")[:120]
    request_hash = hashlib.sha256(metadata_request_id.encode("utf-8")).hexdigest()[:16]
    return f"{root_url.rstrip('/')}/{readable_id or 'request'}-{request_hash}"


def build_dynamo_payload(
    *,
    api_mode: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    stop: str | list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    min_tokens: int | None = None,
    ignore_eos: bool | None = None,
    seed: int | None = None,
    skip_special_tokens: bool | None = None,
    no_stop_trim: bool | None = None,
    spaces_between_special_tokens: bool | None = None,
    stream: bool = True,
    return_logprobs: bool = True,
    response_input: Any = None,
    previous_response_id: str | None = None,
    store: bool = False,
    metadata_upload_url: str | None = None,
    metadata_upload_format: str = "msgpack",
    agent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if api_mode not in DYNAMO_API_MODES:
        raise ValueError(f"Unsupported Dynamo API mode: {api_mode}")
    if not prompt_token_ids:
        raise ValueError("Dynamo requests require at least one prompt token")

    extra_fields = ["stop_reason", "completion_token_ids"]
    nvext: dict[str, Any] = {
        "extra_fields": extra_fields,
    }
    if metadata_upload_url:
        nvext["metadata_upload"] = {
            "url": metadata_upload_url,
            "format": metadata_upload_format,
        }
    if agent_context:
        nvext["agent_context"] = dict(agent_context)

    if api_mode == "responses":
        if response_input is None:
            raise ValueError("/v1/responses requests require response_input")
        if return_logprobs:
            extra_fields.append("completion_token_logprobs")
        payload: dict[str, Any] = {
            "model": model,
            "input": response_input,
            "max_output_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": bool(stream),
            "store": bool(store),
            "nvext": {
                **nvext,
                "token_data": [int(token_id) for token_id in prompt_token_ids],
            },
        }
        if return_logprobs:
            payload["include"] = ["message.output_text.logprobs"]
        if previous_response_id:
            payload["previous_response_id"] = str(previous_response_id)
    else:
        payload = {
            "model": model,
            "prompt": [int(token_id) for token_id in prompt_token_ids],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": bool(stream),
            "return_tokens_as_token_ids": True,
            "nvext": nvext,
        }
        if return_logprobs:
            payload["logprobs"] = 0

    if top_k is not None and int(top_k) > 0:
        payload["top_k"] = int(top_k)
    if ignore_eos is not None:
        payload["ignore_eos"] = bool(ignore_eos)
    if min_tokens is not None:
        payload["min_tokens"] = int(min_tokens)
    if seed is not None:
        payload["seed"] = int(seed)
    if skip_special_tokens is not None:
        payload["skip_special_tokens"] = bool(skip_special_tokens)
    if no_stop_trim is not None:
        payload["no_stop_trim"] = bool(no_stop_trim)
        payload["include_stop_str_in_output"] = bool(no_stop_trim)
    if spaces_between_special_tokens is not None:
        payload["spaces_between_special_tokens"] = bool(spaces_between_special_tokens)
    if stop:
        payload["stop"] = stop
    if stop_token_ids:
        payload["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]

    return payload


def read_uploaded_metadata(base_url: str, payload_format: str, retries: int = 8) -> dict[str, Any]:
    from dynamo.common.storage import get_fs

    import zstandard as zstd

    filename = f"choice_0.{payload_format}.zst"
    fs = get_fs(base_url)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = zstd.ZstdDecompressor().decompress(fs.cat(filename))
            if payload_format == "json":
                payload = json.loads(raw)
            else:
                import msgspec

                payload = msgspec.msgpack.decode(raw)
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            if not isinstance(metadata, dict):
                raise ValueError(f"invalid Dynamo metadata payload at {base_url}/{filename}")
            return metadata
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(min(0.25 * (2**attempt), 2.0))
    raise RuntimeError(f"failed to read Dynamo metadata upload {base_url}/{filename}: {last_error}")


async def read_uploaded_metadata_async(
    base_url: str,
    payload_format: str,
    retries: int = 8,
) -> dict[str, Any]:
    return await asyncio.to_thread(read_uploaded_metadata, base_url, payload_format, retries)


def metadata_token_data(metadata: dict[str, Any]) -> tuple[list[int], list[float]]:
    output_token_logprobs = metadata.get("output_token_logprobs") or []
    token_ids = [int(entry[1]) for entry in output_token_logprobs]
    token_logprobs = [float(entry[0]) for entry in output_token_logprobs]
    return token_ids, token_logprobs


def metadata_finish_reason(metadata: dict[str, Any]) -> str | None:
    finish_reason = metadata.get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    return str(finish_reason) if finish_reason else None


def _serialized_array(value: Any) -> tuple[np.ndarray, tuple[int, ...] | None]:
    if isinstance(value, np.ndarray):
        return value, tuple(value.shape)
    if isinstance(value, list):
        array = np.asarray(value, dtype=np.int32)
        return array, tuple(array.shape)

    shape = None
    dtype = np.dtype(np.int32)
    data = value
    if isinstance(value, dict):
        data = value.get("data")
        raw_shape = value.get("shape")
        if raw_shape is not None:
            shape = tuple(int(dim) for dim in raw_shape)
        if value.get("dtype"):
            dtype = np.dtype(str(value["dtype"]).removeprefix("torch."))

    if isinstance(data, str):
        data = base64.b64decode(data.encode("ascii"))
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=dtype), shape
    raise TypeError(f"Unsupported routed_experts payload: {type(value).__name__}")


def decode_routed_experts(
    metadata: dict[str, Any],
    *,
    expected_shape: tuple[int, int, int],
    trailing_loss_mask: list[int] | None = None,
    num_experts: int | None = None,
) -> np.ndarray | None:
    value = metadata.get("routed_experts")
    if value is None:
        return None

    array, serialized_shape = _serialized_array(value)
    if serialized_shape is not None:
        array = array.reshape(serialized_shape)
    if array.shape != expected_shape:
        per_token = expected_shape[1] * expected_shape[2]
        if array.size % per_token != 0:
            raise ValueError(f"routed_experts shape {array.shape} cannot be aligned to expected shape {expected_shape}")
        array = array.reshape(array.size // per_token, expected_shape[1], expected_shape[2])
        if array.shape[0] < expected_shape[0]:
            missing = expected_shape[0] - array.shape[0]
            if trailing_loss_mask is None or missing > len(trailing_loss_mask) or any(trailing_loss_mask[-missing:]) or num_experts is None:
                raise ValueError(f"routed_experts only covers {array.shape[0]} tokens; expected {expected_shape[0]}")
            padding = np.arange(missing * expected_shape[1] * expected_shape[2], dtype=np.int32).reshape(missing, expected_shape[1], expected_shape[2]) % int(num_experts)
            array = np.concatenate([array, padding], axis=0)
        else:
            array = array[: expected_shape[0]]
    return array.astype(np.int32, copy=False)


def apply_uploaded_metadata_to_sample(
    sample: Any,
    args: Any,
    metadata: dict[str, Any] | None,
) -> None:
    if not metadata:
        return
    if metadata.get("routed_experts") is not None:
        routed_experts = decode_routed_experts(
            metadata,
            expected_shape=(
                len(sample.tokens) - 1,
                args.num_layers,
                args.moe_router_topk,
            ),
            trailing_loss_mask=getattr(sample, "loss_mask", None),
            num_experts=getattr(args, "num_experts", None),
        )
        if routed_experts is not None:
            sample.rollout_routed_experts = routed_experts

    weight_version = metadata.get("weight_version")
    if weight_version is not None and (not sample.weight_versions or sample.weight_versions[-1] != str(weight_version)):
        sample.weight_versions.append(str(weight_version))


def apply_uploaded_metadata_sequence_to_sample(
    sample: Any,
    args: Any,
    metadata_sequence: list[dict[str, Any]] | None,
) -> None:
    if not metadata_sequence:
        return

    latest_routed_metadata = None
    for metadata in metadata_sequence:
        weight_version = metadata.get("weight_version")
        if weight_version is not None and (not sample.weight_versions or sample.weight_versions[-1] != str(weight_version)):
            sample.weight_versions.append(str(weight_version))
        if metadata.get("routed_experts") is not None:
            latest_routed_metadata = metadata

    if latest_routed_metadata is not None:
        apply_uploaded_metadata_to_sample(sample, args, latest_routed_metadata)


def _merge_cumulative(current: list[Any], incoming: list[Any]) -> list[Any]:
    if not incoming:
        return current
    if len(incoming) >= len(current) and incoming[: len(current)] == current:
        return list(incoming)
    return current + list(incoming)


def _token_ids_from_logprob_tokens(tokens: list[Any]) -> list[int]:
    token_ids = []
    for token in tokens:
        if isinstance(token, str) and token.startswith("token_id:"):
            token_ids.append(int(token[len("token_id:") :]))
    return token_ids


def _response_output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


@dataclass
class DynamoGeneration:
    api_mode: str
    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None
    stop_reason: Any = None
    response_id: str | None = None
    response: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    uploaded_metadata: dict[str, Any] | None = None
    terminal_event_received: bool = False

    def _consume_nvext(self, nvext: Any, *, cumulative: bool) -> None:
        if not isinstance(nvext, dict):
            return
        incoming_token_ids = [int(token_id) for token_id in nvext.get("completion_token_ids") or []]
        incoming_logprobs = [float(value) for value in nvext.get("completion_token_logprobs") or []]
        if cumulative:
            self.token_ids = _merge_cumulative(self.token_ids, incoming_token_ids)
            self.token_logprobs = _merge_cumulative(self.token_logprobs, incoming_logprobs)
        else:
            self.token_ids.extend(incoming_token_ids)
            self.token_logprobs.extend(incoming_logprobs)
        if nvext.get("stop_reason") is not None:
            self.stop_reason = nvext["stop_reason"]

    def consume_sse(self, event_type: str | None, data: dict[str, Any]) -> None:
        if data.get("error"):
            raise RuntimeError(f"Dynamo stream error: {data['error']}")
        if self.api_mode == "completions":
            self._consume_completion_chunk(data)
        else:
            self._consume_response_event(event_type or str(data.get("type") or ""), data)

    def _consume_completion_chunk(self, data: dict[str, Any]) -> None:
        choices = data.get("choices") or []
        if not choices:
            self._consume_nvext(data.get("nvext"), cumulative=False)
            if isinstance(data.get("usage"), dict):
                self.usage = dict(data["usage"])
            return

        choice = choices[0]
        self.text += str(choice.get("text") or "")
        logprobs = choice.get("logprobs") or {}
        chunk_logprobs = [float(value) for value in logprobs.get("token_logprobs") or []]
        chunk_token_ids = _token_ids_from_logprob_tokens(logprobs.get("tokens") or [])

        nvext = data.get("nvext") or choice.get("nvext")
        token_count_before = len(self.token_ids)
        logprob_count_before = len(self.token_logprobs)
        self._consume_nvext(nvext, cumulative=False)
        if len(self.token_ids) == token_count_before:
            self.token_ids.extend(chunk_token_ids)
        if chunk_logprobs and len(self.token_logprobs) == logprob_count_before:
            self.token_logprobs.extend(chunk_logprobs)

        if choice.get("finish_reason"):
            self.finish_reason = str(choice["finish_reason"])
            self.terminal_event_received = True
        if choice.get("stop_reason") is not None:
            self.stop_reason = choice["stop_reason"]
        if isinstance(data.get("usage"), dict):
            self.usage = dict(data["usage"])

    def _consume_response_event(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            self.terminal_event_received = True
        self._consume_nvext(data.get("nvext"), cumulative=True)
        if event_type == "response.output_text.delta":
            self.text += str(data.get("delta") or "")

        response = data.get("response")
        if not isinstance(response, dict):
            return
        self.response = response
        self.response_id = response.get("id") or self.response_id
        self._consume_nvext(response.get("nvext"), cumulative=True)
        final_text = _response_output_text(response)
        if final_text:
            self.text = final_text
        if isinstance(response.get("usage"), dict):
            usage = response["usage"]
            self.usage = {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        status = response.get("status")
        if status == "incomplete":
            details = response.get("incomplete_details") or {}
            reason = details.get("reason")
            self.stop_reason = reason or self.stop_reason
            self.finish_reason = "content_filter" if reason == "content_filter" else "length"
        elif status == "completed":
            self.finish_reason = "stop"
        elif status in {"failed", "cancelled"}:
            self.finish_reason = str(status)

    def apply_metadata(self, metadata: dict[str, Any]) -> None:
        self.uploaded_metadata = metadata
        token_ids, token_logprobs = metadata_token_data(metadata)
        if token_ids:
            if self.token_ids:
                streamed_tokens = len(self.token_ids)
                if token_ids[:streamed_tokens] != self.token_ids:
                    raise RuntimeError("Dynamo metadata token IDs do not match streamed completion token IDs")
                token_ids = token_ids[:streamed_tokens]
                token_logprobs = token_logprobs[:streamed_tokens]
            self.token_ids = token_ids
            self.token_logprobs = token_logprobs
        if finish_reason := metadata_finish_reason(metadata):
            if self.finish_reason is None or not self.terminal_event_received:
                self.finish_reason = finish_reason

    def align_logprobs(self, *, required: bool) -> None:
        if len(self.token_logprobs) == len(self.token_ids):
            return
        if required and self.token_ids:
            raise RuntimeError("Dynamo returned generated token IDs without one selected-token logprob per token")
        self.token_logprobs = self.token_logprobs[: len(self.token_ids)]
        self.token_logprobs.extend([0.0] * (len(self.token_ids) - len(self.token_logprobs)))
