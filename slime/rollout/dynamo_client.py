"""Client-side helpers for Dynamo native SGLang ``/generate``.

The endpoint is provided by ai-dynamo/dynamo#11640. It is a streaming,
token-in/token-out API: all rollout metadata needed by Slime travels in the
SSE stream, so the client has no object-store or Dynamo-Python dependency.
The Dynamo frontend must explicitly enable it with
``DYN_SGLANG_ENABLE_GENERATE=1``.
"""

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def build_dynamo_generate_payload(
    *,
    prompt_token_ids: list[int],
    sampling_params: dict[str, Any],
    request_id: str,
    return_routed_experts: bool = False,
) -> dict[str, Any]:
    """Build a request for Dynamo SGLang-native streaming Generate API."""
    if not prompt_token_ids:
        raise ValueError("Dynamo generate requests require at least one prompt token")

    native_sampling_params = {
        name: value
        for name, value in sampling_params.items()
        if value is not None
    }
    if native_sampling_params.get("n", 1) != 1:
        raise ValueError("Dynamo SGLang-native Generate API supports sampling_params.n=1 only")

    return {
        "rid": request_id,
        "input_ids": [int(token_id) for token_id in prompt_token_ids],
        "sampling_params": native_sampling_params,
        "return_logprob": True,
        "logprob_start_len": -1,
        "stream": True,
        "return_routed_experts": bool(return_routed_experts),
    }


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


def apply_dynamo_metadata_to_sample(
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


def apply_dynamo_metadata_sequence_to_sample(
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
        apply_dynamo_metadata_to_sample(sample, args, latest_routed_metadata)


@dataclass
class DynamoGeneration:
    """Accumulate disjoint SGLang-native SSE chunks from Dynamo."""

    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None
    stop_reason: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata_sequence: list[dict[str, Any]] = field(default_factory=list)
    terminal_event_received: bool = False

    def consume_sse(self, data: dict[str, Any]) -> None:
        if data.get("error"):
            raise RuntimeError("Dynamo stream error: {}".format(data["error"]))

        output_ids = [int(token_id) for token_id in data.get("output_ids") or []]
        metadata = data.get("meta_info") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Dynamo native Generate stream returned non-object meta_info")

        logprob_entries = metadata.get("output_token_logprobs") or []
        if logprob_entries and len(logprob_entries) != len(output_ids):
            raise RuntimeError("Dynamo returned a different number of token IDs and selected-token logprobs")
        if output_ids and not logprob_entries:
            raise RuntimeError("Dynamo returned output token IDs without selected-token logprobs")
        for output_id, entry in zip(output_ids, logprob_entries, strict=True):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise RuntimeError("Dynamo returned malformed selected-token logprobs")
            if int(entry[1]) != output_id:
                raise RuntimeError("Dynamo selected-token logprobs do not match output_ids")
            self.token_logprobs.append(float(entry[0]))
        self.token_ids.extend(output_ids)
        self.text += str(data.get("text") or "")
        self.metadata_sequence.append(dict(metadata))

        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            if key in metadata:
                self.usage[key] = metadata[key]

        finish_reason = metadata.get("finish_reason")
        if isinstance(finish_reason, dict):
            self.stop_reason = finish_reason.get("matched", self.stop_reason)
            finish_reason = finish_reason.get("type")
        if finish_reason is not None:
            self.finish_reason = str(finish_reason)
            self.terminal_event_received = True
