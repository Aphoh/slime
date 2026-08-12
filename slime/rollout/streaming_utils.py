"""Helpers for consuming SGLang streaming responses."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from slime.utils.misc import decode_int32_meta_array

_TOP_P_TOKEN_ID_META_KEYS = ("top_p_token_ids", "top_p_kept_token_ids")
_TOP_P_TOKEN_OFFSET_META_KEYS = ("top_p_token_offsets", "top_p_kept_token_offsets")


def _has_full_incremental_top_p_metadata(
    meta_info: dict[str, Any],
    *,
    new_token_count: int,
    reported_length: int,
) -> bool:
    token_ids = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
    offsets = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
    if token_ids is None or offsets is None or offsets.numel() == new_token_count + 1:
        return False
    if offsets.numel() != reported_length + 1:
        raise ValueError(
            "Incremental SGLang top-p metadata must describe either the current chunk or the full response: "
            f"offsets={offsets.numel()}, chunk_tokens={new_token_count}, reported_tokens={reported_length}."
        )

    return True


@dataclass(frozen=True)
class SGLangStreamUpdate:
    """A normalized update ready to apply to the current HTTP call state."""

    replace_call_state: bool
    output_mode: Literal["incremental", "cumulative"] | None
    tokens: list[int]
    log_probs: list[float]
    text: str
    meta_info: dict[str, Any]


@dataclass
class SGLangStreamAccumulator:
    """Classify cumulative versus incremental SGLang stream chunks.

    SGLang's ``output_token_logprobs_length`` is cumulative in both modes.
    The number of logprob pairs in the chunk therefore tells us whether its
    token-aligned payload is a complete snapshot or only the latest delta.
    """

    output_length: int = 0
    tokens: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    text: str = ""
    output_mode: Literal["incremental", "cumulative"] | None = None

    def add(self, chunk: dict[str, Any], *, decode: Callable[[list[int]], str]) -> SGLangStreamUpdate:
        meta_info = chunk.get("meta_info") or {}
        if "output_token_logprobs_length" not in meta_info:
            raise ValueError("SGLang streaming responses must include output_token_logprobs_length.")

        pairs = meta_info.get("output_token_logprobs") or []
        chunk_tokens = [item[1] for item in pairs]
        chunk_log_probs = [item[0] for item in pairs]
        reported_length = int(meta_info["output_token_logprobs_length"])

        is_cumulative = len(chunk_tokens) == reported_length
        is_incremental = self.output_length + len(chunk_tokens) == reported_length
        if is_cumulative and is_incremental:
            detected_mode = None
        elif is_cumulative:
            detected_mode = "cumulative"
        elif is_incremental:
            detected_mode = "incremental"
        else:
            raise ValueError(
                "Inconsistent streaming output_token_logprobs_length: "
                f"received={len(chunk_tokens)}, previous={self.output_length}, reported={reported_length}."
            )
        if detected_mode is not None:
            if self.output_mode is not None and self.output_mode != detected_mode:
                raise ValueError(
                    "SGLang changed streaming output mode within one request "
                    f"(already using {self.output_mode}, received {detected_mode})."
                )
            self.output_mode = detected_mode

        cumulative = self.output_mode != "incremental"

        full_top_p_metadata = not cumulative and _has_full_incremental_top_p_metadata(
            meta_info,
            new_token_count=len(chunk_tokens),
            reported_length=reported_length,
        )

        chunk_text = chunk.get("text")
        if cumulative:
            self.tokens = list(chunk_tokens)
            self.log_probs = list(chunk_log_probs)
            self.text = decode(self.tokens) if chunk_text is None else chunk_text
            update_text = self.text
        else:
            self.tokens.extend(chunk_tokens)
            self.log_probs.extend(chunk_log_probs)
            if chunk_text is not None:
                update_text = chunk_text
                self.text += chunk_text
            else:
                decoded_text = decode(self.tokens)
                if not decoded_text.startswith(self.text):
                    raise ValueError("Decoded incremental stream text does not extend the previously received text.")
                update_text = decoded_text[len(self.text) :]
                self.text = decoded_text

        replace_call_state = cumulative or full_top_p_metadata
        self.output_length = reported_length
        return SGLangStreamUpdate(
            replace_call_state=replace_call_state,
            output_mode=self.output_mode,
            tokens=list(self.tokens) if replace_call_state else chunk_tokens,
            log_probs=list(self.log_probs) if replace_call_state else chunk_log_probs,
            text=self.text if replace_call_state else update_text,
            meta_info=meta_info,
        )
