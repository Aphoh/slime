"""Backend-neutral helpers for token/logprob streaming responses."""

from typing import Any


def merge_stream_chunk(
    *,
    tokens: list[int],
    log_probs: list[float],
    text: str,
    chunk: dict[str, Any],
) -> tuple[list[int], list[float], str]:
    """Merge one cumulative or disjoint SGLang-compatible stream chunk."""
    meta = chunk.get("meta_info") or {}
    pairs = meta.get("output_token_logprobs") or []
    chunk_tokens = [item[1] for item in pairs]
    chunk_log_probs = [item[0] for item in pairs]
    output_length = meta.get("output_token_logprobs_length")

    if output_length is not None:
        output_length = int(output_length)
        if len(chunk_tokens) == output_length:
            cumulative = True
        elif len(tokens) + len(chunk_tokens) == output_length:
            cumulative = False
        else:
            raise ValueError(
                "Inconsistent streaming output_token_logprobs_length: "
                f"received={len(chunk_tokens)}, accumulated={len(tokens)}, reported={output_length}."
            )
    else:
        cumulative = len(chunk_tokens) >= len(tokens) and chunk_tokens[: len(tokens)] == tokens

    chunk_text = chunk.get("text")
    if cumulative:
        return chunk_tokens, chunk_log_probs, text if chunk_text is None else chunk_text
    return tokens + chunk_tokens, log_probs + chunk_log_probs, text + (chunk_text or "")
