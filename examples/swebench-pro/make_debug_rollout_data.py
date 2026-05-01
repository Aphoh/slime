#!/usr/bin/env python3
"""Create synthetic slime debug-rollout data for trainer-only repros.

The output matches ``--save-debug-rollout-data`` so it can be consumed with
``--load-debug-rollout-data``.  This is intentionally trainer-oriented: it does
not try to look like SWE-agent text, it only creates valid token/reward/loss
fields that exercise Megatron logprob and train paths at controlled lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slime.utils.types import Sample


def _parse_lengths(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths:
        raise ValueError("--lengths must contain at least one integer")
    if any(length < 2 for length in lengths):
        raise ValueError("all total lengths must be >= 2")
    return lengths


def _make_tokens(*, sample_idx: int, total_length: int, vocab_size: int, min_token_id: int) -> list[int]:
    usable_vocab = max(1, vocab_size - min_token_id)
    return [min_token_id + ((sample_idx * 104729 + pos) % usable_vocab) for pos in range(total_length)]


def _make_sample(
    *,
    sample_idx: int,
    total_length: int,
    response_length: int,
    vocab_size: int,
    min_token_id: int,
    include_rollout_logprobs: bool,
    reward: float | None = None,
    loss_mask: list[int] | None = None,
    status: Sample.Status = Sample.Status.COMPLETED,
    metadata: dict | None = None,
) -> Sample:
    if response_length >= total_length:
        raise ValueError(f"response_length={response_length} must be smaller than total_length={total_length}")

    reward = float(sample_idx % 2) if reward is None else reward
    if loss_mask is None:
        loss_mask = [1] * response_length
    if len(loss_mask) != response_length:
        raise ValueError(
            f"loss_mask length {len(loss_mask)} must equal response_length={response_length} for sample {sample_idx}"
        )
    sample_metadata = {
        "instance_id": f"synthetic-{sample_idx}",
        "synthetic_debug_rollout": True,
        "total_length": total_length,
        "response_length": response_length,
    }
    if metadata:
        sample_metadata.update(metadata)
        sample_metadata["source_instance_id"] = metadata.get("instance_id")
    sample_metadata["synthetic_debug_rollout"] = True
    sample = Sample(
        group_index=sample_idx,
        index=sample_idx,
        prompt=f"synthetic trainer repro sample {sample_idx}",
        tokens=_make_tokens(
            sample_idx=sample_idx,
            total_length=total_length,
            vocab_size=vocab_size,
            min_token_id=min_token_id,
        ),
        response=f"<synthetic response {sample_idx}>",
        response_length=response_length,
        label=f"synthetic-{sample_idx}",
        reward=reward,
        loss_mask=loss_mask,
        status=status,
        metadata=sample_metadata,
    )
    if include_rollout_logprobs:
        sample.rollout_log_probs = [0.0] * response_length
    return sample


def _load_source_samples(path: Path) -> list[Sample]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{path} does not contain a list-valued 'samples' field")
    return [Sample.from_dict(sample) if isinstance(sample, dict) else sample for sample in raw_samples]


def _make_samples_from_source_shape(
    *,
    source_samples: list[Sample],
    num_samples: int | None,
    vocab_size: int,
    min_token_id: int,
    include_rollout_logprobs: bool,
) -> list[Sample]:
    if not source_samples:
        raise ValueError("source rollout contains no samples")
    selected = source_samples if num_samples is None else source_samples[:num_samples]
    samples = []
    for sample_idx, source in enumerate(selected):
        total_length = len(source.tokens)
        response_length = source.response_length
        if total_length < 2:
            raise ValueError(f"source sample {sample_idx} total length must be >= 2")
        if response_length <= 0:
            raise ValueError(f"source sample {sample_idx} response length must be positive")
        if response_length >= total_length:
            response_length = total_length - 1
        loss_mask = source.loss_mask if source.loss_mask is not None else [1] * response_length
        if len(loss_mask) != response_length:
            raise ValueError(
                f"source sample {sample_idx} loss_mask length {len(loss_mask)} != response_length {response_length}"
            )
        sample = _make_sample(
            sample_idx=sample_idx,
            total_length=total_length,
            response_length=response_length,
            vocab_size=vocab_size,
            min_token_id=min_token_id,
            include_rollout_logprobs=include_rollout_logprobs or source.rollout_log_probs is not None,
            reward=source.reward if isinstance(source.reward, float | int) else None,
            loss_mask=list(loss_mask),
            status=source.status,
            metadata={
                **source.metadata,
                "source_total_length": total_length,
                "source_response_length": source.response_length,
                "source_status": source.status.value,
            },
        )
        if source.rollout_log_probs is not None:
            sample.rollout_log_probs = [0.0] * len(source.rollout_log_probs)
        samples.append(sample)
    return samples


def write_rollout(path: Path, *, rollout_id: int, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rollout_id": rollout_id,
            "samples": [sample.to_dict() for sample in samples],
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-template",
        required=True,
        help="Output path template, e.g. /data/swebench-pro/debug/rollout_{rollout_id}.pt",
    )
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--shape-from-rollout-data",
        type=Path,
        default=None,
        help="Read a saved debug rollout .pt file and synthesize samples with matching lengths, masks, rewards, and statuses.",
    )
    parser.add_argument(
        "--total-length",
        type=int,
        default=131072,
        help="Total sequence length when --lengths is not set.",
    )
    parser.add_argument(
        "--lengths",
        default=None,
        help="Comma-separated total sequence lengths. Cycled until --num-samples is reached.",
    )
    parser.add_argument("--response-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument(
        "--min-token-id",
        type=int,
        default=1000,
        help="First token id used for synthetic data; keeps pad/special ids out of the stream.",
    )
    parser.add_argument("--include-rollout-logprobs", action="store_true")
    args = parser.parse_args()

    if args.num_rollouts <= 0:
        parser.error("--num-rollouts must be positive")
    if args.shape_from_rollout_data is None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.total_length < 2:
        parser.error("--total-length must be >= 2")
    if args.response_length <= 0:
        parser.error("--response-length must be positive")
    if args.min_token_id < 0 or args.min_token_id >= args.vocab_size:
        parser.error("--min-token-id must be in [0, vocab_size)")

    source_samples = _load_source_samples(args.shape_from_rollout_data) if args.shape_from_rollout_data else None
    lengths = _parse_lengths(args.lengths) or [args.total_length]
    for rollout_id in range(args.num_rollouts):
        if source_samples is not None:
            samples = _make_samples_from_source_shape(
                source_samples=source_samples,
                num_samples=args.num_samples if args.num_samples > 0 else None,
                vocab_size=args.vocab_size,
                min_token_id=args.min_token_id,
                include_rollout_logprobs=args.include_rollout_logprobs,
            )
        else:
            samples = []
            for sample_idx in range(args.num_samples):
                total_length = lengths[sample_idx % len(lengths)]
                response_length = min(args.response_length, total_length - 1)
                samples.append(
                    _make_sample(
                        sample_idx=sample_idx,
                        total_length=total_length,
                        response_length=response_length,
                        vocab_size=args.vocab_size,
                        min_token_id=args.min_token_id,
                        include_rollout_logprobs=args.include_rollout_logprobs,
                    )
                )

        output_path = Path(args.output_template.format(rollout_id=rollout_id))
        write_rollout(output_path, rollout_id=rollout_id, samples=samples)
        total_tokens = sum(len(sample.tokens) for sample in samples)
        max_len = max(len(sample.tokens) for sample in samples)
        response_lengths = [sample.response_length for sample in samples]
        print(
            f"wrote {output_path} samples={len(samples)} total_tokens={total_tokens} max_len={max_len} "
            f"response_min={min(response_lengths)} response_max={max(response_lengths)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
