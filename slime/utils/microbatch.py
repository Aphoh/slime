from collections.abc import Sequence


def get_capped_partitions(seqlen_list: Sequence[int], num_partitions: int, max_tokens: int) -> list[list[int]]:
    """First-fit partitioning with a soft per-partition token cap.

    The cap is strict when samples can be packed under it. A single sequence
    longer than ``max_tokens`` cannot be split by microbatching, so it is placed
    alone in an otherwise empty partition.
    """
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be positive, got {num_partitions}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    partitions: list[list[int]] = [[] for _ in range(num_partitions)]
    sums = [0] * num_partitions

    for idx, length in enumerate(seqlen_list):
        if length < 0:
            raise ValueError(f"sequence length must be non-negative, got {length} at index {idx}")

        target = None
        for i in range(num_partitions):
            if sums[i] + length <= max_tokens:
                target = i
                break

        if target is None and length > max_tokens:
            for i in range(num_partitions):
                if not partitions[i]:
                    target = i
                    break

        if target is None:
            raise AssertionError(
                "Unable to create capped microbatch partitions: "
                f"sample index {idx} with length {length} cannot fit into "
                f"{num_partitions} partitions with max_tokens={max_tokens}. "
                f"Current partition token sums: {sums}"
            )

        partitions[target].append(idx)
        sums[target] += length

    return [sorted(p) for p in partitions]
