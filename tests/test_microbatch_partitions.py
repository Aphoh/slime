import pytest

from slime.utils.microbatch import get_capped_partitions


def _partition_sums(lengths, partitions):
    return [sum(lengths[idx] for idx in part) for part in partitions]


def test_get_capped_partitions_respects_cap_when_possible():
    lengths = [3000, 3000, 2000, 2000]

    partitions = get_capped_partitions(lengths, num_partitions=2, max_tokens=5000)

    assert sorted(sum(partitions, [])) == [0, 1, 2, 3]
    assert all(total <= 5000 for total in _partition_sums(lengths, partitions))


def test_get_capped_partitions_allows_oversized_singleton():
    lengths = [4000, 9000, 3000]

    partitions = get_capped_partitions(lengths, num_partitions=2, max_tokens=8192)

    assert sorted(sum(partitions, [])) == [0, 1, 2]
    assert partitions[1] == [1]
    assert _partition_sums(lengths, partitions) == [7000, 9000]


def test_get_capped_partitions_errors_when_partition_count_is_insufficient():
    with pytest.raises(AssertionError, match="Unable to create capped microbatch partitions"):
        get_capped_partitions([6000, 6000, 6000], num_partitions=2, max_tokens=8192)
