from __future__ import annotations

from typing import ClassVar, Self

import pytest

from core.parallel import map_seeds


def _double(context: int, seed: int) -> int:
    return context * seed


def _explode(context: int, seed: int) -> int:
    raise ValueError(f"seed {seed} failed")


class _SerialExecutor:
    """Synchronous stand-in for ProcessPoolExecutor, recording how it was constructed."""

    instances: ClassVar[list[_SerialExecutor]] = []

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers
        self.chunksizes: list[int] = []
        _SerialExecutor.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def map(self, fn, iterable, chunksize=1):
        self.chunksizes.append(chunksize)
        return [fn(item) for item in iterable]


@pytest.fixture(autouse=True)
def _reset_executor_instances():
    _SerialExecutor.instances = []
    yield
    _SerialExecutor.instances = []


class TestMapSeeds:
    def test_returns_one_result_per_seed_in_seed_order(self):
        result = map_seeds(_double, 10, [1, 2, 3], executor_factory=_SerialExecutor)

        assert result == [10, 20, 30]

    def test_preserves_order_regardless_of_completion_order(self):
        result = map_seeds(_double, 1, [5, 4, 3, 2, 1], executor_factory=_SerialExecutor)

        assert result == [5, 4, 3, 2, 1]

    def test_an_empty_seed_list_never_starts_an_executor(self):
        result = map_seeds(_double, 10, [], executor_factory=_SerialExecutor)

        assert result == []
        assert _SerialExecutor.instances == []

    def test_never_requests_more_workers_than_there_are_seeds(self):
        map_seeds(_double, 1, [1, 2], executor_factory=_SerialExecutor, max_workers=32)

        assert _SerialExecutor.instances[0].max_workers == 2

    def test_honours_an_explicit_worker_cap_below_the_seed_count(self):
        map_seeds(_double, 1, list(range(1, 21)), executor_factory=_SerialExecutor, max_workers=3)

        assert _SerialExecutor.instances[0].max_workers == 3

    def test_chunks_so_each_worker_receives_the_context_a_bounded_number_of_times(self):
        map_seeds(_double, 1, list(range(1, 101)), executor_factory=_SerialExecutor, max_workers=4)

        assert _SerialExecutor.instances[0].chunksizes == [25]

    def test_a_worker_failure_propagates_rather_than_being_silently_dropped(self):
        with pytest.raises(ValueError, match="seed 1 failed"):
            map_seeds(_explode, 1, [1, 2, 3], executor_factory=_SerialExecutor)

    def test_a_single_worker_runs_inline_without_paying_for_an_executor(self):
        result = map_seeds(_double, 7, [1, 2, 3], executor_factory=_SerialExecutor, max_workers=1)

        assert result == [7, 14, 21]
        assert _SerialExecutor.instances == []

    def test_a_single_seed_runs_inline_because_workers_cannot_exceed_seeds(self):
        map_seeds(_double, 1, [4], executor_factory=_SerialExecutor, max_workers=8)

        assert _SerialExecutor.instances == []

    def test_runs_the_real_process_pool_by_default(self):
        assert map_seeds(_double, 3, [1, 2, 3]) == [3, 6, 9]

    def test_a_nested_call_inside_a_worker_process_runs_sequentially(self):
        result = map_seeds(
            _double,
            5,
            [1, 2, 3],
            executor_factory=_SerialExecutor,
            in_worker_process=lambda: True,
        )

        assert result == [5, 10, 15]
        assert _SerialExecutor.instances == []

    def test_a_top_level_call_still_uses_the_pool(self):
        map_seeds(
            _double,
            1,
            [1, 2, 3],
            executor_factory=_SerialExecutor,
            max_workers=3,
            in_worker_process=lambda: False,
        )

        assert _SerialExecutor.instances[0].max_workers == 3

    def test_nesting_is_detected_from_the_real_process_by_default(self):
        from core.parallel import in_worker_process

        assert in_worker_process() is False
