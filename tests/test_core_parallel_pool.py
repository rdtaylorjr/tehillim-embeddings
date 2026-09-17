import io
import os

from core.parallel import (
    chunksize_for,
    default_max_workers,
    map_in_order,
    progress_milestones,
    report_progress,
)


def _double(value: int) -> int:
    return value * 2


def _pid(value: int) -> tuple[int, int]:
    return value, os.getpid()


def test_map_in_order_returns_one_result_per_item() -> None:
    assert map_in_order(_double, [1, 2, 3], max_workers=2) == [2, 4, 6]


def test_map_in_order_preserves_submission_order_not_completion_order() -> None:
    """Row order must never depend on worker timing, or reruns stop being byte-comparable."""
    items = list(range(24))

    assert map_in_order(_double, items, max_workers=4) == [i * 2 for i in items]


def test_map_in_order_matches_a_plain_sequential_map() -> None:
    items = list(range(10))

    assert map_in_order(_double, items, max_workers=4) == [_double(i) for i in items]


def test_map_in_order_runs_in_this_process_when_workers_is_one() -> None:
    """A single worker stays in-process, so tests and debugging avoid the spawn cost."""
    results = map_in_order(_pid, [1, 2], max_workers=1)

    assert [value for value, _ in results] == [1, 2]
    assert {pid for _, pid in results} == {os.getpid()}


def test_map_in_order_uses_worker_processes_when_workers_exceeds_one() -> None:
    results = map_in_order(_pid, list(range(8)), max_workers=2)

    assert os.getpid() not in {pid for _, pid in results}


def test_map_in_order_handles_an_empty_item_list() -> None:
    assert map_in_order(_double, [], max_workers=4) == []


def test_default_max_workers_is_a_positive_worker_count() -> None:
    assert default_max_workers() >= 1


def test_map_in_order_chunks_work_so_a_shared_payload_is_not_repickled_per_item() -> None:
    """Per-model jobs close over the same corpus, so chunking keeps IPC off the critical path."""
    assert chunksize_for(n_items=1000, max_workers=4) > 1


def test_chunksize_is_one_when_there_is_at_most_one_item_per_worker() -> None:
    assert chunksize_for(n_items=3, max_workers=4) == 1


def test_chunksize_is_never_zero() -> None:
    assert chunksize_for(n_items=0, max_workers=4) == 1


def test_default_worker_count_does_not_oversubscribe_the_machine() -> None:
    """Scoring is compute-bound, so one worker per core is the most the machine can run."""
    #: Safe only because the pool pins each worker's BLAS to a single thread before spawning.
    assert 1 <= default_max_workers() <= (os.cpu_count() or 1)


def test_workers_start_fresh_rather_than_forking_the_parent() -> None:
    """A forked worker inherits the loaded corpus and copies it page by page, so spawn is used."""
    from core.parallel import WORKER_CONTEXT

    assert WORKER_CONTEXT.get_start_method() == "spawn"


def test_a_worker_serves_more_than_one_item() -> None:
    """Spawning an interpreter per item cost more than the arrays a scored item leaves behind."""
    pids = {pid for _, pid in map_in_order(_pid, list(range(64)), max_workers=2)}
    assert len(pids) == 2
    assert os.getpid() not in pids


def test_progress_milestones_mark_every_twentieth_and_the_end() -> None:
    assert progress_milestones(0) == set()
    assert progress_milestones(1) == {1}
    assert 100 in progress_milestones(100)
    assert len(progress_milestones(1000)) == 20


def test_report_progress_yields_every_result_and_writes_a_line_per_milestone() -> None:
    stream = io.StringIO()

    results = list(report_progress(iter(range(40)), 40, "models", stream=stream))

    lines = stream.getvalue().splitlines()
    assert results == list(range(40))
    assert len(lines) == 20
    assert lines[0].startswith("progress models: 2/40 in ")
    assert lines[-1].startswith("progress models: 40/40 in ")


def test_map_in_order_reports_progress_for_a_pooled_run(capsys) -> None:
    map_in_order(_double, list(range(8)), max_workers=2, label="doubles")

    assert "progress doubles: 8/8" in capsys.readouterr().err
