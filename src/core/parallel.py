"""Process-parallel execution of independent per-seed dataset builds."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from functools import partial
from typing import Any

#: Fresh interpreters per worker: a forked worker copies the parent's loaded corpus page by page.
WORKER_CONTEXT = multiprocessing.get_context("spawn")

#: Every backend numpy might link, since each reads only its own variable.
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    #: numpy links Accelerate on this platform, which ignores the OpenMP variable entirely.
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_worker_blas_threads() -> None:
    """Gives each worker a single BLAS thread, leaving an operator's explicit setting alone."""
    for variable in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")


def default_max_workers() -> int:
    """One worker per core, which only pays once each worker's BLAS is pinned to one thread."""
    return os.cpu_count() or 1


def chunksize_for(n_items: int, max_workers: int) -> int:
    """Items per task: enough chunks to balance load, few enough to stop repickling the payload."""
    return max(1, n_items // (max_workers * 4))


def map_in_order[ItemT, ResultT](
    fn: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    max_workers: int | None = None,
) -> list[ResultT]:
    """Applies fn to every item, returning results in submission order so reruns stay comparable."""
    workers = default_max_workers() if max_workers is None else max_workers
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    #: Set before the pool spawns, because a worker reads these only as it imports numpy.
    pin_worker_blas_threads()
    #: One item per worker lifetime, so memory a scored file left behind returns to the OS.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=WORKER_CONTEXT, max_tasks_per_child=1
    ) as pool:
        return list(pool.map(fn, items, chunksize=chunksize_for(len(items), workers)))


def in_worker_process() -> bool:
    """True when this process is itself a pool worker, whose parent already claimed the cores."""
    return multiprocessing.parent_process() is not None


def map_items[ContextT, ItemT, ResultT](
    worker: Callable[[ContextT, ItemT], ResultT],
    context: ContextT,
    items: Iterable[ItemT],
    *,
    max_workers: int | None = None,
    executor_factory: Callable[..., AbstractContextManager[Any]] = ProcessPoolExecutor,
    in_worker_process: Callable[[], bool] = in_worker_process,
) -> list[ResultT]:
    """Runs `worker(context, item)` for every item across processes, results in submission order."""
    #: `worker` must be module-level: the pool pickles it by qualified name to reach each process.
    item_list = list(items)
    if not item_list:
        return []
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(requested, len(item_list)))
    #: Nesting would square the process count and exhaust memory, so an inner call stays serial.
    if workers == 1 or in_worker_process():
        return [worker(context, item) for item in item_list]
    chunksize = -(-len(item_list) // workers)
    with executor_factory(max_workers=workers, mp_context=WORKER_CONTEXT) as pool:
        return list(pool.map(partial(worker, context), item_list, chunksize=chunksize))


def map_seeds[ContextT, ResultT](
    worker: Callable[[ContextT, int], ResultT],
    context: ContextT,
    seeds: Iterable[int],
    *,
    max_workers: int | None = None,
    executor_factory: Callable[..., AbstractContextManager[Any]] = ProcessPoolExecutor,
    in_worker_process: Callable[[], bool] = in_worker_process,
) -> list[ResultT]:
    """Runs `worker(context, seed)` for every seed across processes, results in seed order."""
    return map_items(
        worker,
        context,
        seeds,
        max_workers=max_workers,
        executor_factory=executor_factory,
        in_worker_process=in_worker_process,
    )


def map_constructions[ContextT, ItemT](
    worker: Callable[[ContextT, ItemT], str | None],
    context: ContextT,
    constructions: Iterable[ItemT],
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Builds every construction across workers, returning the names written, skipping any None."""
    #: A worker returns None for a construction already on disk, which is a skip rather than a name.
    names = map_items(worker, context, constructions, max_workers=max_workers)
    return [name for name in names if name is not None]
