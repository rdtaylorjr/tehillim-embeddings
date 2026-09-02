"""Process-parallel execution of independent per-seed dataset builds."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from functools import partial
from typing import Any


def in_worker_process() -> bool:
    """True when this process is itself a pool worker, whose parent already claimed the cores."""
    return multiprocessing.parent_process() is not None


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
    #: `worker` must be module-level: the pool pickles it by qualified name to reach each process.
    seed_list = list(seeds)
    if not seed_list:
        return []
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(requested, len(seed_list)))
    #: Nesting would square the process count and exhaust memory, so an inner call stays serial.
    if workers == 1 or in_worker_process():
        return [worker(context, seed) for seed in seed_list]
    chunksize = -(-len(seed_list) // workers)
    with executor_factory(max_workers=workers) as pool:
        return list(pool.map(partial(worker, context), seed_list, chunksize=chunksize))
