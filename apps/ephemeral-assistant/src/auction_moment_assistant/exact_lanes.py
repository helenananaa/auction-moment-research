"""Run the exact sampler on worker processes ("lanes") and pool their results.

Each lane is an independent run with 1/L of the count vectors, worlds and ESS
target; its candidate pools and clue tables stay in the process across the
rounds of a game. Two kinds of run go on in the background without waiting:
``prewarm`` starts a run on partial evidence (to fill those caches while the
map is still being scanned), and ``refine`` reruns a decision that ran out of
time with a longer budget, collected with ``poll_refine``. The next
``prewarm``, ``refine`` or ``sample`` cancels either one first. Lanes keep
everything in memory.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from pathlib import Path

import numpy as np

BACKGROUND = ("prewarm", "refine")


class ExactLanesError(RuntimeError):
    pass


def _worker(conn, cancel_event, world_model_path: str, treasures_csv: str) -> None:
    try:
        from threadpoolctl import threadpool_limits

        # Many small matrix products: multi-threaded BLAS only adds overhead.
        _limits = threadpool_limits(1, "blas")  # noqa: F841
    except ImportError:
        pass
    from generative_world_model_v2 import GenerativeWorldModelV2

    from .exact_evidence import RowsTable, build_prior
    from .exact_sampler import Cancelled, sample_posterior
    from .models import load_catalog

    try:
        catalog = load_catalog(Path(treasures_csv))
        model = GenerativeWorldModelV2.load(Path(world_model_path), catalog)
        prior = build_prior(model, catalog)
        rows = RowsTable.packaged()
    except Exception as exc:  # reported to the parent, which fails the start-up
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", None))
    while True:
        message = conn.recv()
        if message is None:
            break
        kind, ev, seed, kwargs = message
        try:
            result = sample_posterior(
                prior, ev, np.random.default_rng(seed), rows_loglik=rows.log_likelihood,
                cancel=cancel_event.is_set if kind in BACKGROUND else None, **kwargs,
            )
            conn.send(("ok", None if kind == "prewarm" else result))
        except Cancelled:
            conn.send(("cancelled", None))
        except Exception as exc:  # keep the lane alive; the caller decides
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class ExactLanes:
    def __init__(self, world_model_path: Path, treasures_csv: Path, *, lanes: int = 4,
                 start_timeout: float = 180.0) -> None:
        if int(lanes) < 1:
            raise ValueError("lanes must be at least 1")
        context = mp.get_context("spawn")
        self._conns, self._procs, self._cancel = [], [], []
        self._lock = threading.Lock()
        self._background = None
        self._replies: list = []
        self._refine_target = 0.0
        self.last_prewarm_wait = 0.0
        for _ in range(int(lanes)):
            parent, child = context.Pipe()
            cancel = context.Event()
            process = context.Process(
                target=_worker, args=(child, cancel, str(world_model_path), str(treasures_csv)), daemon=True,
            )
            process.start()
            self._conns.append(parent)
            self._procs.append(process)
            self._cancel.append(cancel)
        for conn in self._conns:
            if not conn.poll(start_timeout):
                self.close()
                raise ExactLanesError("exact sampler lane did not start in time")
            kind, message = conn.recv()
            if kind != "ready":
                self.close()
                raise ExactLanesError(f"exact sampler lane failed to start: {message}")

    @property
    def lanes(self) -> int:
        return len(self._conns)

    def _kwargs(self, time_budget, n_vectors, worlds, min_ess, min_ess_size_avg):
        lanes = self.lanes
        return dict(
            n_vectors=max(1, n_vectors // lanes), worlds=max(1, worlds // lanes), min_ess=min_ess / lanes,
            min_ess_size_avg=min_ess_size_avg / lanes,
            time_budget=None if time_budget is None else max(0.5, float(time_budget) - 0.3),
        )

    def _send(self, kind: str, ev, seed: int, kwargs: dict) -> None:
        seeds = np.random.SeedSequence(int(seed)).spawn(self.lanes)
        for conn, sequence in zip(self._conns, seeds):
            conn.send((kind, ev, int(sequence.generate_state(1, np.uint64)[0]), kwargs))
        self._replies = [None] * self.lanes

    def _finish_background(self) -> None:
        """Cancel a prewarm or refinement and drain its replies."""
        if self._background is None:
            return
        for cancel in self._cancel:
            cancel.set()
        for index, conn in enumerate(self._conns):
            if self._replies[index] is None:
                self._replies[index] = conn.recv()
        for cancel in self._cancel:
            cancel.clear()
        self._background = None

    def _start_background(self, kind: str, ev, seed: int, kwargs: dict, target: float = 0.0) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._finish_background()
            self._send(kind, ev, seed, kwargs)
            self._background = kind
            self._refine_target = target
            return True
        finally:
            self._lock.release()

    def prewarm(self, ev, seed: int, *, time_budget: float = 15.0, n_vectors: int = 10000,
                worlds: int = 25_000, min_ess: float = 500.0, min_ess_size_avg: float = 1500.0) -> bool:
        """Start filling caches without waiting; False if the lanes are busy."""
        return self._start_background(
            "prewarm", ev, seed, self._kwargs(time_budget, n_vectors, worlds, min_ess, min_ess_size_avg),
        )

    def refine(self, ev, seed: int, *, time_budget: float = 45.0, n_vectors: int = 10000,
               worlds: int = 25_000, min_ess: float = 500.0, min_ess_size_avg: float = 1500.0) -> bool:
        """Start a longer run of a decision without waiting; False if the lanes are busy."""
        return self._start_background(
            "refine", ev, seed, self._kwargs(time_budget, n_vectors, worlds, min_ess, min_ess_size_avg),
            target=min_ess_size_avg if ev.size_avg_value else min_ess,
        )

    def poll_refine(self):
        """("running", None), ("done", merged result) or ("gone", reason): never blocks."""
        from .exact_sampler import merge_results

        if not self._lock.acquire(blocking=False):
            return "gone", "lanes_busy"
        try:
            if self._background != "refine":
                return "gone", "not_refining"
            for index, conn in enumerate(self._conns):
                if self._replies[index] is None and conn.poll():
                    self._replies[index] = conn.recv()
            if any(reply is None for reply in self._replies):
                return "running", None
            self._background = None
            failures = [f"{kind}: {message}" for kind, message in self._replies if kind != "ok"]
            if failures:
                return "gone", "; ".join(failures)
            return "done", merge_results([result for _, result in self._replies], self._refine_target)
        finally:
            self._lock.release()

    def sample(self, ev, seed: int, *, time_budget: float | None = 8.0, n_vectors: int = 10000,
               worlds: int = 25_000, min_ess: float = 500.0, min_ess_size_avg: float = 1500.0):
        import time

        from .exact_sampler import merge_results

        with self._lock:
            started = time.monotonic()
            self._finish_background()
            self.last_prewarm_wait = time.monotonic() - started
            if time_budget is not None:
                # waiting for a background run to stop counts against the budget
                time_budget = float(time_budget) - self.last_prewarm_wait
            self._send("sample", ev, seed, self._kwargs(time_budget, n_vectors, worlds, min_ess, min_ess_size_avg))
            replies = [conn.recv() for conn in self._conns]
        errors = [message for kind, message in replies if kind != "ok"]
        if errors:
            raise ExactLanesError("; ".join(str(e) for e in errors))
        target = min_ess_size_avg if ev.size_avg_value else min_ess
        return merge_results([result for _, result in replies], target)

    def close(self) -> None:
        for cancel in self._cancel:
            cancel.set()
        for conn in self._conns:
            try:
                conn.send(None)
            except (OSError, ValueError):
                pass
        for process in self._procs:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
