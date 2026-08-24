import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np


def is_diagonal(matrix, tol=1e-12):
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("Input should be a square matrix")
    return np.allclose(matrix, np.diag(np.diag(matrix)), atol=tol)


# ---------------------------------------------------------------------------
# Parallel mini-batch support
# ---------------------------------------------------------------------------

def resolve_n_jobs(n_jobs):
    """Map an ``n_jobs`` argument onto a concrete positive worker count.

    ``n_jobs=1`` (or ``None``) means fully sequential execution.
    ``n_jobs=-1`` uses every CPU core, ``-2`` all but one, and so on.
    """
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer (-1 means all cores).")
    if n_jobs < 0:
        n_cpu = os.cpu_count() or 1
        n_jobs = n_cpu + 1 + n_jobs
        if n_jobs < 1:
            raise ValueError(
                f"n_jobs is more negative than the number of cores ({n_cpu}); "
                "nothing left to run on."
            )
    return n_jobs


def chunk_sizes(total, n_chunks):
    """Split ``total`` samples into at most ``n_chunks`` balanced positive chunks."""
    total = int(total)
    if total <= 0:
        raise ValueError("Number of samples must be positive.")
    n_chunks = max(1, min(int(n_chunks), total))
    base, remainder = divmod(total, n_chunks)
    return [base + 1] * remainder + [base] * (n_chunks - remainder)


def _init_worker():
    """Re-seed NumPy's global RNG inside a freshly created worker process.

    Forked workers inherit the parent's RNG state, which would make every
    worker draw the *same* perturbations. Each process therefore reseeds itself
    from the OS entropy pool.
    """
    np.random.seed(int.from_bytes(os.urandom(4), "little"))


class ParallelBatchMixin:
    """Reusable worker pool for evaluating a mini-batch of oracle samples.

    The ``t`` samples that make up one mini-batch are independent, so they can
    be spread over several workers. Samples are grouped into one chunk per
    worker (rather than one task per sample) so scheduling overhead stays
    negligible next to the cost function itself.

    Backends
    --------
    "thread"  : default. Works with any callable -- including lambdas, closures
                and bound methods -- and shares memory with the caller. Gives a
                real speed-up whenever the cost function releases the GIL, which
                covers the usual zeroth-order use cases: NumPy-heavy models,
                compiled extensions, and subprocess- or network-backed
                simulators.
    "process" : true parallelism for cost functions that are pure Python and
                CPU-bound. The cost function must be picklable (a module-level
                function, not a lambda), and results are no longer reproducible
                from ``np.random.seed`` because each worker seeds itself.
    """

    _BACKENDS = ("thread", "process")

    def _init_parallel(self, n_jobs=1, backend="thread"):
        if backend not in self._BACKENDS:
            raise ValueError(
                f"backend must be one of {self._BACKENDS}, got {backend!r}."
            )
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.backend = backend
        self._pool = None

    def _get_pool(self):
        """Return the shared executor, creating it on first use, or None."""
        if self.n_jobs <= 1:
            return None
        if self._pool is None:
            if self.backend == "thread":
                self._pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            else:
                self._pool = ProcessPoolExecutor(max_workers=self.n_jobs,
                                                 initializer=_init_worker)
        return self._pool

    def _map_chunks(self, fn, total, *args):
        """Evaluate ``fn(n, *args)`` over ``total`` samples, split across workers.

        ``fn`` must take a sample count as its first argument and return the
        partial result for that many samples. Returns the list of partial
        results for the caller to reduce.
        """
        pool = self._get_pool()
        if pool is None:
            return [fn(total, *args)]

        sizes = chunk_sizes(total, self.n_jobs)
        if len(sizes) == 1:
            return [fn(total, *args)]

        futures = [pool.submit(fn, n, *args) for n in sizes]
        return [future.result() for future in futures]

    # -- lifecycle ---------------------------------------------------------

    def close(self):
        """Shut down the worker pool. Safe to call more than once."""
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        """Drop the executor when pickling, so instances can cross to workers."""
        state = self.__dict__.copy()
        state["_pool"] = None
        return state
