"""
Tests for ZoSolvers.minimisation.ZO_gauss_min.

Coverage:
  - Initialisation (attributes, B caching, error handling)
  - Sampling (_sample_gaussian, _sample_sphere, _sample dispatch)
  - Oracle (shape, methods, unbiasedness for both oracle types and B structures)
  - Mini-batch (batch_oracle mean, variance reduction with t)
  - Parallelism (n_jobs resolution, chunking, thread/process backends,
                 pool lifecycle, equivalence with sequential execution)
  - ZOGD  (convergence, projection, project_init, early stopping,
           t="iteration", small N)
  - ZOEGm (convergence, projection, project_init, gamma, t="iteration",
           small N)
"""

import threading
import time

import numpy as np
import pytest
from ZoSolvers.minimisation import ZO_gauss_min
from ZoSolvers.utils import chunk_sizes, resolve_n_jobs


# ---------------------------------------------------------------------------
# Shared test problems
# ---------------------------------------------------------------------------

def f_quadratic(x):
    """f(x) = ||x||^2 / 2,  grad f(x) = x."""
    return 0.5 * float(np.dot(x, x))


def proj_box(x, lo=-3.0, hi=3.0):
    return np.clip(x, lo, hi)


def f_slow(x):
    """Quadratic that sleeps, so concurrent workers actually overlap.

    Must stay at module level to remain picklable for the process backend.
    """
    time.sleep(0.002)
    return f_quadratic(x)


def _make_solver(d=2, oracle_type="gaussian", B=None, proj=None, **overrides):
    x0 = np.full(d, 5.0)
    params = dict(h=0.1, mu=1e-8, N=2000, t=10,
                  B=B, proj=proj, oracle_type=oracle_type)
    params.update(overrides)
    return ZO_gauss_min(f_quadratic, x0, **params)


def _oracle_mean(opt, x, method, n=20_000, seed=42):
    """Empirical mean of n oracle evaluations at x."""
    np.random.seed(seed)
    return sum(opt.oracle(x, method) for _ in range(n)) / n


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_attributes(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(3))
        assert opt.d == 3
        assert opt.oracle_type == "gaussian"
        assert opt._B_structure is None
        assert opt._L is None
        assert opt._inv_sqrt_B_diag is None

    @pytest.mark.parametrize("d", [1, 3, 10])
    def test_d_inferred_from_x0(self, d):
        opt = ZO_gauss_min(f_quadratic, np.zeros(d))
        assert opt.d == d

    def test_x0_list_converted_to_float_array(self):
        opt = ZO_gauss_min(f_quadratic, [1, 2, 3])
        assert opt.x0.dtype == float
        assert opt.d == 3

    def test_diagonal_B_cache(self):
        B = np.diag([1.0, 4.0, 9.0])
        opt = ZO_gauss_min(f_quadratic, np.zeros(3), B=B)
        assert opt._B_structure == 'diagonal'
        np.testing.assert_allclose(opt._inv_sqrt_B_diag, [1.0, 0.5, 1.0 / 3])
        assert opt._L is None

    def test_full_B_cache_cholesky_correct(self):
        B = np.array([[3.0, 1.0], [1.0, 2.0]])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), B=B)
        assert opt._B_structure == 'full'
        assert opt._L is not None
        # L must satisfy L L^T = B^{-1}
        np.testing.assert_allclose(opt._L @ opt._L.T, np.linalg.inv(B), atol=1e-12)

    def test_invalid_oracle_type_raises(self):
        with pytest.raises(ValueError, match="oracle_type"):
            ZO_gauss_min(f_quadratic, np.zeros(2), oracle_type="banana")

    def test_non_square_B_raises(self):
        with pytest.raises(RuntimeError):
            ZO_gauss_min(f_quadratic, np.zeros(2), B=np.ones((2, 3)))

    def test_sphere_oracle_type_stored(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), oracle_type="sphere")
        assert opt.oracle_type == "sphere"

    def test_project_init_defaults_to_false(self):
        assert ZO_gauss_min(f_quadratic, np.zeros(2)).project_init is False
        assert ZO_gauss_min(f_quadratic, np.zeros(2),
                            project_init=True).project_init is True


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    @pytest.fixture(autouse=True)
    def seed(self):
        np.random.seed(0)

    def test_gaussian_no_B_shape(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(5))
        assert opt._sample_gaussian().shape == (5,)

    def test_gaussian_diag_B_shape(self):
        B = np.diag([2.0, 3.0, 0.5])
        opt = ZO_gauss_min(f_quadratic, np.zeros(3), B=B)
        assert opt._sample_gaussian().shape == (3,)

    def test_gaussian_full_B_shape(self):
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), B=B)
        assert opt._sample_gaussian().shape == (2,)

    def test_sphere_no_B_on_unit_sphere(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(6), oracle_type="sphere")
        for _ in range(50):
            u = opt._sample_sphere()
            assert u.shape == (6,)
            np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-12)

    def test_sphere_diag_B_shape(self):
        B = np.diag([2.0, 3.0])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), B=B, oracle_type="sphere")
        assert opt._sample_sphere().shape == (2,)

    def test_sphere_full_B_shape(self):
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), B=B, oracle_type="sphere")
        assert opt._sample_sphere().shape == (2,)

    def test_sample_dispatches_to_gaussian(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(3), oracle_type="gaussian")
        np.random.seed(7)
        u1 = opt._sample()
        np.random.seed(7)
        u2 = opt._sample_gaussian()
        np.testing.assert_array_equal(u1, u2)

    def test_sample_dispatches_to_sphere(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(3), oracle_type="sphere")
        np.random.seed(7)
        u1 = opt._sample()
        np.random.seed(7)
        u2 = opt._sample_sphere()
        np.testing.assert_array_equal(u1, u2)

    def test_gaussian_and_sphere_differ(self):
        """Gaussian and sphere samples drawn from the same seed are different
        because sphere normalises the vector."""
        opt_g = ZO_gauss_min(f_quadratic, np.zeros(4), oracle_type="gaussian")
        opt_s = ZO_gauss_min(f_quadratic, np.zeros(4), oracle_type="sphere")
        np.random.seed(99)
        ug = opt_g._sample()
        np.random.seed(99)
        us = opt_s._sample()
        assert not np.allclose(ug, us)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class TestOracle:
    def test_output_shape(self):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.zeros(4))
        g = opt.oracle(np.ones(4), "center")
        assert g.shape == (4,)

    def test_invalid_method_raises(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2))
        with pytest.raises(ValueError, match="Unknown method"):
            opt.oracle(np.ones(2), "bad_method")

    @pytest.mark.parametrize("method", ["forw", "back", "center"])
    def test_all_methods_return_correct_shape(self, method):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.zeros(3), mu=1e-5)
        assert opt.oracle(np.ones(3), method).shape == (3,)

    # --- Unbiasedness: E[oracle(x, "center")] ≈ grad f(x) = x ---------------

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_unbiased_no_B(self, oracle_type):
        x = np.array([2.0, -1.0])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, oracle_type=oracle_type)
        est = _oracle_mean(opt, x, "center")
        np.testing.assert_allclose(est, x, atol=0.1)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_unbiased_diagonal_B(self, oracle_type):
        x = np.array([1.0, 3.0])
        B = np.diag([2.0, 0.5])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, B=B, oracle_type=oracle_type)
        est = _oracle_mean(opt, x, "center")
        np.testing.assert_allclose(est, x, atol=0.15)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_unbiased_full_B(self, oracle_type):
        x = np.array([1.0, -2.0])
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, B=B, oracle_type=oracle_type)
        est = _oracle_mean(opt, x, "center")
        np.testing.assert_allclose(est, x, atol=0.2)

    def test_forward_oracle_has_small_bias_for_quadratic(self):
        """Forward-difference oracle bias is O(mu) for quadratics."""
        x = np.array([1.0])
        opt = ZO_gauss_min(f_quadratic, np.zeros(1), mu=1e-4)
        est = _oracle_mean(opt, x, "forw", n=20_000)
        # Bias from forward difference is O(mu * ||Hess||); for f=x^2/2, bias~mu/2
        np.testing.assert_allclose(est, x, atol=0.1)


# ---------------------------------------------------------------------------
# ZOGD
# ---------------------------------------------------------------------------

class TestZOGD:
    def _converged(self, x):
        """True if f decreased by at least 99%."""
        return f_quadratic(x[-1]) < f_quadratic(x[0]) * 0.01

    def test_returns_ndarray(self):
        np.random.seed(42)
        x = _make_solver(N=50, t=1).ZOGD()
        assert isinstance(x, np.ndarray)

    def test_output_has_correct_number_of_columns(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, -5.0]), h=0.1, mu=1e-8, N=50, t=1)
        x = opt.ZOGD()
        assert x.shape[1] == 2

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_no_B(self, oracle_type):
        np.random.seed(42)
        x = _make_solver(d=2, oracle_type=oracle_type, N=3000, t=15).ZOGD(method="center")
        assert self._converged(x)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_diagonal_B(self, oracle_type):
        np.random.seed(42)
        B = np.diag([2.0, 0.5])
        x = _make_solver(d=2, oracle_type=oracle_type, B=B, N=3000, t=15).ZOGD(method="center")
        assert self._converged(x)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_full_B(self, oracle_type):
        np.random.seed(42)
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        x = _make_solver(d=2, oracle_type=oracle_type, B=B, N=3000, t=15).ZOGD(method="center")
        assert self._converged(x)

    def test_projection_keeps_iterates_in_feasible_set(self):
        """Every computed iterate is feasible. Row 0 is the caller's x0, which
        is returned unprojected unless project_init=True."""
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=1.0, mu=1e-8, N=200, t=1, proj=proj_box)
        x = opt.ZOGD()
        assert np.all(x[1:] >= -3.0 - 1e-10) and np.all(x[1:] <= 3.0 + 1e-10)

    def test_x0_returned_unprojected_by_default(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=1.0, mu=1e-8, N=20, t=1, proj=proj_box)
        assert opt.project_init is False
        np.testing.assert_array_equal(opt.ZOGD()[0], [5.0, 5.0])

    def test_project_init_makes_every_iterate_feasible(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=1.0, mu=1e-8, N=200, t=1, proj=proj_box,
                           project_init=True)
        x = opt.ZOGD()
        np.testing.assert_array_equal(x[0], [3.0, 3.0])
        assert np.all(x >= -3.0 - 1e-10) and np.all(x <= 3.0 + 1e-10)

    def test_project_init_is_noop_without_proj(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=0.1, mu=1e-8, N=20, t=1, project_init=True)
        np.testing.assert_array_equal(opt.ZOGD()[0], [5.0, 5.0])

    def test_project_init_does_not_mutate_x0(self):
        np.random.seed(42)
        x0 = np.array([5.0, 5.0])
        opt = ZO_gauss_min(f_quadratic, x0, h=1.0, mu=1e-8, N=20, t=1,
                           proj=proj_box, project_init=True)
        opt.ZOGD()
        np.testing.assert_array_equal(opt.x0, [5.0, 5.0])

    def test_tol_f_triggers_early_stop(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0]),
                           h=0.1, mu=1e-8, N=5000, t=20, tol_f=1e-3)
        x = opt.ZOGD(method="center")
        assert len(x) < 5000

    def test_tol_g_triggers_early_stop(self):
        np.random.seed(42)
        # Very large tol_g so the check fires immediately after the first period
        opt = ZO_gauss_min(f_quadratic, np.array([5.0]),
                           h=0.1, mu=1e-8, N=5000, t=20, tol_g=1e10)
        x = opt.ZOGD(method="center")
        assert len(x) < 5000

    def test_t_iteration_mode_runs(self):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=0.05, mu=1e-8, N=300, t="iteration")
        x = opt.ZOGD(method="center")
        assert x.shape[1] == 2

    @pytest.mark.parametrize("method", ["forw", "back", "center"])
    def test_all_finite_difference_methods_run(self, method):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.array([3.0, -3.0]),
                           h=0.05, mu=1e-6, N=30, t=5)
        x = opt.ZOGD(method=method)
        assert x.shape[1] == 2

    def test_small_N_no_division_by_zero(self):
        """N < 50 previously caused ZeroDivisionError in _check_period."""
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.array([1.0]),
                           h=0.1, mu=1e-5, N=5, t=1, tol_f=0.01)
        x = opt.ZOGD()
        assert len(x) <= 5


# ---------------------------------------------------------------------------
# ZOEGm
# ---------------------------------------------------------------------------

class TestZOEGm:
    def _converged(self, x):
        return f_quadratic(x[-1]) < f_quadratic(x[0]) * 0.01

    def test_returns_ndarray(self):
        np.random.seed(42)
        x = _make_solver(N=50, t=1).ZOEGm()
        assert isinstance(x, np.ndarray)

    def test_output_has_correct_number_of_columns(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, -5.0]),
                           h=0.1, mu=1e-8, N=50, t=1)
        x = opt.ZOEGm()
        assert x.shape[1] == 2

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_no_B(self, oracle_type):
        np.random.seed(42)
        x = _make_solver(d=2, oracle_type=oracle_type, N=3000, t=15).ZOEGm(method="center")
        assert self._converged(x)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_diagonal_B(self, oracle_type):
        np.random.seed(42)
        B = np.diag([2.0, 0.5])
        x = _make_solver(d=2, oracle_type=oracle_type, B=B, N=3000, t=15).ZOEGm(method="center")
        assert self._converged(x)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_full_B(self, oracle_type):
        np.random.seed(42)
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        x = _make_solver(d=2, oracle_type=oracle_type, B=B, N=3000, t=15).ZOEGm(method="center")
        assert self._converged(x)

    def test_projection_keeps_iterates_in_feasible_set(self):
        """Row 0 is the caller's x0, returned unprojected unless project_init."""
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=0.5, mu=1e-8, N=200, t=1, proj=proj_box)
        x = opt.ZOEGm()
        assert np.all(x[1:] >= -3.0 - 1e-10) and np.all(x[1:] <= 3.0 + 1e-10)

    def test_project_init_makes_every_iterate_feasible(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=0.5, mu=1e-8, N=200, t=1, proj=proj_box,
                           project_init=True)
        x = opt.ZOEGm()
        np.testing.assert_array_equal(x[0], [3.0, 3.0])
        assert np.all(x >= -3.0 - 1e-10) and np.all(x <= 3.0 + 1e-10)

    def test_intermediate_point_is_projected(self):
        """xhat must be projected before it is used for the second stride."""
        np.random.seed(0)
        seen = []
        opt = ZO_gauss_min(lambda v: (seen.append(np.array(v)), f_quadratic(v))[1],
                           np.array([5.0, 5.0]), h=5.0, mu=1e-8, N=3, t=1,
                           proj=proj_box, project_init=True)
        opt.ZOEGm()
        # every point the objective was queried at sits within mu of the box
        assert all(np.all(np.abs(v) <= 3.0 + 1e-6) for v in seen)

    def test_tol_f_triggers_early_stop(self):
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0]),
                           h=0.1, mu=1e-8, N=5000, t=20, tol_f=1e-3)
        x = opt.ZOEGm(method="center")
        assert len(x) < 5000

    def test_gamma_less_than_one_still_converges(self):
        np.random.seed(42)
        x = _make_solver(d=1, N=3000, t=15).ZOEGm(method="center", gamma=0.5)
        assert self._converged(x)

    def test_t_iteration_mode_runs(self):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=0.05, mu=1e-8, N=300, t="iteration")
        x = opt.ZOEGm(method="center")
        assert x.shape[1] == 2

    @pytest.mark.parametrize("method", ["forw", "back", "center"])
    def test_all_finite_difference_methods_run(self, method):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.array([3.0, -3.0]),
                           h=0.05, mu=1e-6, N=30, t=5)
        x = opt.ZOEGm(method=method)
        assert x.shape[1] == 2

    def test_small_N_no_division_by_zero(self):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.array([1.0]),
                           h=0.1, mu=1e-5, N=5, t=1)
        x = opt.ZOEGm()
        assert len(x) <= 5


# ---------------------------------------------------------------------------
# Mini-batch
# ---------------------------------------------------------------------------

class TestBatchOracle:
    """The t samples of one step are independent draws that are averaged."""

    def test_batch_oracle_shape(self):
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.zeros(3), mu=1e-8, t=8)
        assert opt.batch_oracle(np.ones(3), "center", 8).shape == (3,)

    def test_batch_oracle_is_unbiased(self):
        np.random.seed(0)
        x = np.array([2.0, -1.0])
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=4000)
        np.testing.assert_allclose(opt.batch_oracle(x, "center", 4000), x, atol=0.15)

    def test_larger_t_reduces_error(self):
        """Averaging more samples must shrink the error towards grad f(x) = x."""
        x = np.array([2.0, -1.0])
        errors = []
        for t in (1, 16, 256):
            opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=t)
            np.random.seed(0)
            err = np.mean([np.linalg.norm(opt.batch_oracle(x, "center", t) - x)
                           for _ in range(30)])
            errors.append(err)
        assert errors[0] > errors[1] > errors[2]

    def test_samples_within_a_batch_are_independent(self):
        """Two draws of one batch must not be the same vector."""
        np.random.seed(0)
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=2)
        draws = [opt.oracle(np.array([2.0, -1.0]), "center") for _ in range(5)]
        assert not any(np.array_equal(draws[0], d) for d in draws[1:])

    def test_num_samples_respects_t(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), t=7)
        assert opt._num_samples(3) == 7
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), t="iteration")
        assert opt._num_samples(3) == 3


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------

class TestParallelHelpers:
    def test_default_is_sequential(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2))
        assert opt.n_jobs == 1
        assert opt.backend == "thread"

    def test_resolve_n_jobs_positive(self):
        assert resolve_n_jobs(1) == 1
        assert resolve_n_jobs(4) == 4
        assert resolve_n_jobs(None) == 1

    def test_resolve_n_jobs_negative_counts_back_from_all_cores(self):
        import os
        n_cpu = os.cpu_count() or 1
        assert resolve_n_jobs(-1) == n_cpu
        if n_cpu > 1:
            assert resolve_n_jobs(-2) == n_cpu - 1

    def test_resolve_n_jobs_zero_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            resolve_n_jobs(0)

    @pytest.mark.parametrize("total,n_chunks,expected_sum", [(10, 4, 10), (3, 8, 3), (1, 4, 1)])
    def test_chunk_sizes_partition_exactly(self, total, n_chunks, expected_sum):
        sizes = chunk_sizes(total, n_chunks)
        assert sum(sizes) == expected_sum
        assert all(n > 0 for n in sizes)
        assert len(sizes) <= min(n_chunks, total)

    def test_chunk_sizes_are_balanced(self):
        sizes = chunk_sizes(10, 4)
        assert max(sizes) - min(sizes) <= 1

    def test_invalid_n_jobs_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            ZO_gauss_min(f_quadratic, np.zeros(2), n_jobs=0)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            ZO_gauss_min(f_quadratic, np.zeros(2), backend="dask")


class TestParallelExecution:
    def test_pool_is_created_lazily(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=4, n_jobs=2)
        assert opt._pool is None                     # nothing spawned yet
        opt.batch_oracle(np.ones(2), "center", 4)
        assert opt._pool is not None
        opt.close()

    def test_sequential_solver_never_creates_a_pool(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), h=0.1, mu=1e-8, N=20, t=4)
        opt.ZOGD("center")
        assert opt._pool is None

    def test_pool_is_reused_across_iterations(self):
        opt = ZO_gauss_min(f_quadratic, np.full(2, 3.0), h=0.1, mu=1e-8,
                           N=10, t=4, n_jobs=2)
        opt.batch_oracle(np.ones(2), "center", 4)
        first = opt._pool
        opt.ZOGD("center")
        assert opt._pool is first                    # same executor, not respawned
        opt.close()

    def test_close_is_idempotent(self):
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=4, n_jobs=2)
        opt.batch_oracle(np.ones(2), "center", 4)
        opt.close()
        opt.close()
        assert opt._pool is None

    def test_context_manager_closes_pool(self):
        with ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=4, n_jobs=2) as opt:
            opt.batch_oracle(np.ones(2), "center", 4)
            assert opt._pool is not None
        assert opt._pool is None

    def test_worker_threads_are_released_on_close(self):
        before = threading.active_count()
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=8, n_jobs=4)
        opt.batch_oracle(np.ones(2), "center", 8)
        opt.close()
        assert threading.active_count() == before

    def test_batch_really_runs_on_several_threads(self):
        """The samples of one batch are spread over distinct worker threads."""
        seen = set()

        def f_tracking(x):
            seen.add(threading.get_ident())
            time.sleep(0.002)
            return f_quadratic(x)

        np.random.seed(0)
        with ZO_gauss_min(f_tracking, np.zeros(2), mu=1e-8, t=8, n_jobs=4) as opt:
            opt.batch_oracle(np.ones(2), "center", 8)
        assert len(seen) > 1

    def test_sequential_batch_runs_on_one_thread(self):
        seen = set()

        def f_tracking(x):
            seen.add(threading.get_ident())
            return f_quadratic(x)

        np.random.seed(0)
        opt = ZO_gauss_min(f_tracking, np.zeros(2), mu=1e-8, t=8, n_jobs=1)
        opt.batch_oracle(np.ones(2), "center", 8)
        assert seen == {threading.get_ident()}

    def test_every_sample_is_evaluated_exactly_once(self):
        """Chunking must not drop or duplicate samples."""
        counter = {"n": 0}
        lock = threading.Lock()

        def f_counting(x):
            with lock:
                counter["n"] += 1
            return f_quadratic(x)

        np.random.seed(0)
        with ZO_gauss_min(f_counting, np.zeros(2), mu=1e-8, t=10, n_jobs=4) as opt:
            opt.batch_oracle(np.ones(2), "center", 10)
        assert counter["n"] == 20            # 10 samples x 2 evals (centered)

    def test_parallel_and_sequential_agree(self):
        """Both estimate the same gradient; only the summation order differs."""
        x = np.array([2.0, -1.0])
        np.random.seed(11)
        seq = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8,
                           t=3000).batch_oracle(x, "center", 3000)
        np.random.seed(11)
        with ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=3000,
                          n_jobs=4) as opt:
            par = opt.batch_oracle(x, "center", 3000)
        np.testing.assert_allclose(seq, x, atol=0.2)
        np.testing.assert_allclose(par, x, atol=0.2)

    def test_more_workers_than_samples(self):
        np.random.seed(0)
        with ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=3, n_jobs=8) as opt:
            assert opt.batch_oracle(np.ones(2), "center", 3).shape == (2,)

    def test_single_sample_with_many_workers(self):
        np.random.seed(0)
        with ZO_gauss_min(f_quadratic, np.full(2, 3.0), h=0.1, mu=1e-8,
                          N=20, t=1, n_jobs=4) as opt:
            assert opt.ZOGD("center").shape == (20, 2)

    @pytest.mark.parametrize("algorithm", ["ZOGD", "ZOEGm"])
    def test_solvers_converge_with_parallel_workers(self, algorithm):
        np.random.seed(42)
        with _make_solver(d=2, N=1500, t=8, n_jobs=4) as opt:
            x = getattr(opt, algorithm)(method="center")
        assert f_quadratic(x[-1]) < f_quadratic(x[0]) * 0.01

    def test_t_iteration_with_parallel_workers(self):
        np.random.seed(0)
        with ZO_gauss_min(f_quadratic, np.full(2, 5.0), h=0.05, mu=1e-8,
                          N=60, t="iteration", n_jobs=4) as opt:
            assert opt.ZOGD("center").shape == (60, 2)

    def test_tol_f_early_stop_with_parallel_workers(self):
        np.random.seed(42)
        with ZO_gauss_min(f_quadratic, np.array([5.0]), h=0.1, mu=1e-8,
                          N=5000, t=8, tol_f=1e-3, n_jobs=4) as opt:
            assert len(opt.ZOGD("center")) < 5000

    def test_tol_g_early_stop_with_parallel_workers(self):
        np.random.seed(42)
        with ZO_gauss_min(f_quadratic, np.array([5.0]), h=0.1, mu=1e-8,
                          N=5000, t=8, tol_g=1e10, n_jobs=4) as opt:
            assert len(opt.ZOGD("center")) < 5000

    def test_projection_still_holds_with_parallel_workers(self):
        np.random.seed(42)
        with ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]), h=1.0, mu=1e-8,
                          N=100, t=8, proj=proj_box, project_init=True,
                          n_jobs=4) as opt:
            x = opt.ZOGD("center")
        assert np.all(np.abs(x) <= 3.0 + 1e-10)

    def test_solver_is_picklable_without_its_pool(self):
        """Required so bound methods can be shipped to worker processes."""
        import pickle
        opt = ZO_gauss_min(f_quadratic, np.zeros(2), mu=1e-8, t=4, n_jobs=2)
        opt.batch_oracle(np.ones(2), "center", 4)
        assert opt._pool is not None
        clone = pickle.loads(pickle.dumps(opt))
        assert clone._pool is None
        assert clone.n_jobs == 2
        opt.close()


class TestProcessBackend:
    """The process backend needs a picklable func; f_slow lives at module level."""

    def test_process_backend_gives_an_unbiased_mean(self):
        x = np.array([2.0, -1.0])
        with ZO_gauss_min(f_slow, np.zeros(2), mu=1e-8, t=64,
                          n_jobs=2, backend="process") as opt:
            est = opt.batch_oracle(x, "center", 64)
        np.testing.assert_allclose(est, x, atol=1.0)

    def test_process_backend_runs_a_full_solver(self):
        with ZO_gauss_min(f_slow, np.full(2, 5.0), h=0.2, mu=1e-8, N=8, t=8,
                          n_jobs=2, backend="process") as opt:
            x = opt.ZOGD("center")
        assert x.shape == (8, 2)
        assert f_quadratic(x[-1]) < f_quadratic(x[0])
