"""
Tests for ZoSolvers.minimisation.ZO_gauss_min.

Coverage:
  - Initialisation (attributes, B caching, error handling)
  - Sampling (_sample_gaussian, _sample_sphere, _sample dispatch)
  - Oracle (shape, methods, unbiasedness for both oracle types and B structures)
  - ZOGD  (convergence, projection, early stopping, t="iteration", small N)
  - ZOEGm (convergence, projection, gamma, t="iteration", small N)
"""

import numpy as np
import pytest
from ZoSolvers.minimisation import ZO_gauss_min


# ---------------------------------------------------------------------------
# Shared test problems
# ---------------------------------------------------------------------------

def f_quadratic(x):
    """f(x) = ||x||^2 / 2,  grad f(x) = x."""
    return 0.5 * float(np.dot(x, x))


def proj_box(x, lo=-3.0, hi=3.0):
    return np.clip(x, lo, hi)


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
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=1.0, mu=1e-8, N=200, t=1, proj=proj_box)
        x = opt.ZOGD()
        assert np.all(x >= -3.0 - 1e-10) and np.all(x <= 3.0 + 1e-10)

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
        np.random.seed(42)
        opt = ZO_gauss_min(f_quadratic, np.array([5.0, 5.0]),
                           h=0.5, mu=1e-8, N=200, t=1, proj=proj_box)
        x = opt.ZOEGm()
        assert np.all(x >= -3.0 - 1e-10) and np.all(x <= 3.0 + 1e-10)

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
