"""
Tests for ZoSolvers.minimax.ZO_gauss_minmax.

Coverage:
  - Initialisation (attributes, B caching, error handling)
  - Sampling (_sample_x, _sample_y for both oracle types and B structures)
  - Oracle (shape, methods, xy selection, unbiasedness for both oracle types)
  - ZOGDA  (convergence, projection, early stopping, t="iteration", small N)
  - ZOEGmm (convergence, projection, gamma, t="iteration", small N)

Test problem  (separable, saddle at origin):
    f(x, y) = ||x||^2/2  -  ||y||^2/2
    grad_x f =  x,  grad_y f = -y
    min_x max_y f  with box projection [-5, 5]^d  →  saddle at (0, 0).
"""

import numpy as np
import pytest
from ZoSolvers.minimax import ZO_gauss_minmax


# ---------------------------------------------------------------------------
# Shared test problems
# ---------------------------------------------------------------------------

def f_sep(x, y):
    """f(x,y) = ||x||^2/2 - ||y||^2/2.  Saddle at (0, 0)."""
    return 0.5 * float(np.dot(x, x)) - 0.5 * float(np.dot(y, y))


def f_bilinear_1d(x, y):
    """f(x,y) = x[0]*y[0].  grad_x = y[0], grad_y = x[0]."""
    return float(x[0] * y[0])


def proj_box(v, lo=-5.0, hi=5.0):
    return np.clip(v, lo, hi)


def _make_solver(dx=2, dy=2, oracle_type="gaussian",
                 B_x=None, B_y=None, proj_x=None, proj_y=None, **overrides):
    x0 = np.full(dx, 4.0)
    y0 = np.full(dy, 4.0)
    params = dict(h=0.1, tau=1, mu=1e-8, N=2000, t=10,
                  B_x=B_x, B_y=B_y, proj_x=proj_x, proj_y=proj_y,
                  oracle_type=oracle_type)
    params.update(overrides)
    return ZO_gauss_minmax(f_sep, x0, y0, **params)


def _oracle_mean_x(opt, x, y, method, n=20_000, seed=42):
    np.random.seed(seed)
    acc = np.zeros(opt.dx)
    for _ in range(n):
        ux = opt._sample_x()
        uy = opt._sample_y()
        acc += opt.oracle(x, y, ux, uy, 'x', method)
    return acc / n


def _oracle_mean_y(opt, x, y, method, n=20_000, seed=42):
    np.random.seed(seed)
    acc = np.zeros(opt.dy)
    for _ in range(n):
        ux = opt._sample_x()
        uy = opt._sample_y()
        acc += opt.oracle(x, y, ux, uy, 'y', method)
    return acc / n


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_attributes(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(3))
        assert opt.dx == 2
        assert opt.dy == 3
        assert opt.oracle_type == "gaussian"
        assert opt._Bx_structure is None
        assert opt._By_structure is None

    def test_dx_dy_inferred_from_x0_y0(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(4), np.zeros(5))
        assert opt.dx == 4
        assert opt.dy == 5

    def test_diagonal_Bx_cache(self):
        B = np.diag([1.0, 4.0])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_x=B)
        assert opt._Bx_structure == 'diagonal'
        np.testing.assert_allclose(opt._inv_sqrt_Bx_diag, [1.0, 0.5])
        assert opt._Lx is None

    def test_full_Bx_cache_cholesky_correct(self):
        B = np.array([[3.0, 1.0], [1.0, 2.0]])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_x=B)
        assert opt._Bx_structure == 'full'
        assert opt._Lx is not None
        np.testing.assert_allclose(opt._Lx @ opt._Lx.T, np.linalg.inv(B), atol=1e-12)

    def test_diagonal_By_cache(self):
        B = np.diag([2.0, 0.5])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_y=B)
        assert opt._By_structure == 'diagonal'
        np.testing.assert_allclose(opt._inv_sqrt_By_diag, 1.0 / np.sqrt([2.0, 0.5]))

    def test_independent_Bx_By_caches(self):
        Bx = np.diag([1.0, 2.0])
        By = np.array([[3.0, 0.5], [0.5, 2.0]])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_x=Bx, B_y=By)
        assert opt._Bx_structure == 'diagonal'
        assert opt._By_structure == 'full'

    def test_invalid_oracle_type_raises(self):
        with pytest.raises(ValueError, match="oracle_type"):
            ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), oracle_type="nope")

    def test_non_square_Bx_raises(self):
        with pytest.raises(RuntimeError):
            ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_x=np.ones((2, 3)))

    def test_non_square_By_raises(self):
        with pytest.raises(RuntimeError):
            ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_y=np.ones((3, 2)))

    def test_sphere_oracle_type_stored(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), oracle_type="sphere")
        assert opt.oracle_type == "sphere"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    @pytest.fixture(autouse=True)
    def seed(self):
        np.random.seed(0)

    def test_sample_x_no_B_shape(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(4), np.zeros(3))
        assert opt._sample_x().shape == (4,)

    def test_sample_y_no_B_shape(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(4), np.zeros(3))
        assert opt._sample_y().shape == (3,)

    def test_sample_x_diag_B_shape(self):
        B = np.diag([2.0, 3.0])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_x=B)
        assert opt._sample_x().shape == (2,)

    def test_sample_y_full_B_shape(self):
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2), B_y=B)
        assert opt._sample_y().shape == (2,)

    def test_sphere_x_on_unit_sphere_no_B(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(5), np.zeros(3), oracle_type="sphere")
        for _ in range(30):
            u = opt._sample_x()
            assert u.shape == (5,)
            np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-12)

    def test_sphere_y_on_unit_sphere_no_B(self):
        opt = ZO_gauss_minmax(f_sep, np.zeros(3), np.zeros(4), oracle_type="sphere")
        for _ in range(30):
            u = opt._sample_y()
            assert u.shape == (4,)
            np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-12)

    def test_gaussian_and_sphere_samples_differ(self):
        opt_g = ZO_gauss_minmax(f_sep, np.zeros(4), np.zeros(4), oracle_type="gaussian")
        opt_s = ZO_gauss_minmax(f_sep, np.zeros(4), np.zeros(4), oracle_type="sphere")
        np.random.seed(7)
        ug = opt_g._sample_x()
        np.random.seed(7)
        us = opt_s._sample_x()
        assert not np.allclose(ug, us)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class TestOracle:
    def _opt(self, oracle_type="gaussian", B_x=None, B_y=None):
        return ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2),
                               mu=1e-8, oracle_type=oracle_type,
                               B_x=B_x, B_y=B_y)

    def _sample_uxy(self, opt, seed=0):
        np.random.seed(seed)
        return opt._sample_x(), opt._sample_y()

    def test_oracle_x_output_shape(self):
        opt = self._opt()
        ux, uy = self._sample_uxy(opt)
        g = opt.oracle(np.ones(2), np.ones(2), ux, uy, 'x', "center")
        assert g.shape == (2,)

    def test_oracle_y_output_shape(self):
        opt = self._opt()
        ux, uy = self._sample_uxy(opt)
        g = opt.oracle(np.ones(2), np.ones(2), ux, uy, 'y', "center")
        assert g.shape == (2,)

    def test_invalid_method_raises(self):
        opt = self._opt()
        ux, uy = self._sample_uxy(opt)
        with pytest.raises(ValueError, match="Unknown method"):
            opt.oracle(np.ones(2), np.ones(2), ux, uy, 'x', "bad")

    def test_invalid_xy_raises(self):
        opt = self._opt()
        ux, uy = self._sample_uxy(opt)
        with pytest.raises(ValueError):
            opt.oracle(np.ones(2), np.ones(2), ux, uy, 'z', "center")

    @pytest.mark.parametrize("method", ["forw", "back", "center"])
    def test_all_methods_run(self, method):
        opt = self._opt()
        ux, uy = self._sample_uxy(opt)
        gx = opt.oracle(np.ones(2), np.ones(2), ux, uy, 'x', method)
        gy = opt.oracle(np.ones(2), np.ones(2), ux, uy, 'y', method)
        assert gx.shape == (2,) and gy.shape == (2,)

    # --- Unbiasedness ---------------------------------------------------------
    # For f_sep: grad_x f(x,y) = x,  grad_y f(x,y) = -y.

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_x_unbiased_no_B(self, oracle_type):
        x = np.array([2.0, -1.0])
        y = np.array([1.0, 3.0])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2),
                              mu=1e-8, oracle_type=oracle_type)
        est = _oracle_mean_x(opt, x, y, "center")
        np.testing.assert_allclose(est, x, atol=0.15)   # E[oracle_x] = grad_x = x

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_y_unbiased_no_B(self, oracle_type):
        x = np.array([2.0, -1.0])
        y = np.array([1.0, 3.0])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2),
                              mu=1e-8, oracle_type=oracle_type)
        est = _oracle_mean_y(opt, x, y, "center")
        np.testing.assert_allclose(est, -y, atol=0.15)  # E[oracle_y] = grad_y = -y

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_x_unbiased_diagonal_B(self, oracle_type):
        x = np.array([1.0, -2.0])
        y = np.array([0.5, 1.0])
        Bx = np.diag([2.0, 0.5])
        By = np.diag([1.0, 3.0])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2),
                              mu=1e-8, B_x=Bx, B_y=By, oracle_type=oracle_type)
        est = _oracle_mean_x(opt, x, y, "center")
        np.testing.assert_allclose(est, x, atol=0.2)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_oracle_y_unbiased_full_B(self, oracle_type):
        x = np.array([1.0, -1.0])
        y = np.array([2.0, 0.5])
        By = np.array([[3.0, 0.5], [0.5, 2.0]])
        opt = ZO_gauss_minmax(f_sep, np.zeros(2), np.zeros(2),
                              mu=1e-8, B_y=By, oracle_type=oracle_type)
        est = _oracle_mean_y(opt, x, y, "center")
        np.testing.assert_allclose(est, -y, atol=0.2)

    def test_bilinear_oracle_x_unbiased(self):
        """For f(x,y)=x[0]*y[0]: E[oracle_x] = y[0]*e1 via SPSA cross-cancellation."""
        x = np.array([3.0])
        y = np.array([2.0])
        opt = ZO_gauss_minmax(f_bilinear_1d, np.zeros(1), np.zeros(1), mu=1e-8)
        est = _oracle_mean_x(opt, x, y, "center", n=30_000)
        np.testing.assert_allclose(est, y, atol=0.1)   # grad_x = y

    def test_bilinear_oracle_y_unbiased(self):
        x = np.array([3.0])
        y = np.array([2.0])
        opt = ZO_gauss_minmax(f_bilinear_1d, np.zeros(1), np.zeros(1), mu=1e-8)
        est = _oracle_mean_y(opt, x, y, "center", n=30_000)
        np.testing.assert_allclose(est, x, atol=0.1)   # grad_y = x


# ---------------------------------------------------------------------------
# ZOGDA
# ---------------------------------------------------------------------------

class TestZOGDA:
    def _converged(self, x, y):
        """Both x and y reduced by at least 98% in squared norm."""
        return (np.dot(x[-1], x[-1]) < np.dot(x[0], x[0]) * 0.02 and
                np.dot(y[-1], y[-1]) < np.dot(y[0], y[0]) * 0.02)

    def test_returns_two_ndarrays(self):
        np.random.seed(42)
        x, y = _make_solver(N=30, t=1).ZOGDA()
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)

    def test_output_column_counts(self):
        np.random.seed(42)
        opt = ZO_gauss_minmax(f_sep, np.full(3, 4.0), np.full(2, 4.0),
                              h=0.1, mu=1e-8, N=30, t=1)
        x, y = opt.ZOGDA()
        assert x.shape[1] == 3 and y.shape[1] == 2

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_no_B(self, oracle_type):
        np.random.seed(42)
        x, y = _make_solver(oracle_type=oracle_type, N=3000, t=15).ZOGDA(method="center")
        assert self._converged(x, y)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_diagonal_B(self, oracle_type):
        np.random.seed(42)
        B = np.diag([2.0, 0.5])
        x, y = _make_solver(oracle_type=oracle_type, B_x=B, B_y=B,
                            N=3000, t=15).ZOGDA(method="center")
        assert self._converged(x, y)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_full_B(self, oracle_type):
        np.random.seed(42)
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        x, y = _make_solver(oracle_type=oracle_type, B_x=B, B_y=B,
                            N=3000, t=15).ZOGDA(method="center")
        assert self._converged(x, y)

    def test_projection_keeps_iterates_in_feasible_set(self):
        np.random.seed(42)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.5, mu=1e-8, N=200, t=1,
                              proj_x=proj_box, proj_y=proj_box)
        x, y = opt.ZOGDA()
        assert np.all(x >= -5.0 - 1e-10) and np.all(x <= 5.0 + 1e-10)
        assert np.all(y >= -5.0 - 1e-10) and np.all(y <= 5.0 + 1e-10)

    def test_proj_not_mutated_between_calls(self):
        """ZOGDA must not overwrite self.proj_x/y (regression for the old bug)."""
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.1, mu=1e-8, N=50, t=1)
        assert opt.proj_x is None and opt.proj_y is None
        opt.ZOGDA()
        assert opt.proj_x is None and opt.proj_y is None  # must stay None

    def test_tol_triggers_early_stop(self):
        np.random.seed(42)
        # tol=1e10 guarantees the convergence check always passes → early stop
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.1, mu=1e-8, N=5000, t=5, tol=1e10)
        x, y = opt.ZOGDA()
        assert len(x) < 5000 and len(y) < 5000

    def test_t_iteration_mode_runs(self):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.05, mu=1e-8, N=200, t="iteration")
        x, y = opt.ZOGDA(method="center")
        assert x.shape[1] == 2 and y.shape[1] == 2

    @pytest.mark.parametrize("method", ["forw", "back", "center"])
    def test_all_methods_run(self, method):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 3.0), np.full(2, 3.0),
                              h=0.05, mu=1e-6, N=30, t=5)
        x, y = opt.ZOGDA(method=method)
        assert x.shape[1] == 2 and y.shape[1] == 2

    def test_small_N_no_division_by_zero(self):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 1.0), np.full(2, 1.0),
                              h=0.1, mu=1e-5, N=5, t=1, tol=1e10)
        x, y = opt.ZOGDA()
        assert len(x) <= 5 and len(y) <= 5


# ---------------------------------------------------------------------------
# ZOEGmm
# ---------------------------------------------------------------------------

class TestZOEGmm:
    def _converged(self, x, y):
        return (np.dot(x[-1], x[-1]) < np.dot(x[0], x[0]) * 0.02 and
                np.dot(y[-1], y[-1]) < np.dot(y[0], y[0]) * 0.02)

    def test_returns_two_ndarrays(self):
        np.random.seed(42)
        x, y = _make_solver(N=30, t=1).ZOEGmm()
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)

    def test_output_column_counts(self):
        np.random.seed(42)
        opt = ZO_gauss_minmax(f_sep, np.full(3, 4.0), np.full(2, 4.0),
                              h=0.1, mu=1e-8, N=30, t=1)
        x, y = opt.ZOEGmm()
        assert x.shape[1] == 3 and y.shape[1] == 2

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_no_B(self, oracle_type):
        np.random.seed(42)
        x, y = _make_solver(oracle_type=oracle_type, N=3000, t=15).ZOEGmm(method="center")
        assert self._converged(x, y)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_diagonal_B(self, oracle_type):
        np.random.seed(42)
        B = np.diag([2.0, 0.5])
        x, y = _make_solver(oracle_type=oracle_type, B_x=B, B_y=B,
                            N=3000, t=15).ZOEGmm(method="center")
        assert self._converged(x, y)

    @pytest.mark.parametrize("oracle_type", ["gaussian", "sphere"])
    def test_convergence_full_B(self, oracle_type):
        np.random.seed(42)
        B = np.array([[3.0, 0.5], [0.5, 2.0]])
        x, y = _make_solver(oracle_type=oracle_type, B_x=B, B_y=B,
                            N=3000, t=15).ZOEGmm(method="center")
        assert self._converged(x, y)

    def test_projection_keeps_iterates_in_feasible_set(self):
        np.random.seed(42)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.5, mu=1e-8, N=200, t=1,
                              proj_x=proj_box, proj_y=proj_box)
        x, y = opt.ZOEGmm()
        assert np.all(x >= -5.0 - 1e-10) and np.all(x <= 5.0 + 1e-10)
        assert np.all(y >= -5.0 - 1e-10) and np.all(y <= 5.0 + 1e-10)

    def test_proj_not_mutated_between_calls(self):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.1, mu=1e-8, N=50, t=1)
        assert opt.proj_x is None and opt.proj_y is None
        opt.ZOEGmm()
        assert opt.proj_x is None and opt.proj_y is None

    def test_tol_triggers_early_stop(self):
        np.random.seed(42)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.1, mu=1e-8, N=5000, t=5, tol=1e10)
        x, y = opt.ZOEGmm()
        assert len(x) < 5000 and len(y) < 5000

    def test_gamma_less_than_one_still_converges(self):
        np.random.seed(42)
        x, y = _make_solver(N=3000, t=15).ZOEGmm(method="center", gamma=0.5)
        assert self._converged(x, y)

    def test_t_iteration_mode_runs(self):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 4.0), np.full(2, 4.0),
                              h=0.05, mu=1e-8, N=200, t="iteration")
        x, y = opt.ZOEGmm(method="center")
        assert x.shape[1] == 2 and y.shape[1] == 2

    @pytest.mark.parametrize("method", ["forw", "back", "center"])
    def test_all_methods_run(self, method):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 3.0), np.full(2, 3.0),
                              h=0.05, mu=1e-6, N=30, t=5)
        x, y = opt.ZOEGmm(method=method)
        assert x.shape[1] == 2 and y.shape[1] == 2

    def test_small_N_no_division_by_zero(self):
        np.random.seed(0)
        opt = ZO_gauss_minmax(f_sep, np.full(2, 1.0), np.full(2, 1.0),
                              h=0.1, mu=1e-5, N=5, t=1, tol=1e10)
        x, y = opt.ZOEGmm()
        assert len(x) <= 5 and len(y) <= 5
