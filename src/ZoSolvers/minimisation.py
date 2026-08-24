import numpy as np
from .utils import ParallelBatchMixin, is_diagonal


class ZO_gauss_min(ParallelBatchMixin):
    def __init__(self, func, x0, h=1e-3, mu=1e-5, N=10000, t=1,
                 tol_f=None, tol_g=None, B=None, proj=None,
                 oracle_type="gaussian", project_init=False,
                 n_jobs=1, backend="thread"):
        """
        Zeroth-order minimisation with Gaussian or sphere random oracle.

        Parameters
        ----------
        func : callable
            Objective function f(x) -> float.
        x0 : array_like
            Initial point.
        h : float
            Step size (gradient-descent step in ZOGD; first-stride step in ZOEGm).
        mu : float
            Smoothing parameter.
        N : int
            Maximum number of iterations.
        t : int or "iteration"
            Samples per oracle call. "iteration" uses the current iteration index.
        tol_f : float, optional
            Stop when f(x) <= tol_f * f(x0).
        tol_g : float, optional
            Stop when empirical squared oracle norm <= tol_g.
        B : ndarray, optional
            Precision matrix (inverse of covariance). If None, uses the identity.
        proj : callable, optional
            Projection onto the feasible set. If None, unconstrained.
        project_init : bool
            Whether to project the initial guess x0 onto the feasible set
            before the first step. False (default) returns x0 exactly as given,
            so the first row of the trajectory may be infeasible when x0 lies
            outside the constraint set. True makes every returned iterate
            feasible, including the first. Has no effect when proj is None.
        oracle_type : {"gaussian", "sphere"}
            "gaussian" : direction u ~ N(0, B^{-1}).
            "sphere"   : direction u uniform on the B-metric unit sphere,
                         i.e. u = L s where s ~ Uniform(S^{d-1}) and
                         L = chol(B^{-1}). The oracle is scaled by d.
        n_jobs : int
            Number of workers used to evaluate the `t` samples of one
            mini-batch. 1 (default) runs sequentially; -1 uses all CPU cores,
            -2 all but one. Samples are split into one chunk per worker, so
            the speed-up only materialises when the cost function is expensive
            relative to the scheduling overhead.
        backend : {"thread", "process"}
            Parallel backend. "thread" (default) accepts any callable and pays
            off when `func` releases the GIL (NumPy-heavy code, compiled
            extensions, external simulators). "process" gives true parallelism
            for pure-Python CPU-bound `func`, but requires `func` to be
            picklable and breaks `np.random.seed` reproducibility.
        """
        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.h = h
        self.mu = mu
        self.proj = proj
        self.B = B
        self.N = N
        self.t = t
        self.tol = tol_f
        self.tolg = tol_g
        self.d = len(self.x0)
        self.oracle_type = oracle_type
        self.project_init = bool(project_init)
        self._init_parallel(n_jobs, backend)

        if oracle_type not in ("gaussian", "sphere"):
            raise ValueError("oracle_type must be 'gaussian' or 'sphere'.")

        # Precompute and cache the sampling structure for B.
        self._B_structure = None   # None | 'diagonal' | 'full'
        self._inv_sqrt_B_diag = None
        self._L = None             # chol(B^{-1})

        if B is not None:
            if B.shape[0] != B.shape[1]:
                raise RuntimeError("B must be a square PD symmetric matrix.")
            if is_diagonal(B):
                self._B_structure = 'diagonal'
                self._inv_sqrt_B_diag = 1.0 / np.sqrt(np.diag(B))
            else:
                self._B_structure = 'full'
                self._L = np.linalg.cholesky(np.linalg.inv(B))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_gaussian(self):
        """Sample u ~ N(0, B^{-1}); returns u in R^d."""
        z = np.random.normal(0.0, 1.0, size=self.d)
        if self._B_structure is None:
            return z
        elif self._B_structure == 'diagonal':
            return z * self._inv_sqrt_B_diag
        else:
            return self._L @ z

    def _sample_sphere(self):
        """Sample u uniformly from the B-metric unit sphere.

        When B is None (identity), this is the standard uniform sphere sample.
        Otherwise u = L s where s is uniform on S^{d-1} and L = chol(B^{-1}).
        The oracle using this sample must include a factor of d (handled in
        `oracle`).
        """
        z = np.random.normal(0.0, 1.0, size=self.d)
        z /= np.linalg.norm(z)      # uniform on S^{d-1}
        if self._B_structure is None:
            return z
        elif self._B_structure == 'diagonal':
            return z * self._inv_sqrt_B_diag
        else:
            return self._L @ z

    def _sample(self):
        if self.oracle_type == "gaussian":
            return self._sample_gaussian()
        else:
            return self._sample_sphere()

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    def oracle(self, x, method):
        """Zeroth-order gradient oracle at x.

        For oracle_type="gaussian":  E[oracle] ≈ ∇f_μ(x).
        For oracle_type="sphere":    E[oracle] ≈ ∇f_μ(x)  (d scaling applied).
        """
        u = self._sample()

        if method == "forw":
            g = (self.func(x + self.mu * u) - self.func(x)) / self.mu
        elif method == "back":
            g = (self.func(x) - self.func(x - self.mu * u)) / self.mu
        elif method == "center":
            g = (self.func(x + self.mu * u) - self.func(x - self.mu * u)) / (2.0 * self.mu)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'forw', 'back', or 'center'.")

        scale = self.d if self.oracle_type == "sphere" else 1
        if self.B is None:
            return scale * g * u
        else:
            return scale * g * (self.B @ u)

    # ------------------------------------------------------------------
    # Steps and algorithms
    # ------------------------------------------------------------------

    def _check_period(self, base):
        """Return a safe modulo period (at least 1) to avoid ZeroDivisionError."""
        return max(1, int(self.N / base))

    def _num_samples(self, k):
        return k if self.t == "iteration" else self.t

    def _oracle_sum(self, n, x, method):
        """Sum of n independent oracle draws at x. One worker's share of a batch."""
        total = np.zeros(self.d)
        for _ in range(n):
            total += self.oracle(x, method)
        return total

    def batch_oracle(self, x, method, n):
        """Mean of n independent oracle draws at x, evaluated over n_jobs workers."""
        partials = self._map_chunks(self._oracle_sum, n, x, method)
        return sum(partials) / n

    def _oracle_sq_norm_sum(self, n, x, method):
        return sum(np.linalg.norm(self.oracle(x, method)) ** 2 for _ in range(n))

    def _oracle_sq_norm(self, x, method, n):
        """Mean squared oracle norm at x, evaluated over n_jobs workers."""
        partials = self._map_chunks(self._oracle_sq_norm_sum, n, x, method)
        return sum(partials) / n

    def _initial_x(self):
        """Starting iterate, projected onto the feasible set if requested."""
        if self.proj is not None and self.project_init:
            return self.proj(self.x0)
        return self.x0

    def step(self, x, method, k, gamma=1.0):
        grd = self.batch_oracle(x, method, self._num_samples(k))
        return x - gamma * self.h * grd

    def ZOGD(self, method="forw"):
        """Zeroth-order gradient descent."""
        x = [self._initial_x()]
        fval0 = self.func(x[0])
        period_f = self._check_period(50)
        period_g = self._check_period(50)

        for k in range(self.N - 1):
            x_new = self.step(x[-1], method, k + 1)
            if self.proj is not None:
                x_new = self.proj(x_new)
            x.append(x_new)

            if self.tol is not None and (k + 1) % period_f == 0:
                if self.func(x_new) <= self.tol * fval0:
                    break

            if self.tolg is not None and (k + 1) % period_g == 0:
                gg = self._oracle_sq_norm(x_new, method, 10)
                if self.proj is not None:
                    # Projected residual proxy
                    gg = sum(
                        np.linalg.norm(
                            (x_new - self.proj(self.step(x_new, method, k + 1))) / self.h
                        ) ** 2
                        for _ in range(10)
                    ) / 10
                if gg <= self.tolg:
                    break

        return np.array(x)

    def ZOEGm(self, method="forw", gamma=1.0):
        """Zeroth-order extra-gradient (minimisation)."""
        x = [self._initial_x()]
        fval0 = self.func(x[0])
        period_f = self._check_period(10)
        period_g = self._check_period(50)

        for k in range(self.N - 1):
            xhat = self.step(x[-1], method, k + 1)
            if self.proj is not None:
                xhat = self.proj(xhat)
            x_new = self.step(xhat, method, k + 1, gamma=gamma)
            if self.proj is not None:
                x_new = self.proj(x_new)
            x.append(x_new)

            if self.tol is not None and (k + 1) % period_f == 0:
                if self.func(x_new) <= self.tol * fval0:
                    break

            if self.tolg is not None and (k + 1) % period_g == 0:
                if self.proj is None:
                    gg = self._oracle_sq_norm(x_new, method, 10)
                else:
                    gg = 0.0
                    for _ in range(10):
                        xh = self.proj(self.step(x_new, method, k + 1))
                        xx = self.proj(self.step(xh, method, k + 1, gamma=gamma))
                        gg += np.linalg.norm((x_new - xx) / (self.h * gamma)) ** 2
                    gg /= 10
                if gg <= self.tolg:
                    break

        return np.array(x)
