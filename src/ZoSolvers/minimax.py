import numpy as np
from .utils import ParallelBatchMixin, is_diagonal


class ZO_gauss_minmax(ParallelBatchMixin):
    def __init__(self, func, x0, y0, h=1e-3, tau=1, mu=1e-5, N=10000, t=1,
                 tol=None, B_x=None, B_y=None, proj_x=None, proj_y=None,
                 oracle_type="gaussian", project_init=False,
                 n_jobs=1, backend="thread"):
        """
        Zeroth-order minimax solver with Gaussian or sphere random oracle.

        Parameters
        ----------
        func : callable
            Objective function f(x, y) -> float.  Minimised in x, maximised in y.
        x0 : array_like
            Initial point for the minimiser.
        y0 : array_like
            Initial point for the maximiser.
        h : float
            Base step size (y-step in ZOGDA; y-first-stride in ZOEGmm).
        tau : float
            Ratio of maximiser step size to minimiser step size.
            x-step = h / tau, y-step = h.
        mu : float
            Smoothing parameter.
        N : int
            Maximum number of iterations.
        t : int or "iteration"
            Samples per oracle call. "iteration" uses the current iteration index.
        tol : float, optional
            Stop when the empirical squared projected-operator norm <= tol.
        B_x : ndarray, optional
            Precision matrix for x-direction. If None, uses the identity.
        B_y : ndarray, optional
            Precision matrix for y-direction. If None, uses the identity.
        proj_x : callable, optional
            Projection for x. If None, unconstrained.
        proj_y : callable, optional
            Projection for y. If None, unconstrained.
        project_init : bool
            Whether to project the initial guesses x0 and y0 onto their
            feasible sets before the first step. False (default) returns them
            exactly as given, so the first row of each trajectory may be
            infeasible when the initial point lies outside its constraint set.
            True makes every returned iterate feasible, including the first.
            Has no effect where the corresponding projection is None.
        oracle_type : {"gaussian", "sphere"}
            "gaussian" : direction u ~ N(0, B^{-1}).
            "sphere"   : direction u uniform on the B-metric unit sphere,
                         u = L s where s ~ Uniform(S^{d-1}) and L = chol(B^{-1}).
                         The oracle includes the factor d.
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
        self.y0 = np.asarray(y0, dtype=float)
        self.h = h
        self.mu = mu
        self.proj_x = proj_x
        self.proj_y = proj_y
        self.B_x = B_x
        self.B_y = B_y
        self.tau = tau
        self.N = N
        self.t = t
        self.tol = tol
        self.dx = len(self.x0)
        self.dy = len(self.y0)
        self.oracle_type = oracle_type
        self.project_init = bool(project_init)
        self._init_parallel(n_jobs, backend)

        if oracle_type not in ("gaussian", "sphere"):
            raise ValueError("oracle_type must be 'gaussian' or 'sphere'.")

        # Precompute and cache sampling structures for B_x and B_y.
        self._Bx_structure, self._inv_sqrt_Bx_diag, self._Lx = \
            self._build_B_cache(B_x, "B_x")
        self._By_structure, self._inv_sqrt_By_diag, self._Ly = \
            self._build_B_cache(B_y, "B_y")

    @staticmethod
    def _build_B_cache(B, name):
        """Return (structure, inv_sqrt_diag, L) for a precision matrix B."""
        if B is None:
            return None, None, None
        if B.shape[0] != B.shape[1]:
            raise RuntimeError(f"{name} must be a square PD symmetric matrix.")
        if is_diagonal(B):
            return 'diagonal', 1.0 / np.sqrt(np.diag(B)), None
        else:
            return 'full', None, np.linalg.cholesky(np.linalg.inv(B))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_gaussian(self, d, structure, inv_sqrt_diag, L):
        z = np.random.normal(0.0, 1.0, size=d)
        if structure is None:
            return z
        elif structure == 'diagonal':
            return z * inv_sqrt_diag
        else:
            return L @ z

    def _sample_sphere(self, d, structure, inv_sqrt_diag, L):
        z = np.random.normal(0.0, 1.0, size=d)
        z /= np.linalg.norm(z)          # uniform on S^{d-1}
        if structure is None:
            return z
        elif structure == 'diagonal':
            return z * inv_sqrt_diag
        else:
            return L @ z

    def _sample_x(self):
        if self.oracle_type == "gaussian":
            return self._sample_gaussian(
                self.dx, self._Bx_structure, self._inv_sqrt_Bx_diag, self._Lx)
        else:
            return self._sample_sphere(
                self.dx, self._Bx_structure, self._inv_sqrt_Bx_diag, self._Lx)

    def _sample_y(self):
        if self.oracle_type == "gaussian":
            return self._sample_gaussian(
                self.dy, self._By_structure, self._inv_sqrt_By_diag, self._Ly)
        else:
            return self._sample_sphere(
                self.dy, self._By_structure, self._inv_sqrt_By_diag, self._Ly)

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    def _finite_difference(self, x, y, ux, uy, method):
        """Scalar directional finite difference along the pair (ux, uy)."""
        if method == "forw":
            return (self.func(x + self.mu * ux, y + self.mu * uy)
                    - self.func(x, y)) / self.mu
        elif method == "back":
            return (self.func(x, y)
                    - self.func(x - self.mu * ux, y - self.mu * uy)) / self.mu
        elif method == "center":
            return (self.func(x + self.mu * ux, y + self.mu * uy)
                    - self.func(x - self.mu * ux, y - self.mu * uy)) / (2.0 * self.mu)
        raise ValueError(f"Unknown method '{method}'. Choose 'forw', 'back', or 'center'.")

    def _shape_x(self, g, ux):
        scale = self.dx if self.oracle_type == "sphere" else 1
        return scale * g * (ux if self.B_x is None else self.B_x @ ux)

    def _shape_y(self, g, uy):
        scale = self.dy if self.oracle_type == "sphere" else 1
        return scale * g * (uy if self.B_y is None else self.B_y @ uy)

    def oracle(self, x, y, ux, uy, xy, method):
        """Zeroth-order partial oracle.

        Uses simultaneous perturbation (SPSA-style): both ux and uy enter the
        function evaluation, but the oracle is returned only for the requested
        variable (xy='x' or xy='y').

        For oracle_type="sphere" the output is scaled by the corresponding
        dimension (dx for x, dy for y) so E[oracle] ~ partial gradient.
        """
        if xy not in ('x', 'y'):
            raise ValueError("xy must be 'x' or 'y'.")
        g = self._finite_difference(x, y, ux, uy, method)
        return self._shape_x(g, ux) if xy == 'x' else self._shape_y(g, uy)

    def _oracle_pair(self, x, y, ux, uy, method):
        """Both partial oracles from a single finite difference.

        The x- and y-oracles built from the same (ux, uy) share the same scalar
        difference, so evaluating `func` once for the pair is equivalent to
        calling `oracle` twice -- at half the number of function evaluations.
        """
        g = self._finite_difference(x, y, ux, uy, method)
        return self._shape_x(g, ux), self._shape_y(g, uy)

    # ------------------------------------------------------------------
    # Mini-batch (parallel over n_jobs workers)
    # ------------------------------------------------------------------

    def _num_samples(self, k):
        return k if self.t == "iteration" else self.t

    def _grad_sum(self, n, x, y, method):
        """Summed partial oracles over n independent draws. One worker's share.

        Each draw uses a fresh (ux, uy) pair; the x- and y-oracles within a
        draw share that pair, which keeps the coupled SPSA structure intact.
        """
        gx_total = np.zeros(self.dx)
        gy_total = np.zeros(self.dy)
        for _ in range(n):
            ux = self._sample_x()
            uy = self._sample_y()
            gx, gy = self._oracle_pair(x, y, ux, uy, method)
            gx_total += gx
            gy_total += gy
        return gx_total, gy_total

    def batch_oracle(self, x, y, method, n):
        """Mean partial oracles over n draws, evaluated over n_jobs workers."""
        partials = self._map_chunks(self._grad_sum, n, x, y, method)
        gx = sum(part[0] for part in partials) / n
        gy = sum(part[1] for part in partials) / n
        return gx, gy

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _px(self, v):
        return v if self.proj_x is None else self.proj_x(v)

    def _py(self, v):
        return v if self.proj_y is None else self.proj_y(v)

    def _initial_points(self):
        """Starting iterates, projected onto the feasible sets if requested."""
        if self.project_init:
            return self._px(self.x0), self._py(self.y0)
        return self.x0, self.y0

    def _step(self, x, y, method, k, gamma=1.0):
        """One simultaneous descent-ascent stride, projected onto the feasible set."""
        gx, gy = self.batch_oracle(x, y, method, self._num_samples(k))
        x_new = self._px(x - gamma * (self.h / self.tau) * gx)
        y_new = self._py(y + gamma * self.h * gy)
        return x_new, y_new

    def _check_period(self, base):
        return max(1, int(self.N / base))

    def _residual_sq_sum(self, n, x, y, method):
        """Summed squared projected-operator residual over n draws."""
        total = 0.0
        step_x = self.h / self.tau
        for _ in range(n):
            ux = self._sample_x()
            uy = self._sample_y()
            gx, gy = self._oracle_pair(x, y, ux, uy, method)
            sx = (x - self._px(x - step_x * gx)) / step_x
            sy = (self._py(y + self.h * gy) - y) / self.h
            total += np.linalg.norm(sx) ** 2 + np.linalg.norm(sy) ** 2
        return total

    def _residual_sq(self, x, y, method, n=10):
        """Mean squared projected-operator residual, over n_jobs workers."""
        return sum(self._map_chunks(self._residual_sq_sum, n, x, y, method)) / n

    # ------------------------------------------------------------------
    # Algorithms
    # ------------------------------------------------------------------

    def ZOGDA(self, method="forw"):
        """Zeroth-order gradient descent-ascent."""
        x0, y0 = self._initial_points()
        x = [x0]
        y = [y0]
        period = self._check_period(100)

        for k in range(self.N - 1):
            x_new, y_new = self._step(x[-1], y[-1], method, k + 1)
            x.append(x_new)
            y.append(y_new)

            if self.tol is not None and (k + 1) % period == 0:
                if self._residual_sq(x_new, y_new, method) <= self.tol:
                    break

        return np.array(x), np.array(y)

    def ZOEGmm(self, method="forw", gamma=1.0):
        """Zeroth-order extra-gradient minimax."""
        x0, y0 = self._initial_points()
        x = [x0]
        y = [y0]
        period = self._check_period(100)

        for k in range(self.N - 1):
            xhat, yhat = self._step(x[-1], y[-1], method, k + 1)
            x_new, y_new = self._step(xhat, yhat, method, k + 1, gamma)
            x.append(x_new)
            y.append(y_new)

            if self.tol is not None and (k + 1) % period == 0:
                if self._residual_sq(x_new, y_new, method) <= self.tol:
                    break

        return np.array(x), np.array(y)
