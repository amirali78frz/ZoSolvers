import numpy as np
from .utils import is_diagonal


class ZO_gauss_minmax:
    def __init__(self, func, x0, y0, h=1e-3, tau=1, mu=1e-5, N=10000, t=1,
                 tol=None, B_x=None, B_y=None, proj_x=None, proj_y=None,
                 oracle_type="gaussian"):
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
        oracle_type : {"gaussian", "sphere"}
            "gaussian" : direction u ~ N(0, B^{-1}).
            "sphere"   : direction u uniform on the B-metric unit sphere,
                         u = L s where s ~ Uniform(S^{d-1}) and L = chol(B^{-1}).
                         The oracle includes the factor d.
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

    def oracle(self, x, y, ux, uy, xy, method):
        """Zeroth-order partial oracle.

        Uses simultaneous perturbation (SPSA-style): both ux and uy enter the
        function evaluation, but the oracle is returned only for the requested
        variable (xy='x' or xy='y').

        For oracle_type="sphere" the output is scaled by the corresponding
        dimension (dx for x, dy for y) so E[oracle] ≈ partial gradient.
        """
        if method == "forw":
            g = (self.func(x + self.mu * ux, y + self.mu * uy)
                 - self.func(x, y)) / self.mu
        elif method == "back":
            g = (self.func(x, y)
                 - self.func(x - self.mu * ux, y - self.mu * uy)) / self.mu
        elif method == "center":
            g = (self.func(x + self.mu * ux, y + self.mu * uy)
                 - self.func(x - self.mu * ux, y - self.mu * uy)) / (2.0 * self.mu)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'forw', 'back', or 'center'.")

        if xy == 'x':
            scale = self.dx if self.oracle_type == "sphere" else 1
            B = self.B_x
            u = ux
        elif xy == 'y':
            scale = self.dy if self.oracle_type == "sphere" else 1
            B = self.B_y
            u = uy
        else:
            raise ValueError("xy must be 'x' or 'y'.")

        if B is None:
            return scale * g * u
        else:
            return scale * g * (B @ u)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _num_samples(self, k):
        return k if self.t == "iteration" else self.t

    def _step_x(self, x, y, ux, uy, method, k, gamma=1.0):
        ns = self._num_samples(k)
        grd = sum(self.oracle(x, y, ux, uy, 'x', method) for _ in range(ns)) / ns
        return x - gamma * (self.h / self.tau) * grd

    def _step_y(self, x, y, ux, uy, method, k, gamma=1.0):
        ns = self._num_samples(k)
        grd = sum(self.oracle(x, y, ux, uy, 'y', method) for _ in range(ns)) / ns
        return y + gamma * self.h * grd

    def _check_period(self, base):
        return max(1, int(self.N / base))

    def _identity(self, v):
        return v

    # ------------------------------------------------------------------
    # Algorithms
    # ------------------------------------------------------------------

    def ZOGDA(self, method="forw"):
        """Zeroth-order gradient descent-ascent."""
        x = [self.x0]
        y = [self.y0]
        proj_x = self.proj_x if self.proj_x is not None else self._identity
        proj_y = self.proj_y if self.proj_y is not None else self._identity
        period = self._check_period(100)

        for k in range(self.N - 1):
            ux = self._sample_x()
            uy = self._sample_y()
            x_new = proj_x(self._step_x(x[-1], y[-1], ux, uy, method, k + 1))
            y_new = proj_y(self._step_y(x[-1], y[-1], ux, uy, method, k + 1))
            x.append(x_new)
            y.append(y_new)

            if self.tol is not None and (k + 1) % period == 0:
                gg = 0.0
                for _ in range(10):
                    ux = self._sample_x()
                    uy = self._sample_y()
                    gx = self.oracle(x_new, y_new, ux, uy, 'x', method)
                    gy = self.oracle(x_new, y_new, ux, uy, 'y', method)
                    sx = (x_new - proj_x(x_new - (self.h / self.tau) * gx)) \
                         / (self.h / self.tau)
                    sy = (proj_y(y_new + self.h * gy) - y_new) / self.h
                    gg += np.linalg.norm(sx) ** 2 + np.linalg.norm(sy) ** 2
                if gg / 10 <= self.tol:
                    break

        return np.array(x), np.array(y)

    def ZOEGmm(self, method="forw", gamma=1.0):
        """Zeroth-order extra-gradient minimax."""
        x = [self.x0]
        y = [self.y0]
        proj_x = self.proj_x if self.proj_x is not None else self._identity
        proj_y = self.proj_y if self.proj_y is not None else self._identity
        period = self._check_period(100)

        for k in range(self.N - 1):
            ux = self._sample_x()
            uy = self._sample_y()
            xhat = proj_x(self._step_x(x[-1], y[-1], ux, uy, method, k + 1))
            yhat = proj_y(self._step_y(x[-1], y[-1], ux, uy, method, k + 1))

            ux = self._sample_x()
            uy = self._sample_y()
            x_new = proj_x(self._step_x(xhat, yhat, ux, uy, method, k + 1, gamma))
            y_new = proj_y(self._step_y(xhat, yhat, ux, uy, method, k + 1, gamma))
            x.append(x_new)
            y.append(y_new)

            if self.tol is not None and (k + 1) % period == 0:
                gg = 0.0
                for _ in range(10):
                    ux = self._sample_x()
                    uy = self._sample_y()
                    gx = self.oracle(x_new, y_new, ux, uy, 'x', method)
                    gy = self.oracle(x_new, y_new, ux, uy, 'y', method)
                    sx = (x_new - proj_x(x_new - (self.h / self.tau) * gx)) \
                         / (self.h / self.tau)
                    sy = (proj_y(y_new + self.h * gy) - y_new) / self.h
                    gg += np.linalg.norm(sx) ** 2 + np.linalg.norm(sy) ** 2
                if gg / 10 <= self.tol:
                    break

        return np.array(x), np.array(y)
