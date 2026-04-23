# ZoSolvers

**Zeroth-order optimisation solvers with Gaussian and sphere random oracles.**

ZoSolvers provides gradient-free solvers for **minimisation** and **minimax** problems. When the gradient of the objective is unavailable — because the function is non-differentiable, comes from a black-box simulator, or is too expensive to differentiate — ZoSolvers estimates it using random directional perturbations.

📖 Full documentation: [ZoSolvers Manual (PDF)](Docs/ZoSolvers_Manual.pdf)

---

## Features

- **Two oracle types** — Gaussian (`u ~ N(0, B⁻¹)`) and sphere (uniform on the unit sphere), both with optional preconditioning via a precision matrix `B`
- **Four solvers** — ZOGD, ZOEGm (minimisation), ZOGDA, ZOEGmm (minimax)
- **Three finite-difference methods** — forward, backward, centered
- **Flexible mini-batching** — fixed number of oracle samples per step, or a growing schedule (`t="iteration"`)
- **Constrained problems** — pass any projection function
- **Early stopping** — based on relative function decrease or oracle norm
- **Efficient sampling** — Cholesky factorisation cached at construction; diagonal `B` handled without matrix inversion

---

## Installation

**From wheel (system-wide):**
```bash
pip install dist/zosolvers-0.1.0-py3-none-any.whl
```

**From source with Poetry:**
```bash
git clone https://github.com/amirali78frz/ZoSolvers.git
cd ZoSolvers
poetry install
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 2.2, Matplotlib ≥ 3.7

---

## Quick Start

### Minimisation

```python
import numpy as np
from ZoSolvers.minimisation import ZO_gauss_min

def f(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]

x0  = np.array([5.0, -5.0])
opt = ZO_gauss_min(f, x0, h=1e-2, mu=1e-5, N=2000, t=10)

# Zeroth-order gradient descent
x_traj = opt.ZOGD(method="center")

# Zeroth-order extra-gradient
x_traj = opt.ZOEGm(method="center", gamma=1.0)
```

### Minimax

```python
from ZoSolvers.minimax import ZO_gauss_minmax

def f(x, y):
    return x[0]**2 + x[1]**2 - y[0]**2 - y[1]**2 + x[0]*y[1]

x0 = np.array([5.0, -5.0])
y0 = np.array([3.0, -3.0])

opt = ZO_gauss_minmax(f, x0, y0, h=1e-3, tau=1, mu=1e-8, N=10000, t=5)

# Gradient descent-ascent
x_traj, y_traj = opt.ZOGDA(method="center")

# Extra-gradient
x_traj, y_traj = opt.ZOEGmm(method="center", gamma=0.8)
```

### Sphere Oracle and Precision Matrix

```python
B = np.array([[10.0, 0.5],
              [0.5,  2.0]])

opt = ZO_gauss_min(f, x0, h=1e-2, mu=1e-5, N=2000, t=10,
                   B=B, oracle_type="sphere")
x_traj = opt.ZOGD(method="center")
```

### Box-Constrained Problem

```python
proj = lambda x: np.clip(x, -3.0, 3.0)

opt = ZO_gauss_min(f, x0, h=1e-2, mu=1e-5, N=2000, t=10, proj=proj)
x_traj = opt.ZOGD(method="center")
```

---

## Solvers

| Solver | Class | Problem | Description |
|--------|-------|---------|-------------|
| `ZOGD` | `ZO_gauss_min` | Minimisation | Zeroth-order gradient descent |
| `ZOEGm` | `ZO_gauss_min` | Minimisation | Zeroth-order extra-gradient |
| `ZOGDA` | `ZO_gauss_minmax` | Minimax | Zeroth-order gradient descent-ascent |
| `ZOEGmm` | `ZO_gauss_minmax` | Minimax | Zeroth-order extra-gradient minimax |

---

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `h` | Step size |
| `mu` | Smoothing parameter for finite differences |
| `N` | Maximum number of iterations |
| `t` | Oracle samples per step (`int` or `"iteration"` for growing schedule) |
| `B` | Precision matrix shaping the perturbation distribution (`None` = identity) |
| `oracle_type` | `"gaussian"` or `"sphere"` |
| `proj` | Projection onto the feasible set (`None` = unconstrained) |
| `tau` | *(Minimax)* Step-size ratio: x-step = `h/tau`, y-step = `h` |
| `gamma` | *(Extra-gradient)* Second-stride scale factor |

---

## Package Structure

```
ZoSolvers/
├── src/ZoSolvers/
│   ├── minimisation.py   # ZO_gauss_min
│   ├── minimax.py        # ZO_gauss_minmax
│   └── utils.py          # shared utilities
├── tests/
│   ├── min_test.py       # pytest suite for minimisation
│   ├── minimax_test.py   # pytest suite for minimax
│   ├── demo_min.py       # minimisation demo with plots
│   └── demo_minimax.py   # minimax demo with plots
├── Docs/
│   └── ZoSolvers_Manual.pdf
└── pyproject.toml
```

---

## Running the Demos

```bash
poetry run python tests/demo_min.py
poetry run python tests/demo_minimax.py
```

## Running the Tests

```bash
poetry run pytest tests/ -v
```

---

## License

MIT
