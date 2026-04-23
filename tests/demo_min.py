"""
Demo script for ZO_gauss_min (minimisation module).

Run with:
    poetry run python tests/demo_min.py

Demonstrates:
  - ZOGD and ZOEGm with Gaussian and sphere oracles
  - Diagonal and full B (precision) matrices
  - Box-projected (constrained) variants
  - Forward / backward / centered finite-difference methods
  - Fixed vs growing (t="iteration") mini-batch size
  - tol_f early stopping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ZoSolvers.minimisation import ZO_gauss_min


# ---------------------------------------------------------------------------
# Problem:  f(x) = x0^2 + x1^2 + x0*x1,   minimum at x* = [0, 0]
# ---------------------------------------------------------------------------

def f(x):
    return 10*np.linalg.norm(x)**2 + 0.5*np.sin(x[0]) + 0.5*np.cos(x[1])

x0     = np.array([5.0, -5.0])
proj   = lambda x: np.clip(x, -2.0, 2.0)
B_diag = np.diag([10.0, 0.01])
B_full = np.array([[20.0, 0.01], [0.01, 0.02]])

SEP = "=" * 60

def report(label, traj):
    f0, ff = f(traj[0]), f(traj[-1])
    pct = (1 - ff / f0) * 100 if f0 else float("nan")
    print(f"  {label:<40s}  iters={len(traj):>5d}  "
          f"f0={f0:8.3f}  f_final={ff:10.6f}  reduction={pct:6.1f}%")

def f_history(traj):
    return [f(x) for x in traj]

# Shared solver settings
KW = dict(h=1e-2, mu=1e-5, N=2000, t=10)
KW1 = dict(h=1e-2, mu=1e-5, N=200, t=10)
# ===========================================================================
print(SEP)
print("ZOGD — oracle type comparison (no B, centered diff)")
print(SEP)
traj_gauss = ZO_gauss_min(f, x0, **KW, oracle_type="gaussian").ZOGD(method="center")
traj_sph   = ZO_gauss_min(f, x0, **KW, oracle_type="sphere"  ).ZOGD(method="center")
report("Gaussian oracle", traj_gauss)
report("Sphere oracle",   traj_sph)
print()

# ===========================================================================
print(SEP)
print("ZOGD — finite-difference method comparison (Gaussian, no B)")
print(SEP)
for method in ("forw", "back", "center"):
    traj = ZO_gauss_min(f, x0, **KW).ZOGD(method=method)
    report(f"method = {method}", traj)
print()

# ===========================================================================
print(SEP)
print("ZOGD — B matrix comparison (Gaussian, centered diff)")
print(SEP)
traj_I    = ZO_gauss_min(f, x0, **KW).ZOGD(method="center")
traj_diag = ZO_gauss_min(f, x0, **KW, B=B_diag).ZOGD(method="center")
traj_full = ZO_gauss_min(f, x0, **KW, B=B_full).ZOGD(method="center")
report("B = I (no B)",      traj_I)
report("B diagonal",        traj_diag)
report("B full",            traj_full)
print()

# ===========================================================================
print(SEP)
print("ZOGD — sphere oracle with B (centered diff)")
print(SEP)
traj_sph_diag = ZO_gauss_min(f, x0, **KW, B=B_diag, oracle_type="sphere").ZOGD(method="center")
traj_sph_full = ZO_gauss_min(f, x0, **KW, B=B_full, oracle_type="sphere").ZOGD(method="center")
report("Sphere + B diagonal", traj_sph_diag)
report("Sphere + B full",     traj_sph_full)
print()

# ===========================================================================
print(SEP)
print("ZOGD — unconstrained vs projected (box [-2, 2]^2)")
print(SEP)
traj_uncon = ZO_gauss_min(f, x0, **KW).ZOGD(method="center")
traj_proj  = ZO_gauss_min(f, x0, **KW, proj=proj).ZOGD(method="center")
report("Unconstrained",    traj_uncon)
report("Projected",        traj_proj)
feasible = np.all(traj_proj >= -2 - 1e-9) and np.all(traj_proj <= 2 + 1e-9)
print(f"  All projected iterates inside box [-2, 2]^2: {feasible}")
print()

# ===========================================================================
print(SEP)
print("ZOGD — fixed t vs t = 'iteration' (growing mini-batch)")
print(SEP)
traj_t10   = ZO_gauss_min(f, x0, **KW1).ZOGD(method="center")   # t=10 already in KW
traj_titer = ZO_gauss_min(f, x0, **{**KW1, 't': "iteration"}).ZOGD(method="center")
report("t = 10 (fixed)",    traj_t10)
report("t = iteration",     traj_titer)
print()

# ===========================================================================
print(SEP)
print("ZOGD — tol_f early stopping (stop when f <= 0.01 * f(x0))")
print(SEP)
opt_tol  = ZO_gauss_min(f, x0, **{**KW, 'N': 5000, 'tol_f': 0.01})
traj_tol = opt_tol.ZOGD(method="center")
report("tol_f = 0.01", traj_tol)
print(f"  Stopped early (before N=5000): {len(traj_tol) < 5000}")
print()

# ===========================================================================
print(SEP)
print("ZOEGm — Gaussian oracle, different gamma (no B)")
print(SEP)
opt_eg = ZO_gauss_min(f, x0, **KW)
for gamma in (1.0, 0.5, 0.25):
    traj = opt_eg.ZOEGm(method="center", gamma=gamma)
    report(f"gamma = {gamma}", traj)
print()

# ===========================================================================
print(SEP)
print("ZOEGm — Sphere oracle + full B + projection")
print(SEP)
traj_eg_full = ZO_gauss_min(f, x0, **KW, B=B_full,
                             oracle_type="sphere", proj=proj).ZOEGm(method="center")
report("Sphere + full B + proj", traj_eg_full)
feasible = np.all(traj_eg_full >= -2 - 1e-9) and np.all(traj_eg_full <= 2 + 1e-9)
print(f"  All iterates inside box [-2, 2]^2: {feasible}")
print()

# ===========================================================================
# Plots
# ===========================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    r"ZO Minimisation — $f(x)=x_0^2+x_1^2+x_0x_1$,  $x_0=[5,-5]$",
    fontsize=13)

# 1: oracle type
ax = axes[0, 0]
ax.semilogy(f_history(traj_gauss), label="Gaussian")
ax.semilogy(f_history(traj_sph),   label="Sphere")
ax.set_title("Oracle type (ZOGD, no B)")
ax.set_xlabel("iteration"); ax.set_ylabel("f(x)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# 2: B matrix
ax = axes[0, 1]
ax.semilogy(f_history(traj_I),    label="B = I")
ax.semilogy(f_history(traj_diag), label="B diagonal")
ax.semilogy(f_history(traj_full), label="B full")
ax.set_title("B matrix (ZOGD, Gaussian)")
ax.set_xlabel("iteration"); ax.set_ylabel("f(x)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# 3: ZOGD vs ZOEGm
ax = axes[0, 2]
traj_eg1 = opt_eg.ZOEGm(method="center", gamma=1.0)
ax.semilogy(f_history(traj_I),   label="ZOGD")
ax.semilogy(f_history(traj_eg1), label="ZOEGm γ=1")
ax.set_title("ZOGD vs ZOEGm (Gaussian, no B)")
ax.set_xlabel("iteration"); ax.set_ylabel("f(x)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# 4: projection — iterate trajectories
ax = axes[1, 0]
ax.plot(traj_uncon[:, 0], label="uncon x₀", alpha=0.8)
ax.plot(traj_uncon[:, 1], label="uncon x₁", alpha=0.8)
ax.plot(traj_proj[:, 0],  label="proj x₀",  linestyle="--")
ax.plot(traj_proj[:, 1],  label="proj x₁",  linestyle="--")
ax.axhline( 2, color="k", lw=0.8, ls=":")
ax.axhline(-2, color="k", lw=0.8, ls=":")
ax.set_title("Unconstrained vs projected (box [−2,2])")
ax.set_xlabel("iteration"); ax.set_ylabel("x value")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5: mini-batch comparison
ax = axes[1, 1]
ax.semilogy(f_history(traj_t10),   label="t = 10 (fixed)")
ax.semilogy(f_history(traj_titer), label="t = iteration")
ax.set_title("Fixed t vs growing t (ZOGD, Gaussian)")
ax.set_xlabel("iteration"); ax.set_ylabel("f(x)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# 6: sphere + full B + projection
ax = axes[1, 2]
ax.semilogy(f_history(traj_sph_full), label="ZOGD (sphere, B full)")
ax.semilogy(f_history(traj_eg_full),  label="ZOEGm (sphere, B full, proj)")
ax.set_title("Sphere oracle + full B\n(one unconstrained, one projected)")
ax.set_xlabel("iteration"); ax.set_ylabel("f(x)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
