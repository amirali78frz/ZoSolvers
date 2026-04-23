"""
Demo script for ZO_gauss_minmax (minimax module).

Run with:
    poetry run python tests/demo_minimax.py

Demonstrates:
  - ZOGDA and ZOEGmm with Gaussian and sphere oracles
  - Diagonal and full B_x / B_y matrices
  - Box-projected (constrained) variants
  - Forward / backward / centered finite-difference methods
  - Fixed vs growing (t="iteration") mini-batch size
  - tau (step-size ratio) effect
  - tol early stopping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ZoSolvers.minimax import ZO_gauss_minmax


# ---------------------------------------------------------------------------
# Problem:
#   f(x, y) = x0^2 + x1^2 + x0*x1  -  y0^2 - y1^2 + 1.5*y0*y1 + 3*y1
#
# This is the same function from the original demo file.
# Saddle point can be found by solving ∇_x f = 0, ∇_y f = 0.
# ---------------------------------------------------------------------------

def f(x, y):
    return (  x[0]**2 + x[1]**2 + x[0]*x[1]
            - y[0]**2 - y[1]**2 + 1.5*y[0]*y[1] + 3*y[1])

x0   = np.array([10.0, -10.0])
y0   = np.array([ 9.0,  -9.0])
proj = lambda v: np.clip(v, -7.0, 7.0)

B_diag = np.diag([2.0, 0.2])
B_full = np.array([[2.0, 0.01], [0.01, 0.2]])

SEP = "=" * 70

def report(label, x_traj, y_traj):
    norm_x0 = np.linalg.norm(x_traj[0])
    norm_xf = np.linalg.norm(x_traj[-1])
    norm_y0 = np.linalg.norm(y_traj[0])
    norm_yf = np.linalg.norm(y_traj[-1])
    print(f"  {label:<45s}  iters={len(x_traj):>6d}")
    print(f"    ||x||: {norm_x0:.3f} → {norm_xf:.4f}   "
          f"||y||: {norm_y0:.3f} → {norm_yf:.4f}")

def f_vals(x_traj, y_traj):
    return [f(x, y) for x, y in zip(x_traj, y_traj)]

# Shared solver settings
KW = dict(h=1e-3, tau=1, mu=1e-8, N=10000, t=5)

# ===========================================================================
print(SEP)
print("ZOGDA — oracle type comparison (no B, centered diff)")
print(SEP)
traj_gx, traj_gy = ZO_gauss_minmax(f, x0, y0, **KW,
                                    oracle_type="gaussian").ZOGDA(method="center")
traj_sx, traj_sy = ZO_gauss_minmax(f, x0, y0, **KW,
                                    oracle_type="sphere"  ).ZOGDA(method="center")
report("Gaussian oracle", traj_gx, traj_gy)
report("Sphere oracle",   traj_sx, traj_sy)
print()

# ===========================================================================
print(SEP)
print("ZOGDA — finite-difference method comparison (Gaussian, no B)")
print(SEP)
for method in ("forw", "back", "center"):
    xx, yy = ZO_gauss_minmax(f, x0, y0, **KW).ZOGDA(method=method)
    report(f"method = {method}", xx, yy)
print()

# ===========================================================================
print(SEP)
print("ZOGDA — B matrix comparison (Gaussian, centered diff)")
print(SEP)
traj_Ix, traj_Iy = ZO_gauss_minmax(f, x0, y0, **KW).ZOGDA(method="center")
traj_dx, traj_dy = ZO_gauss_minmax(f, x0, y0, **KW,
                                    B_x=B_diag, B_y=B_diag).ZOGDA(method="center")
traj_fx, traj_fy = ZO_gauss_minmax(f, x0, y0, **KW,
                                    B_x=B_full, B_y=B_full).ZOGDA(method="center")
report("B = I (no B)",  traj_Ix, traj_Iy)
report("B diagonal",    traj_dx, traj_dy)
report("B full",        traj_fx, traj_fy)
print()

# ===========================================================================
print(SEP)
print("ZOGDA — sphere oracle with B (centered diff)")
print(SEP)
traj_sdx, traj_sdy = ZO_gauss_minmax(f, x0, y0, **KW,
                                      B_x=B_diag, B_y=B_diag,
                                      oracle_type="sphere").ZOGDA(method="center")
traj_sfx, traj_sfy = ZO_gauss_minmax(f, x0, y0, **KW,
                                      B_x=B_full, B_y=B_full,
                                      oracle_type="sphere").ZOGDA(method="center")
report("Sphere + B diagonal", traj_sdx, traj_sdy)
report("Sphere + B full",     traj_sfx, traj_sfy)
print()

# ===========================================================================
print(SEP)
print("ZOGDA — tau (step-size ratio x vs y) comparison")
print(SEP)
for tau in (0.5, 1.0, 2.0):
    xx, yy = ZO_gauss_minmax(f, x0, y0, **{**KW, 'tau': tau}).ZOGDA(method="center")
    report(f"tau = {tau}", xx, yy)
print()

# ===========================================================================
print(SEP)
print("ZOGDA — unconstrained vs projected (box [-7, 7]^2)")
print(SEP)
traj_ux, traj_uy = ZO_gauss_minmax(f, x0, y0, **KW).ZOGDA(method="center")
traj_px, traj_py = ZO_gauss_minmax(f, x0, y0, **KW,
                                    proj_x=proj, proj_y=proj).ZOGDA(method="center")
report("Unconstrained", traj_ux, traj_uy)
report("Projected",     traj_px, traj_py)
fx = np.all(traj_px >= -7 - 1e-9) and np.all(traj_px <= 7 + 1e-9)
fy = np.all(traj_py >= -7 - 1e-9) and np.all(traj_py <= 7 + 1e-9)
print(f"  x iterates inside box: {fx},  y iterates inside box: {fy}")
print()

# ===========================================================================
print(SEP)
print("ZOGDA — tol early stopping")
print(SEP)
opt_tol = ZO_gauss_minmax(f, x0, y0, **{**KW, 'N': 50000, 'tol': 1e-5})
traj_tx, traj_ty = opt_tol.ZOGDA(method="center")
report("tol = 1e-5", traj_tx, traj_ty)
print(f"  Stopped early (before N=50000): {len(traj_tx) < 50000}")
print()

# ===========================================================================
print(SEP)
print("ZOEGmm — Gaussian oracle, different gamma (no B)")
print(SEP)
opt_eg = ZO_gauss_minmax(f, x0, y0, **KW)
for gamma in (1.0, 0.8, 0.5):
    xx, yy = opt_eg.ZOEGmm(method="center", gamma=gamma)
    report(f"gamma = {gamma}", xx, yy)
print()

# ===========================================================================
print(SEP)
print("ZOEGmm — Sphere oracle + full B + projection")
print(SEP)
traj_egx, traj_egy = ZO_gauss_minmax(f, x0, y0, **KW,
                                      B_x=B_full, B_y=B_full,
                                      oracle_type="sphere",
                                      proj_x=proj, proj_y=proj).ZOEGmm(method="center")
report("Sphere + full B + proj", traj_egx, traj_egy)
print()

# ===========================================================================
# t = "iteration"
# ===========================================================================
print(SEP)
print("ZOEGmm — t = 'iteration' (growing mini-batch)")
print(SEP)
traj_itx, traj_ity = ZO_gauss_minmax(f, x0, y0,
                                       h=1e-3, tau=1, mu=1e-8,
                                       N=200, t="iteration").ZOEGmm(method="center")
report("t = iteration", traj_itx, traj_ity)
print()

# ===========================================================================
# Plots
# ===========================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    r"ZO Minimax — $f(x,y)=x_0^2+x_1^2+x_0x_1-y_0^2-y_1^2+1.5y_0y_1+3y_1$",
    fontsize=12)

def norm_traj(traj):
    return [np.linalg.norm(v) for v in traj]

# 1: oracle type — ||x|| trajectory
ax = axes[0, 0]
ax.semilogy(norm_traj(traj_gx), label="Gaussian — x")
ax.semilogy(norm_traj(traj_gy), label="Gaussian — y", linestyle="--")
ax.semilogy(norm_traj(traj_sx), label="Sphere — x",   color="C2")
ax.semilogy(norm_traj(traj_sy), label="Sphere — y",   color="C2", linestyle="--")
ax.set_title("Oracle type (ZOGDA, no B)")
ax.set_xlabel("iteration"); ax.set_ylabel("||·||")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

# 2: B matrix — f(x,y) trajectory
ax = axes[0, 1]
ax.plot(f_vals(traj_Ix, traj_Iy), label="B = I")
ax.plot(f_vals(traj_dx, traj_dy), label="B diagonal")
ax.plot(f_vals(traj_fx, traj_fy), label="B full")
ax.set_title("B matrix (ZOGDA, Gaussian)")
ax.set_xlabel("iteration"); ax.set_ylabel("f(x, y)")
ax.legend(); ax.grid(True, alpha=0.3)

# 3: ZOGDA vs ZOEGmm
ax = axes[0, 2]
eg1x, eg1y = opt_eg.ZOEGmm(method="center", gamma=1.0)
ax.semilogy(norm_traj(traj_Ix),  label="ZOGDA — x")
ax.semilogy(norm_traj(traj_Iy),  label="ZOGDA — y", linestyle="--")
ax.semilogy(norm_traj(eg1x),     label="ZOEGmm γ=1 — x",   color="C2")
ax.semilogy(norm_traj(eg1y),     label="ZOEGmm γ=1 — y",   color="C2", linestyle="--")
ax.set_title("ZOGDA vs ZOEGmm (Gaussian, no B)")
ax.set_xlabel("iteration"); ax.set_ylabel("||·||")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

# 4: projection — x and y iterate trajectories
ax = axes[1, 0]
ax.plot(traj_ux[:, 0], label="uncon x₀", alpha=0.7)
ax.plot(traj_ux[:, 1], label="uncon x₁", alpha=0.7)
ax.plot(traj_px[:, 0], label="proj x₀",  linestyle="--")
ax.plot(traj_px[:, 1], label="proj x₁",  linestyle="--")
ax.axhline( 7, color="k", lw=0.8, ls=":")
ax.axhline(-7, color="k", lw=0.8, ls=":")
ax.set_title("Unconstrained vs projected x (box [−7,7])")
ax.set_xlabel("iteration"); ax.set_ylabel("x value")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5: gamma comparison (ZOEGmm)
ax = axes[1, 1]
for gamma in (1.0, 0.8, 0.5):
    xx, yy = opt_eg.ZOEGmm(method="center", gamma=gamma)
    ax.semilogy(norm_traj(xx), label=f"γ={gamma} — x")
ax.set_title("ZOEGmm: effect of gamma (Gaussian, no B)")
ax.set_xlabel("iteration"); ax.set_ylabel("||x||")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# 6: sphere + full B + proj
ax = axes[1, 2]
ax.semilogy(norm_traj(traj_sfx), label="ZOGDA (sphere, B full) — x")
ax.semilogy(norm_traj(traj_sfy), label="ZOGDA (sphere, B full) — y", linestyle="--")
ax.semilogy(norm_traj(traj_egx), label="ZOEGmm (sphere, B full, proj) — x",   color="C2")
ax.semilogy(norm_traj(traj_egy), label="ZOEGmm (sphere, B full, proj) — y",   color="C2", linestyle="--")
ax.set_title("Sphere oracle + full B\n(one unconstrained, one projected)")
ax.set_xlabel("iteration"); ax.set_ylabel("||·||")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
