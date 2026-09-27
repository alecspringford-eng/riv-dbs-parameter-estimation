"""
Direct visualization of the (tau1,tau2) likelihood surface, using the
already-validated `_v1_forward_kf_nll` (fixed-theta forward KF pass, no
RIV) -- reused as-is, only theta_fixed is swept over a grid instead of
R/q_eta. k is held at its Entry 32-converged value (well-identified,
Phase 4 FIM) so this isolates the (tau1,tau2) confounding specifically.

Motivation: Entry 32 found real data splits into two groups -- 3/6
recordings converge to distinct non-floor (tau1,tau2), 3/6 collapse to
the tau_min clamp floor. This should show up as a qualitatively different
NLL surface shape: a genuine (if shallow/elongated) interior minimum for
the "converged" group vs a monotonically-decreasing-toward-the-corner
surface for the "collapsed" group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import (
    CFG, DUAL_CFG, N_ETA, load_all_recordings, fit_ar_beta_model_per_recording,
    build_kf_setup, _v1_forward_kf_nll,
)
from riv_v1_logparam_jit import _theta_from_kt1t2, POLE_MIN, POLE_MAX


def nll_surface(u, y, phi_coeffs, sigma2_ar, cfg, dual_cfg, k_fixed: float,
                 tau1_grid: np.ndarray, tau2_grid: np.ndarray) -> np.ndarray:
    setup = build_kf_setup(u, y, phi_coeffs, sigma2_ar, 0.5 * float(np.var(y)), cfg, dual_cfg)
    u_norm, u_scale, x0, P0_kf, p = (setup["u_norm"], setup["u_scale"], setup["x0"],
                                      setup["P0_kf"], setup["p"])
    R_heur = 0.05 * sigma2_ar
    q_eta_heur = 0.5 * float(np.var(y))
    q_off = dual_cfg.q_off_frac * float(np.var(y))
    params = np.array([np.log(R_heur), np.log(q_eta_heur)])

    nll = np.full((len(tau1_grid), len(tau2_grid)), np.nan)
    for i, t1 in enumerate(tau1_grid):
        for j, t2 in enumerate(tau2_grid):
            a1, a2, b1, b2 = _theta_from_kt1t2(k_fixed, t1, t2, cfg.Ts)
            theta_fixed = np.array([a1, a2, b1 * u_scale, b2 * u_scale])
            val, _, _ = _v1_forward_kf_nll(params, u_norm, y, phi_coeffs, sigma2_ar,
                                            N_ETA, p, theta_fixed, x0, P0_kf, q_off)
            nll[i, j] = val
    return nll


def plot_ridge(name: str, k_fixed: float, tau1_conv: float, tau2_conv: float,
               tau1_grid: np.ndarray, tau2_grid: np.ndarray, nll: np.ndarray, path: Path) -> None:
    tau_min = -CFG.Ts / np.log(POLE_MIN)
    tau_max = -CFG.Ts / np.log(POLE_MAX)
    fig, ax = plt.subplots(figsize=(6, 5))
    finite = nll[np.isfinite(nll)]
    lo = np.percentile(finite, 1)
    levels = np.linspace(lo, np.percentile(finite, 90), 25)
    cs = ax.contourf(tau2_grid, tau1_grid, np.clip(nll, lo, np.percentile(finite, 90)),
                      levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="NLL (fixed k, R, q_eta)")
    ax.scatter([tau2_conv], [tau1_conv], color="red", marker="x", s=80,
               label=f"converged (tau1={tau1_conv:.4f}, tau2={tau2_conv:.4f})")
    ax.axhline(tau_min, color="white", ls=":", lw=1, label=f"tau_min floor ({tau_min:.4f}s)")
    ax.axvline(tau_min, color="white", ls=":", lw=1)
    ax.set_xlabel("tau2 (s)"); ax.set_ylabel("tau1 (s)")
    ax.set_title(f"{name}: (tau1,tau2) NLL surface at k={k_fixed:.1f}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run(cfg, dual_cfg, out_dir: Path = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_logparam") -> None:
    from riv_v1_logparam_jit import run_identification_logparam_jit
    recordings = load_all_recordings(cfg)
    ar_fits = fit_ar_beta_model_per_recording(recordings, cfg)

    # one recording that converged to a non-floor (tau1,tau2) (Entry 32),
    # one that collapsed to the tau_min floor
    targets = ["system_id_4.csv", "system_id_5.csv"]
    tau_min = -cfg.Ts / np.log(POLE_MIN)
    tau_max = -cfg.Ts / np.log(POLE_MAX)
    tau1_grid = np.geomspace(tau_min, 0.15, 40)
    tau2_grid = np.geomspace(tau_min, 0.15, 40)

    for name in targets:
        key = name.replace(".csv", "")
        u, y = recordings[name]
        f = ar_fits[name]
        res = run_identification_logparam_jit(u, y, f["phi_coeffs"], f["sigma2_ar"],
                                               0.05 * f["sigma2_ar"], 0.5 * float(np.var(y)),
                                               cfg, dual_cfg, n_epochs=200)
        print(f"{key}: converged k={res['k']:.2f} tau1={res['tau1']:.5f} tau2={res['tau2']:.5f}")
        nll = nll_surface(u, y, f["phi_coeffs"], f["sigma2_ar"], cfg, dual_cfg,
                           res["k"], tau1_grid, tau2_grid)
        rec_dir = out_dir / key
        rec_dir.mkdir(parents=True, exist_ok=True)
        plot_ridge(key, res["k"], res["tau1"], res["tau2"], tau1_grid, tau2_grid, nll,
                   rec_dir / "tau_nll_surface.png")
        finite = nll[np.isfinite(nll)]
        i_min = np.unravel_index(np.nanargmin(nll), nll.shape)
        print(f"  grid min NLL={finite.min():.3f} at tau1={tau1_grid[i_min[0]]:.5f}, "
              f"tau2={tau2_grid[i_min[1]]:.5f}; NLL range [{finite.min():.3f}, {finite.max():.3f}]")


if __name__ == "__main__":
    run(CFG, DUAL_CFG)
