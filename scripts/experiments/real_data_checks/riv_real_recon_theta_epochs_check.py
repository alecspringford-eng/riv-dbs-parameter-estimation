"""riv_real_recon_theta_epochs_check.py -- per-fs "parameter trajectories
across epochs" plots for the real-recon weighted-multisine sweep
(riv_real_recon_multisine_sweep_check.py), in the same stacked-epochs
layout as the OneDrive reference project's real-data plots
(MPC_linear/reference/Linear-models/RIV Kalman filter/real_data_output/
theta_epochs/*.png: one subplot per epoch, a1_hat/a2_hat/b1_hat/b2_hat
traces, dotted reference lines), except the dotted lines here are the
actual known-truth theta (this is a known-truth synthetic scenario, not
real data with an unknown truth) rather than the reference plots'
"final epoch's value" stand-in.

Reuses plot_theta_epochs_stacked UNMODIFIED from
riv_instrument_update_rate_experiment.py (lines 184-201) -- it already
plots exactly this (a1/a2/b1/b2_hat per epoch, dotted true value), no
changes needed. Also reuses Config/cfg_at_fs/make_scenario_real_recon/
run_dual_estimator_instrument from riv_real_recon_multisine_sweep_check.py
and riv_instrument_update_rate_experiment.py unmodified.

FS_POINTS below reproduces the exact fs grid that
riv_real_recon_multisine_sweep_check.py's completed sweep evaluated
(coarse descent + bisection + ascent), by the same closed-form arithmetic
(COARSE_RATIO=1/1.4 stepping, geometric-mean bisection) -- NOT by
re-running the expensive 30-seed ok-rate evaluation at each point, since
the fs VALUES themselves only depend on which branch the bisection took
at each step, and that's already known from the completed run's printed
log (both bisection midpoints failed the ok-rate>=0.5 bar, so fs_fail
moved twice and fs_pass never did -- confirmed by that run's own boundary
bracket [11.965, 13.015]Hz output). Each of these fs points here only
needs ONE seed=0 detailed run (cheap: n_snapshots=20, save_epochs), not
the 30-seed sweep, so this script is fast even though it covers every
sweep point.

Run: python3 scripts/riv_real_recon_theta_epochs_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root, for riv_* module imports

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
from riv_instrument_update_rate_experiment import (
    AUX_DECIMATED, run_dual_estimator_instrument, plot_theta_epochs_stacked,
)
from riv_real_recon_multisine_sweep_check import (
    Config, cfg_at_fs, make_scenario_real_recon, COARSE_RATIO,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "real_recon_theta_epochs"

# Reproduces riv_real_recon_multisine_sweep_check.py's completed sweep grid
# (see docstring for why this is exact without re-running the 30-seed eval).
_descent = [50.0 * COARSE_RATIO ** i for i in range(6)]  # 50 ... 9.296721604093533
_fs_pass, _fs_fail = _descent[4], _descent[5]
_mid1 = float(np.sqrt(_fs_pass * _fs_fail))
_fs_fail = _mid1
_mid2 = float(np.sqrt(_fs_pass * _fs_fail))
_bisection = [_mid1, _mid2]
_ascent = []
_fs = 50.0
while True:
    _fs /= COARSE_RATIO
    if _fs > 100.0:
        break
    _ascent.append(_fs)

FS_POINTS = sorted(_descent + _bisection + _ascent)


def collect(fs: float, seed: int = 0) -> dict:
    """Raw per-fs simulation data (shared by make_detailed_plot here and
    riv_real_recon_theta_epochs_grid.py's small-multiples figures)."""
    cfg = cfg_at_fs(fs)
    n_epochs = cfg.n_epochs
    selected_epochs = sorted(set(range(min(4, n_epochs))) | {n_epochs - 1})

    rng = np.random.default_rng(seed)
    scn = make_scenario_real_recon(cfg, rng)
    eta_aux_full = np.zeros(cfg.N_sim)  # unused placeholder, AUX_DECIMATED ignores it
    x_hat, theta_hat, y_hat, Pkf_final, Prls_final, theta_epoch_hist, theta_hat_by_epoch = \
        run_dual_estimator_instrument(
            scn["u"], scn["y_true"], scn["phi_c"], scn["Q_c"], cfg.R, scn["x0"], scn["P0_kf"],
            scn["theta_hat0"], scn["P0_rls"], cfg.lambda_ff, 2, scn["p"],
            AUX_DECIMATED, 1, eta_aux_full, n_epochs=n_epochs, n_snapshots=20,
            save_epochs=set(selected_epochs))

    t = np.arange(cfg.N_sim) * cfg.Ts
    theta_true = scn["theta_true"]
    final_err = float(np.linalg.norm(theta_hat[-1] - theta_true))
    return dict(fs=fs, t=t, theta_true=theta_true, selected_epochs=selected_epochs,
                theta_hat_by_epoch=theta_hat_by_epoch,
                epoch0=theta_hat_by_epoch[selected_epochs[0]],
                final=theta_hat_by_epoch[selected_epochs[-1]],
                final_epoch_idx=selected_epochs[-1], final_err=final_err)


def make_detailed_plot(fs: float, seed: int = 0) -> float:
    r = collect(fs, seed)
    fs_tag = f"fs{fs:07.3f}".replace(".", "p")
    out = OUT_DIR / fs_tag
    out.mkdir(parents=True, exist_ok=True)
    plot_theta_epochs_stacked(r["t"], r["selected_epochs"], r["theta_hat_by_epoch"], r["theta_true"],
                               out / "theta_epochs_stacked.png")

    print(f"  fs={fs:7.3f}Hz -> {out}/theta_epochs_stacked.png  (theta_err={r['final_err']:.4f})")
    return r["final_err"]


def main() -> None:
    print(f"fs grid ({len(FS_POINTS)} points, reproduced from the completed sweep): "
          f"{[round(f, 3) for f in FS_POINTS]}\n")
    for fs in FS_POINTS:
        make_detailed_plot(fs)


if __name__ == "__main__":
    main()
