"""Verify that the dotted "true theta" lines in the theta_epochs plots match
cfg.k_true/tau1_true/tau2_true, two ways:

1. Round trip (no simulation, no noise): for every fs used in both sweeps,
   params_to_theta(k_true, tau1_true, tau2_true, Ts) is fed through
   theta_to_params (its analytic inverse) and compared to the original,
   isolating the discretization math from estimator behavior.

2. Physical-parameter trajectories: the theta_hat_by_epoch data from the epoch
   grids, converted from (a1,a2,b1,b2) to (k,tau1,tau2) with a vectorized
   theta_to_params (the same z^2-a1*z-a2=0 root formula applied elementwise,
   since per-sample np.roots over 80000 samples x 5 epochs would be slow).
   NaN where theta is not two real stable poles (common in the early
   transient). Dashed lines are cfg.k_true/tau1_true/tau2_true directly, so
   this also exercises the Config constants.

Run: python3 scripts/experiments/real_data_checks/riv_theta_to_physical_verify_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/sampling_rate"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
from riv_dual_estimation import params_to_theta, theta_to_params
from riv_sampling_rate_theta_epochs_check import FS_POINTS as FS_3TONE
from riv_sampling_rate_theta_epochs_grid import collect as collect_3tone
from riv_real_recon_theta_epochs_check import FS_POINTS as FS_REAL, collect as collect_real
from riv_instrument_update_rate_experiment import Config

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "theta_to_physical_verify"


def roundtrip_check(fs_values: list[float], label: str) -> None:
    cfg = Config()
    print(f"\n{label}: params_to_theta -> theta_to_params round-trip "
          f"(k_true={cfg.k_true}, tau1_true={cfg.tau1_true}, tau2_true={cfg.tau2_true})")
    for fs in fs_values:
        Ts = 1.0 / fs
        theta_true = params_to_theta(cfg.k_true, cfg.tau1_true, cfg.tau2_true, Ts)
        k_rt, tau1_rt, tau2_rt = theta_to_params(theta_true, Ts)
        ok = (abs(k_rt - cfg.k_true) < 1e-9 and abs(tau1_rt - cfg.tau1_true) < 1e-9
              and abs(tau2_rt - cfg.tau2_true) < 1e-9)
        print(f"  fs={fs:7.3f}Hz: k={k_rt:.10f} tau1={tau1_rt:.10f} tau2={tau2_rt:.10f}  "
              f"{'OK exact' if ok else 'MISMATCH'}")


def theta_traj_to_physical(theta_traj: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized theta_to_params: the closed-form root of z^2-a1*z-a2=0 applied
    elementwise. NaN where poles are not two real values in (0,1)."""
    a1, a2, b1, b2 = theta_traj[:, 0], theta_traj[:, 1], theta_traj[:, 2], theta_traj[:, 3]
    disc = a1 ** 2 + 4 * a2
    valid = disc >= 0
    sqrt_disc = np.sqrt(np.clip(disc, 0, None))
    p1 = (a1 + sqrt_disc) / 2
    p2 = (a1 - sqrt_disc) / 2
    valid &= (p1 > 0) & (p1 < 1) & (p2 > 0) & (p2 < 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        tau_a = -Ts / np.log(np.where(valid, p1, np.nan))
        tau_b = -Ts / np.log(np.where(valid, p2, np.nan))
    tau1 = np.minimum(tau_a, tau_b)
    tau2 = np.maximum(tau_a, tau_b)
    k = np.where(valid, (b1 + b2) / (1.0 - a1 - a2), np.nan)
    return k, tau1, tau2


def plot_physical_epochs(run_final_ref: dict, label: str, fs: float, path: Path) -> None:
    """One subplot per selected epoch: k_hat on the left axis (dashed=k_true),
    tau1_hat/tau2_hat on the right (dashed=tau_true). Like
    plot_theta_epochs_stacked, in physical rather than theta space."""
    epochs = run_final_ref["selected_epochs"]
    t = run_final_ref["t"]
    cfg = Config()
    fig, axes = plt.subplots(len(epochs), 1, figsize=(8, 2.6 * len(epochs)), sharex=True)
    for row, epoch in enumerate(epochs):
        ax = axes[row]
        k, tau1, tau2 = theta_traj_to_physical(run_final_ref["theta_hat_by_epoch"][epoch], 1.0 / fs)
        ax.plot(t, k, color="tab:blue", lw=1, label="k_hat")
        ax.axhline(cfg.k_true, color="tab:blue", ls=":", lw=1)
        ax.set_ylabel("k_hat", color="tab:blue")

        ax2 = ax.twinx()
        ax2.plot(t, tau1, color="tab:orange", lw=1, label="tau1_hat")
        ax2.plot(t, tau2, color="tab:green", lw=1, label="tau2_hat")
        ax2.axhline(cfg.tau1_true, color="tab:orange", ls=":", lw=1)
        ax2.axhline(cfg.tau2_true, color="tab:green", ls=":", lw=1)
        ax2.set_ylabel("tau_hat (s)")
        ax.set_title(f"epoch {epoch}")
        if row == 0:
            lines = [l for l in ax.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
            ax.legend(lines, [l.get_label() for l in lines], ncol=3, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{label}: physical (k,tau1,tau2) trajectories vs true, fs={fs}Hz "
                 f"(dotted=cfg.k_true/tau1_true/tau2_true)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> None:
    roundtrip_check(FS_3TONE, "3-tone sweep fs grid")
    roundtrip_check(FS_REAL, "real-recon sweep fs grid")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fs = 50.0
    r3 = collect_3tone(fs)
    plot_physical_epochs(r3, "3-tone signal", fs, OUT_DIR / "physical_epochs_3tone_fs50.png")
    rr = collect_real(fs)
    plot_physical_epochs(rr, "real-recon weighted-multisine", fs,
                          OUT_DIR / "physical_epochs_realrecon_fs50.png")
    print(f"\nsaved: {OUT_DIR / 'physical_epochs_3tone_fs50.png'}")
    print(f"saved: {OUT_DIR / 'physical_epochs_realrecon_fs50.png'}")


if __name__ == "__main__":
    main()
