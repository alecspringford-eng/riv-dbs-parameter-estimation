"""riv_sampling_rate_theta_epochs_grid.py -- two comparison figures for the
current 3-tone signal's fs sweep (same FS_POINTS grid as
riv_sampling_rate_theta_epochs_check.py): one small-multiples grid with
every fs's EPOCH 0 parameter trajectory side by side, one with every fs's
FINAL epoch (29) trajectory side by side -- so the fs-to-fs progression is
visible in one glance instead of paging through 10 separate stacked plots.

Reuses rate_mod.run_arm_for_seed UNMODIFIED (riv_instrument_update_rate_
experiment.py:476-492) to get the raw theta_hat_by_epoch/theta_true arrays
directly (run_arm_detailed itself only saves plots, doesn't return data).
Reuses ADAPTIVE_ARM/cfg_at_fs/FS_POINTS from riv_sampling_rate_
theta_epochs_check.py unmodified -- same fs grid, same seed=0 convention.

Run: python3 scripts/riv_sampling_rate_theta_epochs_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root, for riv_* module imports

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
import riv_instrument_update_rate_experiment as rate_mod
from riv_sampling_rate_sweep_experiment import ADAPTIVE_ARM, cfg_at_fs
from riv_sampling_rate_theta_epochs_check import FS_POINTS

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "sampling_rate_theta_epochs"
LABELS = ["a1", "a2", "b1", "b2"]


def collect(fs: float) -> dict:
    cfg = cfg_at_fs(fs)
    selected_epochs = sorted(set(range(min(4, cfg.n_epochs))) | {cfg.n_epochs - 1})
    scn, (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, theta_epoch_hist,
          theta_hat_by_epoch) = rate_mod.run_arm_for_seed(
        ADAPTIVE_ARM, cfg, seed=0, save_epochs=set(selected_epochs))
    t = np.arange(cfg.N_sim) * cfg.Ts
    return dict(fs=fs, t=t, theta_true=scn["theta_true"],
                selected_epochs=selected_epochs, theta_hat_by_epoch=theta_hat_by_epoch,
                epoch0=theta_hat_by_epoch[selected_epochs[0]],
                final=theta_hat_by_epoch[selected_epochs[-1]],
                final_epoch_idx=selected_epochs[-1])


def plot_grid(runs: list[dict], key: str, title: str, path: Path) -> None:
    n = len(runs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(runs):
        ax = axes[i // ncols, i % ncols]
        traj = r[key]
        theta_true = r["theta_true"]
        for j, lab in enumerate(LABELS):
            ax.plot(r["t"], traj[:, j], label=f"{lab}_hat", lw=1)
            ax.axhline(theta_true[j], color=ax.lines[-1].get_color(), ls=":", lw=1)
        ax.set_title(f"fs={r['fs']:.2f}Hz", fontsize=10)
        if i == 0:
            ax.legend(ncol=4, fontsize=7)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    runs = [collect(fs) for fs in FS_POINTS]  # FS_POINTS already ascending
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_grid(runs, "epoch0", "Epoch 0 parameter trajectories across fs (3-tone signal, dotted=true)",
               OUT_DIR / "epoch0_grid_vs_fs.png")
    plot_grid(runs, "final",
               f"Final epoch ({runs[0]['final_epoch_idx']}) parameter trajectories across fs "
               "(3-tone signal, dotted=true)",
               OUT_DIR / "final_epoch_grid_vs_fs.png")
    print(f"saved: {OUT_DIR / 'epoch0_grid_vs_fs.png'}")
    print(f"saved: {OUT_DIR / 'final_epoch_grid_vs_fs.png'}")


if __name__ == "__main__":
    main()
