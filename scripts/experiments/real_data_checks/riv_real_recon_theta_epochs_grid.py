"""Small-multiples grid of theta trajectories for the real-recon signal.

As riv_sampling_rate_theta_epochs_grid.py, but for the real-recon weighted
multisine: one figure with every fs's epoch-0 trajectory side by side and one
with every fs's final-epoch trajectory, for comparison with the 3-tone signal.
Reuses collect() and FS_POINTS from riv_real_recon_theta_epochs_check.py.

Run: python3 scripts/experiments/real_data_checks/riv_real_recon_theta_epochs_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riv_real_recon_theta_epochs_check import collect, FS_POINTS, OUT_DIR

LABELS = ["a1", "a2", "b1", "b2"]


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
    runs = [collect(fs) for fs in FS_POINTS]

    plot_grid(runs, "epoch0",
              "Epoch 0 parameter trajectories across fs (real-recon weighted-multisine, dotted=true)",
              OUT_DIR / "epoch0_grid_vs_fs.png")
    plot_grid(runs, "final",
              f"Final epoch ({runs[0]['final_epoch_idx']}) parameter trajectories across fs "
              "(real-recon weighted-multisine, dotted=true)",
              OUT_DIR / "final_epoch_grid_vs_fs.png")
    print(f"saved: {OUT_DIR / 'epoch0_grid_vs_fs.png'}")
    print(f"saved: {OUT_DIR / 'final_epoch_grid_vs_fs.png'}")


if __name__ == "__main__":
    main()
