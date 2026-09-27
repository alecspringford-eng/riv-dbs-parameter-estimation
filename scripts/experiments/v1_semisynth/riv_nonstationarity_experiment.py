"""
Experiment: SEMISYNTHETIC_VS_REAL_REPORT.md next step #1. Does the true
(k, tau) drift systematically within a single real recording (~2-4 min)?
V1's epoch loop assumes ONE fixed plant per recording; if the real response
adapts/habituates over the session, that alone would explain why even a
perfectly-specified estimator (V2/V3, no attractor problem) gets an
inconsistent answer -- not a bug in V1 specifically, a violation of the
stationarity assumption every version (V1/V2/V3) shares.

Method: slide a window across each real recording, fit V2's grey-box OE
model (riv_v2_greybox.identify, fast least_squares, no attractor -- unlike
V1, a bad window fit fails loudly via nrmse/uy_corr rather than silently
converging to a wrong fixed point) independently per window, and check
whether k trends with window center time.

main() untouched -- report only. See INVESTIGATION.md Entry 19.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v2"))
import riv_dual_estimation as R
import riv_v2_greybox as V2

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
N_WINDOWS_TARGET = 7  # approx, actual count depends on recording length


def make_windows(N: int) -> list[tuple[int, int]]:
    """~N_WINDOWS_TARGET overlapping (50%) windows spanning the recording,
    window length scaled to N so short and long recordings get a similar
    window count."""
    win_len = max(400, N // 4)
    step = max(1, win_len // 2)
    windows = []
    s = 0
    while s + win_len <= N:
        windows.append((s, s + win_len))
        s += step
    if not windows:
        windows = [(0, N)]
    return windows


def _check_windows() -> None:
    w = make_windows(2745)
    assert len(w) >= 3
    assert all(e - s >= 400 for s, e in w)
    assert all(e <= 2745 for s, e in w)


def main() -> None:
    _check_windows()
    cfg = R.CFG
    recordings = R.load_all_recordings(cfg)

    rows = []
    for name in cfg.system_id_files:
        u, y = recordings[name]
        N = len(u)
        windows = make_windows(N)
        t_center, k_vals, order_vals, nrmse_vals, uy_vals = [], [], [], [], []
        for s, e in windows:
            with contextlib.redirect_stdout(io.StringIO()):
                f = V2.identify(u[s:e], y[s:e])
            t_center.append((s + e) / 2 * cfg.Ts)
            k_vals.append(f.k); order_vals.append(f.order)
            nrmse_vals.append(f.nrmse); uy_vals.append(f.uy_corr)

        t_center, k_vals = np.array(t_center), np.array(k_vals)
        # linear trend in k vs window-center time, on windows V2 itself trusts
        # (not flagged low-coherence, matching V2's own data-quality gate)
        trust = np.array(uy_vals)
        trusted = np.abs(trust) >= V2.UY_COHERENCE_FLOOR
        if trusted.sum() >= 3:
            slope, intercept = np.polyfit(t_center[trusted], k_vals[trusted], 1)
            k_pred = slope * t_center[trusted] + intercept
            ss_res = np.sum((k_vals[trusted] - k_pred) ** 2)
            ss_tot = np.sum((k_vals[trusted] - k_vals[trusted].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            drift_over_session = slope * (t_center[trusted][-1] - t_center[trusted][0])
            pct_drift = 100 * drift_over_session / np.median(np.abs(k_vals[trusted]))
        else:
            slope, r2, pct_drift = float("nan"), float("nan"), float("nan")

        rows.append({"name": name, "n_windows": len(windows), "n_trusted": int(trusted.sum()),
                     "t_center": t_center, "k_vals": k_vals, "trusted": trusted,
                     "slope_k_per_s": slope, "r2": r2, "pct_drift_over_session": pct_drift})
        print(f"{name}: {len(windows)} windows ({int(trusted.sum())} trusted, uy_corr>= "
              f"{V2.UY_COHERENCE_FLOOR}), k range [{k_vals.min():.1f}, {k_vals.max():.1f}], "
              f"slope={slope:+.3f} k/s, R^2={r2:.3f}, total drift over session={pct_drift:+.1f}% of median|k|")

    n_significant_trend = sum(1 for r in rows if np.isfinite(r["r2"]) and r["r2"] > 0.3)
    print(f"\nrecordings with a clear monotone k-vs-time trend (R^2>0.3): "
          f"{n_significant_trend}/{len(rows)}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, r in zip(axes.flat, rows):
        t, k, trust = r["t_center"], r["k_vals"], r["trusted"]
        ax.plot(t[trust], k[trust], "o-", color="tab:blue", label="trusted (uy_corr ok)")
        ax.plot(t[~trust], k[~trust], "x", color="tab:red", label="flagged (low uy_corr)")
        if np.isfinite(r["slope_k_per_s"]):
            tt = np.linspace(t.min(), t.max(), 20)
            ax.plot(tt, r["slope_k_per_s"] * tt + (k[trust].mean() - r["slope_k_per_s"] * t[trust].mean()),
                    "--", color="gray", lw=1)
        ax.set_title(f"{r['name'].removesuffix('.csv')}\nslope={r['slope_k_per_s']:+.2f} k/s, "
                     f"R2={r['r2']:.2f}", fontsize=9)
        ax.set_xlabel("window center (s)"); ax.set_ylabel("k")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("V2 sliding-window k estimate vs time within each real recording "
                 "(non-stationarity test)")
    fig.tight_layout()
    out_path = FIG_DIR / f"nonstationarity_sliding_window.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
