"""
Experiment: SEMISYNTHETIC_VS_REAL_REPORT.md §12 item 5 / INVESTIGATION.md
Entry 26 next step. V4's real-data tau1/tau2 robustly collapse to
near-zero (0.002-0.03s) on the 4 informative recordings -- confirmed a
genuine global ML optimum (4-init check, 67.6-nat profile-likelihood
cost), not a bug. Two open readings: (a) the variance-channel suppression
response genuinely IS near-instantaneous, or (b) the model is exploiting
real u's own sample-to-sample jitter to explain moment-to-moment variance
swings that are actually just noise, via a near-instant gate.

Test: apply a CAUSAL moving-average smoother to u (only past+current
samples -- no lookahead) at increasing window sizes, refit V4 (pure MLE,
no tau prior) on each smoothed version. If tau recovers toward the
physical range as smoothing increases (fast jitter removed, slower
structure preserved), that favors reading (b) -- the near-zero tau was
tracking noise. If tau stays near-zero even under heavy smoothing, that
favors reading (a) -- genuine fast dynamics, not a jitter artifact.

main() untouched -- report only. See INVESTIGATION.md Entry 27.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v4"))
import riv_dual_estimation as R
import riv_v4_coupled_suppression as V4

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
WINDOWS = (1, 3, 5, 10, 20, 40)  # samples; 1 = unsmoothed (reference)
INFORMATIVE_RECORDINGS = ("system_id_3.csv", "system_id_4.csv",
                           "system_id_5.csv", "system_id_long_run.csv")


def causal_moving_average(u: np.ndarray, window: int) -> np.ndarray:
    """Causal (past+current only, no lookahead) moving average -- avoids
    leaking future samples into the smoothed input, which would make the
    test's causality/timing not match a real online estimator's."""
    if window <= 1:
        return u.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate((np.full(window - 1, u[0]), u))
    return np.convolve(padded, kernel, mode="valid")


def _check_causal_ma() -> None:
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = causal_moving_average(u, 1)
    assert np.allclose(out, u)
    out3 = causal_moving_average(u, 3)
    assert len(out3) == len(u)
    assert np.isclose(out3[-1], (3.0 + 4.0 + 5.0) / 3)  # last window uses only past+current
    assert np.isclose(out3[0], 1.0)  # first sample: padded with u[0], average = u[0]


def main() -> None:
    _check_causal_ma()
    cfg = R.CFG
    recs = R.load_all_recordings(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        _, phi_ar, _ = V4._pooled_ar()

    rows = []
    for name in INFORMATIVE_RECORDINGS:
        u_raw, y = recs[name]
        print(f"{name}:")
        for w in WINDOWS:
            u_smooth = causal_moving_average(u_raw, w)
            jitter_removed = 1 - np.std(np.diff(u_smooth)) / np.std(np.diff(u_raw))
            f = V4.fit_v4(u_smooth, y, phi_ar)
            print(f"  window={w:3d} (jitter removed {jitter_removed:5.1%}): "
                  f"k={f.k:7.2f}  tau1={f.tau1:.4f}  tau2={f.tau2:.4f}  "
                  f"NIS={f.nis_mean:.3f}  nrmse={f.nrmse:.3f}  success={f.success}")
            rows.append({"name": name, "window": w, "jitter_removed": jitter_removed,
                         "k": f.k, "tau1": f.tau1, "tau2": f.tau2, "nis": f.nis_mean})

    print("\n=== Summary: does tau grow with smoothing window? ===")
    for name in INFORMATIVE_RECORDINGS:
        sub = [r for r in rows if r["name"] == name]
        tau1_w1 = sub[0]["tau1"]; tau1_wmax = sub[-1]["tau1"]
        tau2_w1 = sub[0]["tau2"]; tau2_wmax = sub[-1]["tau2"]
        print(f"{name}: tau1 {tau1_w1:.4f} -> {tau1_wmax:.4f} (window 1->{WINDOWS[-1]}), "
              f"tau2 {tau2_w1:.4f} -> {tau2_wmax:.4f}")

    names_seen = INFORMATIVE_RECORDINGS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in names_seen:
        sub = [r for r in rows if r["name"] == name]
        w = [r["window"] for r in sub]
        axes[0].plot(w, [r["tau1"] for r in sub], marker="o", label=name.removesuffix(".csv"))
        axes[1].plot(w, [r["tau2"] for r in sub], marker="o", label=name.removesuffix(".csv"))
        axes[2].plot(w, [r["k"] for r in sub], marker="o", label=name.removesuffix(".csv"))
    axes[0].axhline(R.AVG_PATIENT_PARAMS["tau1"], color="k", lw=0.8, ls=":", label="avg-patient prior")
    axes[1].axhline(R.AVG_PATIENT_PARAMS["tau2"], color="k", lw=0.8, ls=":", label="avg-patient prior")
    axes[0].set_xlabel("smoothing window (samples)"); axes[0].set_ylabel("tau1 (s)")
    axes[1].set_xlabel("smoothing window (samples)"); axes[1].set_ylabel("tau2 (s)")
    axes[2].set_xlabel("smoothing window (samples)"); axes[2].set_ylabel("k")
    axes[0].set_title("tau1 vs u-smoothing"); axes[1].set_title("tau2 vs u-smoothing")
    axes[2].set_title("k vs u-smoothing (should stay stable)")
    axes[0].legend(fontsize=7)
    fig.suptitle("V4: does causally-smoothing u (removing fast jitter) restore physical tau?")
    fig.tight_layout()
    out_path = FIG_DIR / f"v4_u_smoothing.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
