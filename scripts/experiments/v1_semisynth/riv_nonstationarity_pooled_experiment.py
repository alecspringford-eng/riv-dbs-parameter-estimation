"""
Experiment: Entry 19's own next step. Per-recording, ~7-window linear
regression of k vs time was inconclusive (0/6 recordings cleared R^2>0.3,
but 3/6 showed a same-direction R^2~0.25 trend) -- too few points per
recording for real power. This pools early-vs-late window k across ALL 6
recordings (normalized per-recording so different baseline k's don't
swamp the comparison) for a single higher-power test: is there a
consistent, cross-recording early-vs-late shift at all?

Reuses riv_nonstationarity_experiment's window machinery (V2 sliding-window
fits, same trust filter) -- adds pooling + a paired significance test
(Wilcoxon signed-rank) instead of stopping at per-recording R^2.

main() untouched -- report only. See INVESTIGATION.md Entry 24.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy import stats

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v2"))
import riv_dual_estimation as R
import riv_v2_greybox as V2
from riv_nonstationarity_experiment import make_windows

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"


def main() -> None:
    cfg = R.CFG
    recordings = R.load_all_recordings(cfg)

    early_norm, late_norm, per_recording = [], [], []
    for name in cfg.system_id_files:
        u, y = recordings[name]
        N = len(u)
        windows = make_windows(N)
        k_vals, uy_vals = [], []
        for s, e in windows:
            with contextlib.redirect_stdout(io.StringIO()):
                f = V2.identify(u[s:e], y[s:e])
            k_vals.append(f.k); uy_vals.append(f.uy_corr)
        k_vals = np.array(k_vals)
        trusted = np.abs(np.array(uy_vals)) >= V2.UY_COHERENCE_FLOOR
        k_trust = k_vals[trusted]
        if len(k_trust) < 4:
            print(f"{name}: only {len(k_trust)} trusted windows, skipped (need >=4 for early/late split)")
            continue
        half = len(k_trust) // 2
        early, late = k_trust[:half], k_trust[-half:]
        med = np.median(k_trust)
        early_n, late_n = early / med, late / med  # normalize by this recording's own median k
        early_norm.extend(early_n); late_norm.extend(late_n)
        per_recording.append({"name": name, "early_median": float(np.median(early)),
                               "late_median": float(np.median(late)),
                               "pct_change": 100 * (np.median(late) - np.median(early)) / med})
        print(f"{name}: {len(k_trust)} trusted windows, early_median={np.median(early):.2f}, "
              f"late_median={np.median(late):.2f}, change={per_recording[-1]['pct_change']:+.1f}% "
              f"of recording median k")

    n_recordings_pos = sum(1 for r in per_recording if r["pct_change"] > 0)
    print(f"\n{n_recordings_pos}/{len(per_recording)} recordings show k INCREASING from early to late "
          f"(direction consistency check)")

    # paired test on per-recording (early_median, late_median) pairs -- most honest
    # unit of analysis (6 independent recordings), not pooled individual windows
    # (which are correlated within a recording, would overstate power).
    early_meds = np.array([r["early_median"] for r in per_recording])
    late_meds = np.array([r["late_median"] for r in per_recording])
    if len(early_meds) >= 4:
        stat, p = stats.wilcoxon(early_meds, late_meds)
        print(f"\nWilcoxon signed-rank test (per-recording early vs late median k, "
              f"n={len(early_meds)} recordings): stat={stat:.2f}, p={p:.4f}")
    else:
        print(f"\nToo few recordings ({len(early_meds)}) with enough trusted windows for a "
              f"paired test at this sample size.")

    print(f"\nmean |pct_change| across recordings: "
          f"{np.mean([abs(r['pct_change']) for r in per_recording]):.1f}%")
    print("(for reference: Entry 19's per-recording linear-fit R^2 range was 0.002-0.264, "
          "0/6 recordings >0.3)")


if __name__ == "__main__":
    main()
