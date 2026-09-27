"""
Derive an AR disturbance-model order/shape from a richer external dataset,
for the whiteness investigation (see INVESTIGATION.md, RESEARCH_NOTEBOOK.md).

Our own no-stim data is too thin to test whiteness meaningfully: 3/6
system_id recordings have zero usable no-stim segments, the rest have one
125-sample segment -- at that sample size the whiteness test has ~zero
statistical power (order 3 through 15 all trivially "pass" with
frac_exceed=0.000, see INVESTIGATION.md Entry 10).

External data: a DIFFERENT recording (nndbsProject, 50 Hz, different
session/subject than real_recordings's 25 Hz), with 4 true zero-stim
segments of 5000-11000 samples each -- real statistical power. This script
fits the AR order/shape on that data, downsamples exactly to our 25 Hz via
resample_ar_model (matrix-square, no approximation), and caches the result
to external_ar_model.npz so production code never needs the external .mat file.

NOTE: this is a cross-dataset structural probe, not a claim the two
recordings share identical physiology -- flagged throughout downstream use.
Re-run this script only if the source .mat file changes; the cache is what
everything else reads.
"""
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R

# Path to the external .mat file, supplied by the user -- not included in this
# repo. Set via the EXTERNAL_AR_MAT env var, or edit the fallback below.
EXTERNAL_MAT = Path(os.environ.get("EXTERNAL_AR_MAT", "external_data/dbs_segments.mat"))
CACHE_PATH = Path(__file__).parent / "external_ar_model.npz"
EXTERNAL_TS = 0.02  # 50 Hz
TRIM_SAMPLES = 250  # 5s @ 50Hz, matches the source study's own trim convention
ORDER_MIN, ORDER_MAX = 1, 20


def main() -> None:
    if not EXTERNAL_MAT.exists():
        raise SystemExit(f"external data not found at {EXTERNAL_MAT} -- "
                          f"set EXTERNAL_AR_MAT to point at your own .mat file; "
                          f"production code does not depend on it, only this cache file does.")

    d = sio.loadmat(EXTERNAL_MAT, simplify_cells=True)
    segs = d["usable_segments"]
    zero_stim = [s["beta"] for s in segs if abs(s["stim_level"]) < 1e-9]
    print(f"loaded {len(segs)} segments, {len(zero_stim)} at stim_level=0 "
          f"(lengths {[len(s) for s in zero_stim]})")

    min_len = ORDER_MAX + 20
    cleaned = []
    for s in zero_stim:
        seg = R.clean_beta_segments(s, [(0, len(s))], TRIM_SAMPLES, min_len)
        cleaned.extend(seg)
    total_samples = sum(len(c) for c in cleaned)
    print(f"{len(cleaned)} usable segments after trim, {total_samples} samples total "
          f"({total_samples * EXTERNAL_TS:.1f}s)")

    order, phi_50hz, sigma2_50hz = R._select_consensus_ar(cleaned, ORDER_MIN, ORDER_MAX, verbose=True)
    stable_50, eig_50 = R.ar_stability_check(phi_50hz)
    print(f"external AR({order}) @50Hz: stable={stable_50}, sigma2={sigma2_50hz:.4e}")
    assert stable_50, "external AR fit is unstable -- do not cache"

    phi_25hz = R.resample_ar_model(phi_50hz, factor=2)
    stable_25, eig_25 = R.ar_stability_check(phi_25hz)
    print(f"downsampled to 25Hz (exact, matrix-square): stable={stable_25}, "
          f"max|eig|={np.max(np.abs(eig_25)):.4f}")
    assert stable_25, "downsampled AR model is unstable -- do not cache"

    np.savez(CACHE_PATH, order=order, phi_50hz=phi_50hz, sigma2_50hz=sigma2_50hz,
             phi_25hz=phi_25hz, source_ts=EXTERNAL_TS, target_ts=R.CFG.Ts,
             n_segments=len(cleaned), n_samples=total_samples)
    print(f"cached to {CACHE_PATH}")


if __name__ == "__main__":
    main()
