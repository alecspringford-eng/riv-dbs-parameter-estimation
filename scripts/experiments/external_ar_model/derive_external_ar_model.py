"""Derive an AR disturbance-model order/shape from a richer external dataset.

Our own no-stim data is too thin for a whiteness test: 3/6 system_id
recordings have no usable no-stim segment and the rest have one 125-sample
segment, so every order from 3 to 15 trivially passes (frac_exceed=0.000).

The external recording (nndbsProject, 50 Hz, a different session/subject)
has 4 zero-stim segments of 5000-11000 samples. The AR order/shape is fit on
it, downsampled exactly to 25 Hz with resample_ar_model, and cached to
external_ar_model.npz so production code never needs the .mat file.

This is a cross-dataset structural probe, not a claim that the two
recordings share physiology. Re-run only if the source .mat file changes.
"""
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R

# External .mat file (not included in this repo): set EXTERNAL_AR_MAT or edit the fallback
EXTERNAL_MAT = Path(os.environ.get("EXTERNAL_AR_MAT", "external_data/dbs_segments.mat"))
CACHE_PATH = Path(__file__).parent / "external_ar_model.npz"
EXTERNAL_TS = 0.02  # 50 Hz
TRIM_SAMPLES = 250  # 5s @ 50Hz, the source study's trim
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
    stable_50, _ = R.ar_stability_check(phi_50hz)
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
