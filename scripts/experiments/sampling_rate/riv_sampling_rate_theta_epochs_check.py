"""Per-fs parameter trajectories across epochs for the 3-tone signal.

As riv_real_recon_theta_epochs_check.py, but for the current 3-tone
rate-limited multisine (the "adaptive" arm of
riv_sampling_rate_sweep_experiment.py), for side-by-side comparison.

Reuses run_arm_detailed from riv_instrument_update_rate_experiment.py (seed=0
detailed run, saves theta_epochs_stacked.png and 5 other diagnostic plots),
redirected per fs through rate_mod.OUT_DIR as the sweep script does, plus
ADAPTIVE_ARM / cfg_at_fs / COARSE_RATIO from the sweep script.

FS_POINTS reproduces the sweep's fs grid (descent 50, 35.714, 25.510, 18.222,
13.015, 9.297, 6.641, 4.743, then bisection 5.612, 6.105) by the same
arithmetic instead of re-running the 10-seed evaluation. The branches taken
are known from the sweep's logged ok-rates: 6.641Hz 5/10 passes (>=0.5) so
descent continues to 4.743Hz 0/10 (fails); midpoint 5.612Hz 4/10 fails
(fs_fail moves); midpoint 6.105Hz 5/10 passes (fs_pass moves); the bracket is
then within the 10% tolerance. Each point needs one cheap seed=0 run.

Run: python3 scripts/experiments/sampling_rate/riv_sampling_rate_theta_epochs_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
import riv_instrument_update_rate_experiment as rate_mod
from riv_sampling_rate_sweep_experiment import ADAPTIVE_ARM, cfg_at_fs, COARSE_RATIO

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "sampling_rate_theta_epochs"

# Reproduces the sweep's fs grid (see docstring)
_descent = [50.0 * COARSE_RATIO ** i for i in range(8)]
_fs_pass, _fs_fail = _descent[6], _descent[7]  # 6.641 (last >=0.5), 4.743 (<0.5)
_mid1 = float(np.sqrt(_fs_pass * _fs_fail))     # 5.612, fails
_fs_fail = _mid1
_mid2 = float(np.sqrt(_fs_pass * _fs_fail))     # 6.105, passes
_bisection = [_mid1, _mid2]

# Ascent above the 50Hz anchor, with the same fs values (70, 98) as the real-recon
# sweep so both signals are compared at identical sampling rates
_ascent = [50.0 / COARSE_RATIO, 50.0 / COARSE_RATIO ** 2]

FS_POINTS = sorted(_descent + _bisection + _ascent)


def main() -> None:
    print(f"fs grid ({len(FS_POINTS)} points, reproduced from an earlier experiment): "
          f"{[round(f, 3) for f in FS_POINTS]}\n")
    for fs in FS_POINTS:
        fs_tag = f"fs{fs:07.3f}".replace(".", "p")
        rate_mod.OUT_DIR = OUT_DIR / fs_tag
        rate_mod.run_arm_detailed(ADAPTIVE_ARM, cfg_at_fs(fs), seed=0)


if __name__ == "__main__":
    main()
