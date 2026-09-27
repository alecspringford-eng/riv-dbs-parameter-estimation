"""riv_sampling_rate_theta_epochs_check.py -- same idea as
riv_real_recon_theta_epochs_check.py, but for the CURRENT 3-tone
rate-limited multisine signal (Entry 50 / riv_sampling_rate_sweep_
experiment.py's "adaptive" arm sweep), so the two can be compared directly
side by side.

Reuses run_arm_detailed UNMODIFIED from riv_instrument_update_rate_
experiment.py -- it already does exactly this (seed=0 detailed run,
saves theta_epochs_stacked.png among 5 other diagnostic plots), redirected
per-fs via the same rate_mod.OUT_DIR reassignment trick
riv_sampling_rate_sweep_experiment.py itself already uses (see its
run_detailed block). Also reuses ADAPTIVE_ARM/cfg_at_fs/COARSE_RATIO from
riv_sampling_rate_sweep_experiment.py unmodified.

FS_POINTS reproduces Entry 50's documented fs grid (INVESTIGATION.md,
10 points: descent 50,35.714,25.510,18.222,13.015,9.297,6.641,4.743 then
bisection trace 5.612,6.105) by the same closed-form arithmetic used in
riv_real_recon_theta_epochs_check.py -- not by re-running the 10-seed
ok-rate evaluation, since the branch each bisection step took is already
known from Entry 50's own logged ok-rate values (6.641Hz: 5/10=0.5 passes
the >=0.5 bar so descent continues one more step to 4.743Hz: 0/10 fails;
bisection midpoint 5.612Hz: 4/10 fails -> fs_fail moves; next midpoint
6.105Hz: 5/10 passes -> fs_pass moves; bracket then within the 10%
tolerance). Each point here only needs one cheap seed=0 detailed run.

Run: python3 scripts/riv_sampling_rate_theta_epochs_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root, for riv_* module imports

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
import riv_instrument_update_rate_experiment as rate_mod
from riv_sampling_rate_sweep_experiment import ADAPTIVE_ARM, cfg_at_fs, COARSE_RATIO

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "sampling_rate_theta_epochs"

# Reproduces Entry 50's documented fs grid (see docstring for why this is
# exact without re-running the 10-seed eval).
_descent = [50.0 * COARSE_RATIO ** i for i in range(8)]  # 50 ... 4.743 (0/10 fails)
_fs_pass, _fs_fail = _descent[6], _descent[7]  # 6.641 (last >=0.5), 4.743 (<0.5)
_mid1 = float(np.sqrt(_fs_pass * _fs_fail))     # 5.612, fails -> fs_fail moves
_fs_fail = _mid1
_mid2 = float(np.sqrt(_fs_pass * _fs_fail))     # 6.105, passes -> fs_pass moves
_bisection = [_mid1, _mid2]

# Ascent above the 50Hz anchor, same COARSE_RATIO stepping used for the
# real-recon sweep's ascent (riv_real_recon_multisine_sweep_check.py) --
# matched fs values (70, 98) so the two signals are compared at identical
# sampling rates, not just identical relative steps.
_ascent = [50.0 / COARSE_RATIO, 50.0 / COARSE_RATIO ** 2]  # 70.0, 98.0

FS_POINTS = sorted(_descent + _bisection + _ascent)


def main() -> None:
    print(f"fs grid ({len(FS_POINTS)} points, reproduced from Entry 50): "
          f"{[round(f, 3) for f in FS_POINTS]}\n")
    for fs in FS_POINTS:
        fs_tag = f"fs{fs:07.3f}".replace(".", "p")
        rate_mod.OUT_DIR = OUT_DIR / fs_tag
        rate_mod.run_arm_detailed(ADAPTIVE_ARM, cfg_at_fs(fs), seed=0)


if __name__ == "__main__":
    main()
