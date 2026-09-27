"""riv_sampling_rate_pole_diagnostic.py -- Entry 53's next step: do the
identified discrete-pole locations explain the two-sided fs failure
(collapse below fs~6Hz, degradation above fs~75Hz)?

For each fs, computes the TRUE discrete pole magnitudes (mechanical
consequence of ZOH discretization: exp(-Ts/tau) -- these move toward z=1
as fs rises and toward z=0 as fs falls, independent of any estimation
error) alongside the IDENTIFIED poles' magnitude at the end of estimation
(median/max over seeds, from theta_hat[-1] via the same
z^2-a1*z-a2=0 characteristic equation `theta_to_params` uses). This arm
(instrument_mode=AUX_DECIMATED, M=1, from
riv_instrument_update_rate_experiment.py) has NO pole clamp (Entry 47's
faithful old-prototype replication), so identified poles are free to
drift wherever the RLS update takes them -- their raw trajectory is the
direct evidence for or against the "poles approach the z=1/z=0 boundary"
hypothesis, not masked by a clamp.

Run: python3 scripts/riv_sampling_rate_pole_diagnostic.py
"""
from __future__ import annotations

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
import riv_instrument_update_rate_experiment as rate_mod
from riv_sampling_rate_sweep_experiment import BASE_CFG, ADAPTIVE_ARM, cfg_at_fs

N_SEEDS = 10
FS_VALUES = [3.0, 4.0, 4.743, 5.612, 6.641, 9.297, 13.015, 18.222, 25.510,
             35.714, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0]


def pole_magnitudes(theta: np.ndarray) -> np.ndarray:
    a1, a2 = theta[0], theta[1]
    return np.abs(np.roots([1.0, -a1, -a2]))


def main() -> None:
    print(f"{'fs (Hz)':>8s} {'true |p1|':>10s} {'true |p2|':>10s} "
          f"{'hat |p| median':>16s} {'hat |p| max':>12s} {'frac unstable':>14s}")
    for fs in FS_VALUES:
        cfg = cfg_at_fs(fs)
        true_mags = None
        seed_max_mags = []
        n_unstable = 0
        for seed in range(N_SEEDS):
            scn, result = rate_mod.run_arm_for_seed(ADAPTIVE_ARM, cfg, seed)
            theta_hat = result[1]
            theta_true = scn["theta_true"]
            if true_mags is None:
                true_mags = np.sort(pole_magnitudes(theta_true))
            hat_mags = pole_magnitudes(theta_hat[-1])
            seed_max_mags.append(hat_mags.max())
            if hat_mags.max() >= 1.0:
                n_unstable += 1
        seed_max_mags = np.array(seed_max_mags)
        print(f"{fs:8.3f} {true_mags[0]:10.4f} {true_mags[1]:10.4f} "
              f"{np.median(seed_max_mags):16.4f} {seed_max_mags.max():12.4f} "
              f"{n_unstable / N_SEEDS:14.2f}")


if __name__ == "__main__":
    main()
