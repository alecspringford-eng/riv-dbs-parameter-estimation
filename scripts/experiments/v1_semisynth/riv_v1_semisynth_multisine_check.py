"""
Entry 36 found a well-conditioned 3-tone multisine (PE4 cond ~1e3, ACF1
0.64) still recovers WORSE than real u (cond 9.4e3-1.2e5, ACF1 0.98-0.997)
in V1's semi-synthetic construction -- i.e. static PE conditioning alone
does not predict V1's online recursive convergence. Follow-up question:
is the multisine's PERIODICITY (a short cycle replayed many times) the
missing variable, separate from its conditioning/autocorrelation numbers?
Adds PRBS2 (2-level, dwell=1, aperiodic pseudo-random) and APRBS
(independent-per-dwell random level, aperiodic) as two more input
conditions -- both are non-periodic like real u, letting periodicity be
varied independently of PE cond and ACF1 (PRBS2 has the best PE
conditioning of anything tested AND near-zero autocorrelation AND is
aperiodic; multisine has good-but-worse conditioning, low autocorrelation,
AND is periodic -- if PRBS converges and multisine doesn't, periodicity,
not conditioning, is implicated).

Same construction as riv_v1_semisynth_experiment.main() with ONE variable
changed: u. All conditions scored on (a) V1 k/tau1/tau2 recovery and (b)
the same pe_order_check used in Entry 30/35/36, measured on the exact
input fed to the estimator, not inferred separately.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import lfilter

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/v1_logparam"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/excitation_design"))
import riv_dual_estimation as R
from riv_v1_semisynth_experiment import moving_block_bootstrap
from riv_v1_logparam_diagnostics import pe_order_check
from riv_excitation_protocol_design_check import make_prbs2, make_aprbs, acf1

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
K_TRUE, TAU1_TRUE, TAU2_TRUE = 62.11, 0.05, 0.25
N_SEEDS = 30  # bumped from 6 (Entry 38 next-step (a): resolve whether the 2/6
# recordings that moved the "wrong" direction under shuffling are real or seed noise
BLOCK_LEN = 15
MEAS_NOISE_STD = 0.02
TONES_HZ = (0.5, 2.0, 6.0)  # log/octave-spaced -- (0.3,1.0,3.0) tested first and
# scored PE4 cond 1.5e4-2.6e4 (worse than this set's ~1e3-3e3, see script docstring
# investigation log): near-harmonic ratios (3.33x, 3x) between closely-linear-spaced
# tones make the Toeplitz matrix ill-conditioned regardless of tone COUNT -- frequency
# spacing, not just how many tones, determines PE conditioning for a multisine.


def make_multisine(n: int, lo: float, hi: float, Ts: float, freqs_hz: tuple[float, ...],
                    rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) * Ts
    x = np.zeros(n)
    for f in freqs_hz:
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x = (x - x.min()) / (x.max() - x.min())
    return lo + x * (hi - lo)


def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal first-order low-pass (rate limiter surrogate):
    y[k] = alpha*x[k] + (1-alpha)*y[k-1]. alpha=1 -> no smoothing,
    alpha->0 -> heavy smoothing (y nearly constant). IIR form via lfilter
    for speed (called many times inside the alpha-calibration bisection)."""
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x)


def make_rate_limited_aprbs(n: int, lo: float, hi: float, min_dwell: int, max_dwell: int,
                             target_diff_std: float, rng: np.random.Generator) -> np.ndarray:
    """Entry 39's flagged combined surrogate: APRBS (aperiodic, Entry 35/37's
    best-performing synthetic input) passed through a rate limiter, alpha
    calibrated so the OUTPUT's increment std matches real u's (0.0005-0.0012)
    -- tests whether increment scale + aperiodicity together close the
    remaining gap to real u (49% vs APRBS's un-smoothed 35%)."""
    raw = make_aprbs(n, 0.0, 1.0, min_dwell, max_dwell, rng)

    def diff_std_at(alpha: float) -> float:
        y = _exp_smooth(raw, alpha)
        y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
        return float(np.diff(lo + y * (hi - lo)).std())

    lo_a, hi_a = 1e-4, 1.0
    for _ in range(40):
        mid = 0.5 * (lo_a + hi_a)
        if diff_std_at(mid) > target_diff_std:
            hi_a = mid
        else:
            lo_a = mid
    alpha = 0.5 * (lo_a + hi_a)
    y = _exp_smooth(raw, alpha)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return lo + y * (hi - lo)


def main() -> None:
    cfg, dual_cfg = R.CFG, R.DUAL_CFG
    recordings = R.load_all_recordings(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        ar_fits = R.fit_ar_beta_model_per_recording(recordings, cfg)
        all_outputs = np.concatenate([y for _, y in recordings.values()])
        epsilon = R.compute_epsilon_floor(all_outputs)
        thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac
        trim_samples = round(cfg.settle_time_s / cfg.Ts)
        min_len = cfg.seg_ar_order_max + 20
        seg_pool = {name: R.no_stim_segments_for_recording(u, y, thresh, trim_samples, min_len,
                                                             cfg.ar_log_domain, epsilon)
                    for name, (u, y) in recordings.items()}

    theta_true = R.params_to_theta(K_TRUE, TAU1_TRUE, TAU2_TRUE, cfg.Ts)

    print(f"{'recording':<20}{'input':<20}{'PE4 cond':>12}{'acf1':>7}{'diff_std':>10}"
          f"{'diff_kurt':>11}{'k_med':>9}{'ok/N':>8}{'tau1_med':>10}{'tau2_med':>10}{'floor_frac':>11}")
    print("-" * 130)

    for name in cfg.system_id_files:
        u_real, _ = recordings[name]
        N = len(u_real)
        fit = ar_fits[name]
        own = seg_pool[name]
        segments = own if own else R._find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)[0]
        lo, hi = float(u_real.min()), float(u_real.max())
        R_heur = 0.05 * fit["sigma2_ar"]
        target_diff_std = float(np.diff(u_real).std())

        for label, u_variant_fn in (
            ("real u", lambda rng: u_real),
            ("shuffled real u (same dist)", lambda rng: rng.permutation(u_real)),
            ("3-tone multisine (periodic)", lambda rng: make_multisine(N, lo, hi, cfg.Ts, TONES_HZ, rng)),
            ("PRBS2 dwell=1 (aperiodic)", lambda rng: make_prbs2(N, lo, hi, 1, rng)),
            ("APRBS dwell 1-5 (aperiodic)", lambda rng: make_aprbs(N, lo, hi, 1, 5, rng)),
            ("rate-limited APRBS (matched)", lambda rng: make_rate_limited_aprbs(
                N, lo, hi, 1, 5, target_diff_std, rng)),
        ):
            k_arr, t1_arr, t2_arr = [], [], []
            pe4_conds, acf1s, diff_stds, diff_kurts = [], [], [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                u = u_variant_fn(rng)
                pe4_conds.append(pe_order_check(u, 4)["cond"])
                acf1s.append(acf1(u))
                du = np.diff(u)
                diff_stds.append(float(du.std()))
                diff_kurts.append(float(stats.kurtosis(du)))
                eta_true = R.simulate_arx_recursion(theta_true, u)
                c0 = float(np.mean(np.concatenate([s for s in segments])))
                beta_bb = moving_block_bootstrap(segments, N, BLOCK_LEN, rng)
                y = c0 + beta_bb - eta_true + rng.normal(0, MEAS_NOISE_STD, N)
                with contextlib.redirect_stdout(io.StringIO()):
                    res = R.run_identification(u, y, fit["phi_coeffs"], fit["sigma2_ar"],
                                                 R_heur, 0.5 * float(np.var(y)), cfg, dual_cfg,
                                                 FIG_DIR / "_t", make_plots=False)
                k_arr.append(res["k"]); t1_arr.append(res["tau1"]); t2_arr.append(res["tau2"])

            k_arr, t1_arr, t2_arr = np.array(k_arr), np.array(t1_arr), np.array(t2_arr)
            err = np.abs(k_arr - K_TRUE) / K_TRUE
            ok = int(np.sum((k_arr > 0) & (err < 0.3)))
            tau_min = -cfg.Ts / np.log(1e-3)
            floor_frac = float(np.mean((np.abs(t1_arr - tau_min) < 1e-4) |
                                        (np.abs(t2_arr - tau_min) < 1e-4)))
            print(f"{name:<20}{label:<20}{np.median(pe4_conds):>12.3e}{np.median(acf1s):>7.2f}"
                  f"{np.median(diff_stds):>10.4f}{np.median(diff_kurts):>11.2f}"
                  f"{np.median(k_arr):>9.2f}{ok:>5}/{N_SEEDS:<3}{np.median(t1_arr):>10.4f}"
                  f"{np.median(t2_arr):>10.4f}{floor_frac:>11.2f}")
        print()


if __name__ == "__main__":
    main()
