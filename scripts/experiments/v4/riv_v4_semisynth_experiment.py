"""
Experiment: Entry 27's flagged follow-up. Does V4's real-data tau
collapse (INVESTIGATION.md Entries 26/27) appear even when the generative
process is EXACTLY V4's own model (no misspecification), driven by real u
and real (bootstrapped) beta content -- or is it specific to real data's
own residual misfit (plant misspecification, e.g. Entry 14's still-open
missing periodic/oscillatory structure during stim)?

Construction: known truth (k,tau1,tau2,c0) EXACTLY as assumed by V4,
g_true=exp(-eta_true) from the same verified cascade, beta content from a
moving-block bootstrap of that recording's own real no-stim segments
(riv_v1_semisynth_experiment's construction, reused -- rules out beta's
own shape as a confound too, matching Entries 15/18's convention), driven
by that recording's own real u. y = g_true*(c0+beta_bootstrapped) + noise.
Fit V4, check whether tau1/tau2 recover near truth (as Entry 26's
synthetic-AR-beta self-check showed under real u) or still collapse to
near-zero the way real data does.

If tau recovers here: the real-data collapse is due to MISSPECIFICATION
(real y doesn't actually follow V4's multiplicative model), not an
intrinsic property of the architecture on real u. If tau STILL collapses
here (correctly-specified process, real u, real beta): something about
real u itself, independent of misspecification, drives the collapse in
this architecture.

main() untouched -- report only. See INVESTIGATION.md Entry 28.
"""
import contextlib
import io
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v2"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v4"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/v1_semisynth"))
import riv_dual_estimation as R
import riv_v2_greybox as V2
import riv_v4_coupled_suppression as V4
from riv_v1_semisynth_experiment import moving_block_bootstrap

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
K_TRUE, TAU1_TRUE, TAU2_TRUE, C0_TRUE, R_TRUE = 62.11, 0.05, 0.25, 1.10, 0.01
N_SEEDS = 6
BLOCK_LEN = 15
INFORMATIVE_RECORDINGS = ("system_id_3.csv", "system_id_4.csv",
                           "system_id_5.csv", "system_id_long_run.csv")


def main() -> None:
    cfg = R.CFG
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
        _, phi_ar_pooled, _ = V4._pooled_ar()

    rows = []
    for name in INFORMATIVE_RECORDINGS:
        u, _ = recordings[name]
        N = len(u)
        own = seg_pool[name]
        segments = own if own else R._find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)[0]

        eta_true = V2._eta_cascade(K_TRUE, (TAU1_TRUE, TAU2_TRUE), u)
        g_true = np.exp(-eta_true)

        k_vals, t1_vals, t2_vals = [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            beta_bb = moving_block_bootstrap(segments, N, BLOCK_LEN, rng)
            y = g_true * (C0_TRUE + beta_bb) + rng.normal(0, np.sqrt(R_TRUE), N)
            with contextlib.redirect_stdout(io.StringIO()):
                f = V4.fit_v4(u, y, phi_ar_pooled)
            k_vals.append(f.k); t1_vals.append(f.tau1); t2_vals.append(f.tau2)

        k_vals, t1_vals, t2_vals = np.array(k_vals), np.array(t1_vals), np.array(t2_vals)
        print(f"{name}: k median={np.median(k_vals):.2f} (truth {K_TRUE}), "
              f"tau1 median={np.median(t1_vals):.4f} (truth {TAU1_TRUE}), "
              f"tau2 median={np.median(t2_vals):.4f} (truth {TAU2_TRUE})")
        print(f"  tau1 range=[{t1_vals.min():.4f},{t1_vals.max():.4f}], "
              f"tau2 range=[{t2_vals.min():.4f},{t2_vals.max():.4f}]")
        rows.append({"name": name, "k_med": float(np.median(k_vals)),
                     "t1_med": float(np.median(t1_vals)), "t2_med": float(np.median(t2_vals))})

    print("\n=== Comparison: semi-synthetic (correctly-specified) vs real-data V4 tau ===")
    print("(real-data numbers from Entry 26/27's pure-MLE run, window=1/unsmoothed)")
    real_tau = {"system_id_3.csv": (0.0020, 0.0026), "system_id_4.csv": (0.0157, 0.0157),
                "system_id_5.csv": (0.0017, 0.0042), "system_id_long_run.csv": (0.0020, 0.0029)}
    for r in rows:
        rt1, rt2 = real_tau[r["name"]]
        print(f"{r['name']}: semi-synthetic tau=({r['t1_med']:.4f},{r['t2_med']:.4f})  "
              f"vs  real-data tau=({rt1:.4f},{rt2:.4f})  vs  truth=({TAU1_TRUE},{TAU2_TRUE})")


if __name__ == "__main__":
    main()
