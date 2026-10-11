"""Does a smaller P_rls(a1,a2) min eigenvalue go with worse recovery?

On one seed/recording, V1's P_rls(a1,a2) min eigenvalue was 15x smaller under
a bad synthetic u than under real u, matching the "confidently wrong"
diagnostic in riv_dual_estimation.py (a small, collapsed P_rls block that
still lands wrong means recursion bias/attractor, not unresolved
excitation). This replicates it over N_SEEDS seeds and 3 recordings to see
whether the correlation holds or was a coincidence.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R
from riv_v1_semisynth_experiment import moving_block_bootstrap
from riv_v1_semisynth_multisine_check import make_rate_limited_aprbs

RECORDINGS = ("system_id_1.csv", "system_id_4.csv", "system_id_5.csv")
N_SEEDS = 10
K_TRUE, TAU1_TRUE, TAU2_TRUE = 62.11, 0.05, 0.25
BLOCK_LEN = 15
MEAS_NOISE_STD = 0.02


def instrument(u: np.ndarray, y: np.ndarray, fit: dict, cfg, dual_cfg) -> dict:
    R_heur = 0.05 * fit["sigma2_ar"]
    q_eta = 0.5 * float(np.var(y))
    setup = R.build_kf_setup(u, y, fit["phi_coeffs"], fit["sigma2_ar"], q_eta, cfg, dual_cfg)
    u_norm, u_scale, theta0 = setup["u_norm"], setup["u_scale"], setup["theta0"]
    Q_c, x0, P0_kf, P0_rls = setup["Q_diag"], setup["x0"], setup["P0_kf"], setup["P0_rls"]
    phi_c, a1_prior, a2_prior = setup["phi_c"], setup["a1_prior"], setup["a2_prior"]
    n_off, p = setup["n_off"], setup["p"]

    (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, Pkf_checks, _, theta_epoch_hist,
     theta_hat_by_epoch, elapsed, innov_final, S_final) = R.run_dual_estimator(
        u_norm, y, phi_c, Q_c, R_heur, x0, P0_kf, theta0, P0_rls, dual_cfg.lambda_ff,
        R.N_ETA, p, n_off=n_off, gamma_pole=dual_cfg.gamma_pole, a1_prior=a1_prior, a2_prior=a2_prior,
        n_epochs=dual_cfg.n_epochs, n_snapshots=0, save_epochs=set())

    Prls_sym = 0.5 * (Prls_final + Prls_final.T)
    eig_a1a2 = np.linalg.eigvalsh(Prls_sym[:2, :2])
    k_e, t1_e, t2_e = R.theta_to_params_physical(theta_hat[-1], u_scale, cfg.Ts)
    return {"min_eig_a1a2": float(eig_a1a2.min()), "cond_a1a2": float(eig_a1a2.max() / eig_a1a2.min()),
            "k": k_e, "tau1": t1_e, "tau2": t2_e}


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
    pooled_min_eig, pooled_k_err, pooled_label = [], [], []

    print(f"{'recording':<20}{'condition':<28}{'min_eig_med':>13}{'cond_med':>12}"
          f"{'k_err_med':>11}{'ok/N':>8}")
    print("-" * 92)

    for name in RECORDINGS:
        u_real, _ = recordings[name]
        N = len(u_real)
        fit = ar_fits[name]
        own = seg_pool[name]
        segments = own if own else R._find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)[0]
        lo, hi = float(u_real.min()), float(u_real.max())
        target_diff_std = float(np.diff(u_real).std())

        for label, u_variant_fn in (
            ("real u", lambda rng: u_real),
            ("rate-limited APRBS (bad)", lambda rng: make_rate_limited_aprbs(
                N, lo, hi, 1, 5, target_diff_std, rng)),
        ):
            min_eigs, conds, k_errs, oks = [], [], [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                u = u_variant_fn(rng)
                eta_true = R.simulate_arx_recursion(theta_true, u)
                c0 = float(np.mean(np.concatenate([s for s in segments])))
                beta_bb = moving_block_bootstrap(segments, N, BLOCK_LEN, rng)
                y = c0 + beta_bb - eta_true + rng.normal(0, MEAS_NOISE_STD, N)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = instrument(u, y, fit, cfg, dual_cfg)
                min_eigs.append(r["min_eig_a1a2"]); conds.append(r["cond_a1a2"])
                k_err = abs(r["k"] - K_TRUE) / K_TRUE
                k_errs.append(k_err)
                oks.append(int(r["k"] > 0 and k_err < 0.3))
                pooled_min_eig.append(r["min_eig_a1a2"]); pooled_k_err.append(k_err)
                pooled_label.append(label)

            print(f"{name:<20}{label:<28}{np.median(min_eigs):>13.4e}{np.median(conds):>12.3e}"
                  f"{np.median(k_errs):>11.3f}{sum(oks):>5}/{N_SEEDS:<3}")
        print()

    rho, pval = spearmanr(np.log10(pooled_min_eig), np.log10(np.array(pooled_k_err) + 1e-6))
    print(f"Pooled (both conditions, all recordings, N={len(pooled_min_eig)}): "
          f"Spearman rho(log min_eig_a1a2, log k_err) = {rho:.3f} (p={pval:.2e})")
    print("(rho<0: smaller min eigenvalue [more 'collapsed'/confident] associates with LARGER "
          "error -- the 'confidently wrong' pattern from an earlier experiment. rho>0 or ~0 would contradict it.)")


if __name__ == "__main__":
    main()
