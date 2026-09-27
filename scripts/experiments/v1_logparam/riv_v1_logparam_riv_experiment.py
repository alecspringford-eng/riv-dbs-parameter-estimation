"""
Method A for V1 root-cause #2 (`k=(b1+b2)/(1-a1-a2)` near-zero-denominator
blowup, RESEARCH_NOTEBOOK.md root-cause rank 2 / PHASE4_ROOT_CAUSE_ANALYSIS.md).

Phase 6/8 only clamped or regularised the raw theta=[a1,a2,b1,b2] AFTER each
RLS/IV update -- the division is still computed every step, just with its
inputs bent back into range. Proven capped (PHASE8_V1_FURTHER_TUNING.md):
every knob traded real-data sign for synthetic accuracy, none fixed both.

This script removes the division itself. The recursive IV update runs
directly on logp=[log k, log tau1, log tau2] via a recursive Gauss-Newton /
extended-IV step (pseudo-linear regression generalised to a nonlinear
reparametrisation -- Ljung, "System Identification: Theory for the User",
ch.11; Soderstrom & Stoica's RIV gain structure carried over unchanged,
just projected through the Jacobian d(theta)/d(logp)). k=exp(logp[0]) can
structurally never come from a near-zero denominator again, because
`(1-a1-a2)` is never computed inside the recursion at all.

Everything else -- structured KF over eta/beta/offset, epoch loop,
noise-free instrument z, data loading, AR disturbance fit, avg-patient
init -- is reused unmodified from riv_dual_estimation.py, so this isolates
the theta-parametrisation vs logp-parametrisation choice as the only
variable under test.

Cost: params_to_theta calls the `control` package (matrix exponential via
python objects), not numba-jittable -- this core is pure-python, timed at
~0.2ms/call, 6 calls/sample (finite-difference Jacobian) -> ~1-2ms/sample.
Fine for this offline comparison (n_epochs reduced from 30 to 10-15, Phase 6
showed convergence already plateaus by epoch ~10). NOT a production
replacement without an analytic or numba-compiled Jacobian.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import (
    CFG, DUAL_CFG, N_ETA, POLE_MAX, AVG_PATIENT_PARAMS,
    load_all_recordings, fit_ar_beta_model_per_recording,
    params_to_theta, build_kf_setup, residual_whiteness_test,
    nis_consistency_check, simulate_arx_recursion, ar_to_state_space,
    plot_input, plot_output_fit, plot_residual, plot_components,
    plot_input_excitation, plot_k_tau_vs_epoch, plot_final_params_bar,
)

POLE_MIN = 1e-3  # matches _clamp_pole_pair's hardcoded floor in riv_dual_estimation.py
LOGK_BOUND = 10.0  # k in [4.5e-5, 2.2e4] -- runaway breaker only, mirrors B_MAX's role


def _tau_bounds(Ts: float) -> tuple[float, float]:
    # smaller pole -> faster decay -> smaller tau; larger pole -> slower decay -> larger tau
    tau_min = -Ts / np.log(POLE_MIN)
    tau_max = -Ts / np.log(POLE_MAX)
    return tau_min, tau_max


def theta_from_logp(logp: np.ndarray, Ts: float, u_scale: float) -> np.ndarray:
    k, tau1, tau2 = np.exp(logp)
    theta = params_to_theta(k, tau1, tau2, Ts)
    theta[2] *= u_scale
    theta[3] *= u_scale
    return theta


def _theta_jacobian(logp: np.ndarray, Ts: float, u_scale: float, eps: float) -> np.ndarray:
    """Central-difference d(theta)/d(logp), shape (4,3)."""
    J = np.empty((4, 3))
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = eps
        J[:, i] = (theta_from_logp(logp + dp, Ts, u_scale)
                   - theta_from_logp(logp - dp, Ts, u_scale)) / (2 * eps)
    return J


def _clamp_logp(logp: np.ndarray, Ts: float) -> np.ndarray:
    tau_min, tau_max = _tau_bounds(Ts)
    out = logp.copy()
    out[0] = np.clip(out[0], -LOGK_BOUND, LOGK_BOUND)
    out[1] = np.clip(out[1], np.log(tau_min), np.log(tau_max))
    out[2] = np.clip(out[2], np.log(tau_min), np.log(tau_max))
    return out


def _dual_estimator_core_logparam(u, y, phi_ar, Q_diag, R, x0, P0_kf, logp0, P0_rls,
                                   lam, n_eta, p, n_off, Ts, u_scale, jac_eps):
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = len(y)

    x = x0.copy()
    Pkf = P0_kf.copy()
    logp = logp0.copy()
    Prls = P0_rls.copy()
    z1 = z2 = 0.0

    x_hat = np.zeros((N, n))
    logp_hat = np.zeros((N, 3))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    theta = theta_from_logp(logp, Ts, u_scale)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a1, a2, b1, b2 = theta
            eta1, eta2 = x[0], x[1]
            x_new = np.empty(n)
            x_new[0] = a1 * eta1 + eta2 + b1 * u_prev
            x_new[1] = a2 * eta1 + b2 * u_prev
            x_new[idx_beta] = phi_ar @ x[idx_beta:idx_beta + p]
            if p > 1:
                x_new[idx_beta + 1:idx_beta + p] = x[idx_beta:idx_beta + p - 1]
            if n_off == 1:
                x_new[idx_off] = x[idx_off]
            x = x_new

            A = np.zeros((n, n))
            A[0, 0], A[0, 1] = a1, 1.0
            A[1, 0] = a2
            A[idx_beta, idx_beta:idx_beta + p] = phi_ar
            if p > 1:
                A[idx_beta + 1:idx_beta + p, idx_beta:idx_beta + p - 1] = np.eye(p - 1)
            if n_off == 1:
                A[idx_off, idx_off] = 1.0
            Pkf = A @ Pkf @ A.T
            Pkf += np.diag(Q_diag)
            Pkf = 0.5 * (Pkf + Pkf.T)

        off = x[idx_off] if n_off == 1 else 0.0
        h_pred = x[idx_beta] - x[0] + off
        innov = y[kk] - h_pred

        H = np.zeros(n)
        H[0] = -1.0
        H[idx_beta] = 1.0
        if n_off == 1:
            H[idx_off] = 1.0
        HP = Pkf @ H
        S = float(H @ HP + R)
        innov_hist[kk] = innov
        S_hist[kk] = S
        K = HP / S
        x = x + K * innov
        Pkf = Pkf - np.outer(K, HP)
        Pkf = 0.5 * (Pkf + Pkf.T)

        x_hat[kk] = x
        y_hat[kk] = x[idx_beta] - x[0] + (x[idx_off] if n_off == 1 else 0.0)

        eta_hat_k = x[0]
        eta_km1 = x_hat[kk - 1, 0] if kk - 1 >= 0 else 0.0
        eta_km2 = x_hat[kk - 2, 0] if kk - 2 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        u_km2 = u[kk - 2] if kk - 2 >= 0 else 0.0
        phi_vec = np.array([eta_km1, eta_km2, u_km1, u_km2])
        z_vec = np.array([z1, z2, u_km1, u_km2])

        # sensitivity of the ARX prediction w.r.t. logp, via chain rule through
        # the (validated) theta(k,tau1,tau2) map -- this replaces the raw
        # phi_vec/z_vec RIV gain with its projection into logp-space.
        J = _theta_jacobian(logp, Ts, u_scale, jac_eps)
        psi_phi = J.T @ phi_vec
        psi_z = J.T @ z_vec

        pred = phi_vec @ theta
        err = eta_hat_k - pred

        # extend the noise-free instrument using pre-update theta, same as
        # riv_dual_estimation._dual_estimator_core
        z_new = theta[0] * z1 + theta[1] * z2 + theta[2] * u_km1 + theta[3] * u_km2
        z2 = z1
        z1 = z_new

        Pz = Prls @ psi_z
        denom = lam + psi_phi @ Pz
        Krls = Pz / denom
        logp = logp + Krls * err
        logp = _clamp_logp(logp, Ts)
        Prls = (Prls - np.outer(Krls, psi_phi @ Prls)) / lam
        Prls = 0.5 * (Prls + Prls.T)

        theta = theta_from_logp(logp, Ts, u_scale)
        logp_hat[kk] = logp

    return x_hat, logp_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def run_dual_estimator_logparam(u, y, phi_ar, Q_diag, R, x0, P0_kf, logp0, P0_rls,
                                 lam, n_eta, p, n_off, Ts, u_scale, n_epochs, jac_eps=1e-4):
    logp_cur = logp0.copy()
    logp_epoch_hist = np.zeros((n_epochs, 3))
    x_hat = logp_hat = y_hat = Pkf_final = Prls_final = innov_final = S_final = None
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        x0_epoch = x0.copy()
        (x_hat, logp_hat, y_hat, Pkf_final, Prls_final,
         innov_final, S_final) = _dual_estimator_core_logparam(
            u, y, phi_ar, Q_diag, R, x0_epoch, P0_kf.copy(), logp_cur, P0_rls.copy(),
            lam, n_eta, p, n_off, Ts, u_scale, jac_eps)
        logp_cur = logp_hat[-1].copy()
        logp_epoch_hist[epoch] = logp_cur
    elapsed = time.perf_counter() - t0
    return x_hat, logp_hat, y_hat, Pkf_final, Prls_final, logp_epoch_hist, elapsed, innov_final, S_final


def run_identification_logparam(u, y, phi_coeffs, sigma2_ar, R, q_eta, cfg, dual_cfg,
                                 n_epochs=None, jac_eps=1e-4) -> dict:
    setup = build_kf_setup(u, y, phi_coeffs, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale = setup["u_norm"], setup["u_scale"]
    Q_c, x0, P0_kf = setup["Q_diag"], setup["x0"], setup["P0_kf"]
    phi_c, n_off, p = setup["phi_c"], setup["n_off"], setup["p"]

    logp0 = np.log(np.array([AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"],
                              AVG_PATIENT_PARAMS["tau2"]]))
    P0_rls = dual_cfg.p0_rls * np.eye(3)
    n_epochs = dual_cfg.n_epochs if n_epochs is None else n_epochs

    (x_hat, logp_hat, y_hat, Pkf_final, Prls_final, logp_epoch_hist, elapsed,
     innov_final, S_final) = run_dual_estimator_logparam(
        u_norm, y, phi_c, Q_c, R, x0, P0_kf, logp0, P0_rls, dual_cfg.lambda_ff,
        N_ETA, p, n_off, cfg.Ts, u_scale, n_epochs, jac_eps)

    logp_final = logp_hat[-1]
    k_f = float(np.exp(logp_final[0]))
    tau1_f = float(np.exp(logp_final[1]))
    tau2_f = float(np.exp(logp_final[2]))

    passed_white, frac_white = residual_whiteness_test(innov_final)
    _, nis_mean, nis_ok = nis_consistency_check(innov_final, S_final)

    epoch_k = np.exp(logp_epoch_hist[:, 0])
    epoch_tau1 = np.exp(logp_epoch_hist[:, 1])
    epoch_tau2 = np.exp(logp_epoch_hist[:, 2])

    return {"k": k_f, "tau1": tau1_f, "tau2": tau2_f, "elapsed": elapsed,
            "whiteness_pass": passed_white, "whiteness_frac_exceed": frac_white,
            "nis_mean": nis_mean, "nis_consistent": nis_ok,
            "epoch_k": epoch_k, "epoch_tau1": epoch_tau1, "epoch_tau2": epoch_tau2,
            "x_hat": x_hat, "logp_hat": logp_hat, "y_hat": y_hat,
            "innov": innov_final, "S": S_final, "u_norm": u_norm, "u_scale": u_scale,
            "n_off": n_off, "p": p}


# ==========================================================================
# Synthetic self-check: identical generator to
# riv_dual_estimation.test_v1_recovers_synthetic_plant (S4.8), so the
# positive-k / within-30% recovery rate is directly comparable to that
# test's documented baseline (>=1/6 asserted; broader 30-seed sweep ~23%,
# see PHASE6_TUNING_REPORT.md).
# ==========================================================================

def synthetic_selfcheck(cfg, dual_cfg, n_epochs: int = 10, jac_eps: float = 1e-4) -> list[dict]:
    k_true, tau1_true, tau2_true, c0_true = 80.0, 0.05, 0.25, 1.10
    phi_ar_true = np.array([0.6, -0.2, 0.1])
    sigma2_true = 0.05
    N = 3000
    seeds = range(6)
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.004, 0.026, size=N)
        eta_true = simulate_arx_recursion(params_to_theta(k_true, tau1_true, tau2_true, cfg.Ts), u)
        A_beta, C_beta = ar_to_state_space(phi_ar_true)
        p = len(phi_ar_true)
        xb = np.zeros(p)
        beta_true = np.zeros(N)
        for kk in range(N):
            beta_true[kk] = (C_beta @ xb)[0]
            xb = A_beta @ xb
            xb[0] += rng.normal(0, np.sqrt(sigma2_true))
        y = c0_true + beta_true - eta_true + rng.normal(0, 0.05, N)

        R_heur = 0.05 * sigma2_true
        q_eta_heur = 0.5 * float(np.var(y))
        t0 = time.perf_counter()
        res = run_identification_logparam(u, y, phi_ar_true, sigma2_true, R_heur, q_eta_heur,
                                           cfg, dual_cfg, n_epochs=n_epochs, jac_eps=jac_eps)
        dt = time.perf_counter() - t0
        k_err = abs(res["k"] - k_true) / k_true
        ok = res["k"] > 0 and k_err < 0.3
        results.append({"seed": seed, "k": res["k"], "tau1": res["tau1"], "tau2": res["tau2"],
                         "k_err_pct": 100 * k_err, "ok": ok,
                         "whiteness_frac_exceed": res["whiteness_frac_exceed"],
                         "nis_mean": res["nis_mean"]})
        print(f"seed={seed}: k={res['k']:.2f} (true 80, err {100*k_err:.1f}%) "
              f"tau1={res['tau1']:.4f} tau2={res['tau2']:.4f} ok={ok} "
              f"whiteness_exceed={res['whiteness_frac_exceed']:.2f} "
              f"nis_mean={res['nis_mean']:.2f} [{dt:.1f}s]")
    n_ok = sum(r["ok"] for r in results)
    n_pos = sum(r["k"] > 0 for r in results)
    print(f"\nlogparam self-check: {n_pos}/{len(seeds)} positive-k, "
          f"{n_ok}/{len(seeds)} within 30% of k_true=80 with correct sign "
          f"(V1-original baseline on this exact generator: >=1/6 asserted floor; "
          f"documented broader 30-seed sweep 7/30~23%, PHASE6_TUNING_REPORT.md)")
    return results


def plot_kt_trajectory(t: np.ndarray, logp_hat: np.ndarray, path: Path) -> None:
    """logp-parametrisation analog of riv_dual_estimation.plot_theta_trajectory
    -- (k, tau1, tau2) directly, not the raw ARX coefficients, since that's
    the whole point of this reparametrisation (no post-hoc inversion)."""
    kt = np.exp(logp_hat)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, kt[:, 0], lw=1, color="tab:blue")
    axes[0].set_ylabel("k"); axes[0].set_title("k, tau1, tau2 trajectory (final epoch, logp-parametrised)")
    axes[1].plot(t, kt[:, 1], lw=1, color="tab:green")
    axes[1].set_ylabel("tau1 (s)")
    axes[2].plot(t, kt[:, 2], lw=1, color="tab:red")
    axes[2].set_ylabel("tau2 (s)"); axes[2].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ==========================================================================
# Real-data run: same 6 system_id recordings, same AR disturbance fit,
# same R/q_eta heuristic as riv_dual_estimation.main() -- only the
# theta-vs-logp parametrisation differs. n_epochs reduced from the
# production default (30) to 10 for wall-clock budget; Phase 6 found
# k/tau convergence already plateaus by epoch ~10 (low n_epochs sensitivity).
# ==========================================================================

def real_data_run(cfg, dual_cfg, n_epochs: int = 30, jac_eps: float = 1e-4,
                   make_plots: bool = True,
                   out_dir: Path = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_logparam") -> dict:
    recordings = load_all_recordings(cfg)
    ar_fits = fit_ar_beta_model_per_recording(recordings, cfg)

    results = {}
    for name in cfg.system_id_files:
        key = name.replace(".csv", "")
        u, y = recordings[name]
        f = ar_fits[name]
        R_heur = 0.05 * f["sigma2_ar"]
        q_eta_heur = 0.5 * float(np.var(y))
        t0 = time.perf_counter()
        res = run_identification_logparam(u, y, f["phi_coeffs"], f["sigma2_ar"], R_heur, q_eta_heur,
                                           cfg, dual_cfg, n_epochs=n_epochs, jac_eps=jac_eps)
        dt = time.perf_counter() - t0
        results[key] = res
        print(f"{key}: k={res['k']:.2f} tau1={res['tau1']:.4f} tau2={res['tau2']:.4f} "
              f"whiteness_exceed={res['whiteness_frac_exceed']:.2f} "
              f"nis_mean={res['nis_mean']:.2f} [{dt:.1f}s, N={len(u)}]")

        if make_plots:
            rec_dir = out_dir / key
            rec_dir.mkdir(parents=True, exist_ok=True)
            t = np.arange(len(u)) * cfg.Ts
            plot_input(t, u, rec_dir / "input.png",
                       title=f"Real stimulation input ({key})")
            plot_input_excitation(u, cfg.Ts, res["tau1"], res["tau2"],
                                   AVG_PATIENT_PARAMS["tau1"], AVG_PATIENT_PARAMS["tau2"],
                                   rec_dir / "excitation_diagnostic.png")
            plot_output_fit(t, y, res["y_hat"], rec_dir / "output_fit.png",
                             y_hat_label="y_hat (V1 logp-parametrised)")
            plot_residual(t, res["innov"], rec_dir / "residual.png",
                           ylabel="innovation", title="KF innovation vs time")
            plot_components(t, res["x_hat"][:, 0], res["x_hat"][:, N_ETA],
                             rec_dir / "components.png")
            plot_kt_trajectory(t, res["logp_hat"], rec_dir / "kt_trajectory.png")
            plot_k_tau_vs_epoch(np.arange(len(res["epoch_k"])), list(res["epoch_k"]),
                                 list(res["epoch_tau1"]), list(res["epoch_tau2"]),
                                 rec_dir / "k_tau_vs_epoch.png")

    ks = [r["k"] for r in results.values()]
    n_pos = sum(k > 0 for k in ks)
    print(f"\npositive-k: {n_pos}/{len(ks)}")
    print("comparison (from PHASE7_BEST_METHOD.md):")
    print("  V1 orig : [-39.5, -40.9, -687.6, -25.3, -23.0, -18.3]  0/6 positive")
    print("  V1 tuned: [ 65.5,  43.4,   -4.6,  19.8,   3.0,  14.1]  5/6 positive")
    print("  V2 (OE) : [ 44.3,  45.0,   29.3,  44.3,  47.4,  40.5]  6/6 positive")
    print("  V3 (PEM): [ 71.3,  48.8,   29.3,  44.3,  47.4,  39.9]  6/6 positive, NRMSE 0.33")

    if make_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_final_params_bar(results, out_dir / "final_params_bar.png", include_reference=True)

    return results


if __name__ == "__main__":
    print("=== synthetic self-check (logp-parametrised recursive IV) ===")
    synthetic_selfcheck(CFG, DUAL_CFG)
    print("\n=== real data (6 system_id recordings) ===")
    real_data_run(CFG, DUAL_CFG)
