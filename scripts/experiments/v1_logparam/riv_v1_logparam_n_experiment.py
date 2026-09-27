"""
General n-lag-plant generalisation of riv_v1_logparam_riv_experiment.py's
log-space recursive IV core (n_eta=2 there, hardcoded). Built to test
Entry 14's other still-open whiteness candidate ("plant order too low")
directly -- nobody has actually tried a 3-lag cascade before, only the
disturbance (beta) AR order has ever been swept.

PE-order caution (user-requested, riv_v1_logparam_diagnostics.pe_report):
real u's Toeplitz-autocovariance condition number at order 4 (current
n_eta=2, na+nb=4) is already 9.4e3-1.2e5 -- 2-4 orders of magnitude worse
than the synthetic IID-uniform u used in the self-check (cond ~25-39,
where the method works cleanly). Order 6 (n_eta=3) is uniformly ~1.7-2x
worse still on every recording. Ljung's sufficient condition for ARX(na,nb)
identifiability is PE order >= na+nb -- real u does not clear this even at
the CURRENT order, let alone a proposed larger one. Real-data n_eta=3
numbers below are reported anyway (transparency) but should NOT be trusted
as a physical finding -- they test whether the generalised code is
correct, not whether the physical model is validated. Synthetic self-check
(genuinely PE-rich IID input) is the only trustworthy validation of the
n_eta=3 mechanism itself.
"""

from __future__ import annotations

import time
from pathlib import Path

import control
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import (
    CFG, DUAL_CFG, N_ETA as N_ETA_2, POLE_MAX, AVG_PATIENT_PARAMS,
    load_all_recordings, fit_ar_beta_model_per_recording,
    params_to_theta, build_kf_setup, residual_whiteness_test,
    nis_consistency_check, simulate_arx_recursion, ar_to_state_space,
)
from riv_v1_logparam_diagnostics import pe_order_check

POLE_MIN = 1e-3


# ==========================================================================
# General n-lag plant math (n=2 must reduce EXACTLY to riv_dual_estimation's
# validated build_true_ss/params_to_theta -- checked by _selfcheck_n2 below)
# ==========================================================================

def build_true_ss_n(k: float, taus: np.ndarray) -> control.StateSpace:
    n = len(taus)
    Ac = np.zeros((n, n))
    Bc = np.zeros((n, 1))
    Cc = np.zeros((1, n))
    Ac[0, 0] = -1.0 / taus[0]
    Bc[0, 0] = k / taus[0]
    for i in range(1, n):
        Ac[i, i] = -1.0 / taus[i]
        Ac[i, i - 1] = 1.0 / taus[i]
    Cc[0, -1] = 1.0
    return control.ss(Ac, Bc, Cc, np.zeros((1, 1)))


def arx_theta_from_discrete_n(sys_d: control.StateSpace, n: int) -> np.ndarray:
    tf_d = control.tf(sys_d)
    num = np.atleast_1d(np.squeeze(tf_d.num[0][0]))
    den = np.atleast_1d(np.squeeze(tf_d.den[0][0]))
    den = den / den[0]
    num = np.pad(num, (n + 1 - len(num), 0))
    den = np.pad(den, (n + 1 - len(den), 0), constant_values=0.0)
    den[0] = 1.0
    a = -den[1:]
    b = num[1:]
    return np.concatenate([a, b])  # theta = [a1..an, b1..bn]


def params_to_theta_n(k: float, taus: np.ndarray, Ts: float) -> np.ndarray:
    sys_c = build_true_ss_n(k, taus)
    sys_d = control.c2d(sys_c, Ts, method="zoh")
    return arx_theta_from_discrete_n(sys_d, len(taus))


def _selfcheck_n2(Ts: float) -> None:
    """params_to_theta_n(k,[tau1,tau2],Ts) must exactly match the
    already-validated 2-state params_to_theta -- regression guard before
    trusting the n=3 case built the same way."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = rng.uniform(10, 200)
        tau1, tau2 = sorted(rng.uniform(0.01, 0.5, size=2))
        theta_ref = params_to_theta(k, tau1, tau2, Ts)
        theta_n = params_to_theta_n(k, np.array([tau1, tau2]), Ts)
        assert np.allclose(theta_ref, theta_n, atol=1e-9), \
            f"n-generalised params_to_theta_n disagrees with validated params_to_theta: {theta_ref} vs {theta_n}"
    print("[selfcheck] params_to_theta_n(n=2) matches validated params_to_theta on 20 random draws")


def _tau_bounds(Ts: float) -> tuple[float, float]:
    tau_min = -Ts / np.log(POLE_MIN)
    tau_max = -Ts / np.log(POLE_MAX)
    return tau_min, tau_max


def theta_from_logp_n(logp: np.ndarray, Ts: float, u_scale: float) -> np.ndarray:
    n = len(logp) - 1
    k = np.exp(logp[0])
    taus = np.exp(logp[1:])
    theta = params_to_theta_n(k, taus, Ts)
    theta[n:] *= u_scale  # b1..bn are the u-normalized-input gain terms
    return theta


def _theta_jacobian_n(logp: np.ndarray, Ts: float, u_scale: float, eps: float) -> np.ndarray:
    n_theta = 2 * (len(logp) - 1)
    n_p = len(logp)
    J = np.empty((n_theta, n_p))
    for i in range(n_p):
        dp = np.zeros(n_p)
        dp[i] = eps
        J[:, i] = (theta_from_logp_n(logp + dp, Ts, u_scale)
                   - theta_from_logp_n(logp - dp, Ts, u_scale)) / (2 * eps)
    return J


def _clamp_logp_n(logp: np.ndarray, Ts: float) -> np.ndarray:
    tau_min, tau_max = _tau_bounds(Ts)
    out = logp.copy()
    out[0] = np.clip(out[0], -10.0, 10.0)
    out[1:] = np.clip(out[1:], np.log(tau_min), np.log(tau_max))
    return out


# ==========================================================================
# General-n recursive core (same structure as
# riv_v1_logparam_riv_experiment._dual_estimator_core_logparam, loops
# instead of hardcoded [0],[1] indices)
# ==========================================================================

def _dual_estimator_core_logparam_n(u, y, phi_ar, Q_diag, R, x0, P0_kf, logp0, P0_rls,
                                     lam, n_eta, p, n_off, Ts, u_scale, jac_eps):
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = len(y)

    x = x0.copy()
    Pkf = P0_kf.copy()
    logp = logp0.copy()
    Prls = P0_rls.copy()
    z_lag = np.zeros(n_eta)   # noise-free instrument state, oldest last

    x_hat = np.zeros((N, n))
    logp_hat = np.zeros((N, n_eta + 1))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    theta = theta_from_logp_n(logp, Ts, u_scale)  # [a1..a_n_eta, b1..b_n_eta]

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a = theta[:n_eta]
            b = theta[n_eta:]
            x_new = np.empty(n)
            for i in range(n_eta):
                x_new[i] = a[i] * x[0] + (x[i + 1] if i < n_eta - 1 else 0.0) + b[i] * u_prev
            x_new[idx_beta] = phi_ar @ x[idx_beta:idx_beta + p]
            if p > 1:
                x_new[idx_beta + 1:idx_beta + p] = x[idx_beta:idx_beta + p - 1]
            if n_off == 1:
                x_new[idx_off] = x[idx_off]
            x = x_new

            A = np.zeros((n, n))
            for i in range(n_eta):
                A[i, 0] = a[i]
                if i < n_eta - 1:
                    A[i, i + 1] = 1.0
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
        eta_lags = np.array([x_hat[kk - j, 0] if kk - j >= 0 else 0.0 for j in range(1, n_eta + 1)])
        u_lags = np.array([u[kk - j] if kk - j >= 0 else 0.0 for j in range(1, n_eta + 1)])
        phi_vec = np.concatenate([eta_lags, u_lags])
        z_vec = np.concatenate([z_lag, u_lags])

        J = _theta_jacobian_n(logp, Ts, u_scale, jac_eps)
        psi_phi = J.T @ phi_vec
        psi_z = J.T @ z_vec

        pred = phi_vec @ theta
        err = eta_hat_k - pred

        z_new = float(theta @ z_vec)
        z_lag = np.concatenate([[z_new], z_lag[:-1]])

        Pz = Prls @ psi_z
        denom = lam + psi_phi @ Pz
        Krls = Pz / denom
        logp = logp + Krls * err
        logp = _clamp_logp_n(logp, Ts)
        Prls = (Prls - np.outer(Krls, psi_phi @ Prls)) / lam
        Prls = 0.5 * (Prls + Prls.T)

        theta = theta_from_logp_n(logp, Ts, u_scale)
        logp_hat[kk] = logp

    return x_hat, logp_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def run_dual_estimator_logparam_n(u, y, phi_ar, Q_diag, R, x0, P0_kf, logp0, P0_rls,
                                   lam, n_eta, p, n_off, Ts, u_scale, n_epochs, jac_eps=1e-4):
    logp_cur = logp0.copy()
    logp_epoch_hist = np.zeros((n_epochs, n_eta + 1))
    x_hat = logp_hat = y_hat = Pkf_final = Prls_final = innov_final = S_final = None
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        x0_epoch = x0.copy()
        (x_hat, logp_hat, y_hat, Pkf_final, Prls_final,
         innov_final, S_final) = _dual_estimator_core_logparam_n(
            u, y, phi_ar, Q_diag, R, x0_epoch, P0_kf.copy(), logp_cur, P0_rls.copy(),
            lam, n_eta, p, n_off, Ts, u_scale, jac_eps)
        logp_cur = logp_hat[-1].copy()
        logp_epoch_hist[epoch] = logp_cur
    elapsed = time.perf_counter() - t0
    return x_hat, logp_hat, y_hat, Pkf_final, Prls_final, logp_epoch_hist, elapsed, innov_final, S_final


def run_identification_logparam_n(u, y, phi_coeffs, sigma2_ar, R, q_eta, cfg, dual_cfg,
                                   n_eta: int, tau_init: np.ndarray,
                                   n_epochs=None, jac_eps=1e-4) -> dict:
    setup = build_kf_setup(u, y, phi_coeffs, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale = setup["u_norm"], setup["u_scale"]
    p, n_off = setup["p"], setup["n_off"]

    idx_beta = n_eta
    idx_off = n_eta + p
    n_state = n_eta + p + n_off
    Q_diag = np.concatenate((np.full(n_eta, q_eta),
                              [sigma2_ar], np.full(p - 1, 0.1 * sigma2_ar),
                              [setup["Q_diag"][-1]]))
    x0 = np.zeros(n_state)
    x0[idx_off] = float(np.mean(y))
    P0_kf = dual_cfg.p0_kf_state_var * np.eye(n_state)
    phi_c = setup["phi_c"]

    logp0 = np.log(np.concatenate([[AVG_PATIENT_PARAMS["k"]], tau_init]))
    P0_rls = dual_cfg.p0_rls * np.eye(n_eta + 1)
    n_epochs = dual_cfg.n_epochs if n_epochs is None else n_epochs

    (x_hat, logp_hat, y_hat, Pkf_final, Prls_final, logp_epoch_hist, elapsed,
     innov_final, S_final) = run_dual_estimator_logparam_n(
        u_norm, y, phi_c, Q_diag, R, x0, P0_kf, logp0, P0_rls, dual_cfg.lambda_ff,
        n_eta, p, n_off, cfg.Ts, u_scale, n_epochs, jac_eps)

    logp_final = logp_hat[-1]
    k_f = float(np.exp(logp_final[0]))
    taus_f = np.exp(logp_final[1:])

    passed_white, frac_white = residual_whiteness_test(innov_final)
    _, nis_mean, nis_ok = nis_consistency_check(innov_final, S_final)

    return {"k": k_f, "taus": taus_f, "elapsed": elapsed,
            "whiteness_pass": passed_white, "whiteness_frac_exceed": frac_white,
            "nis_mean": nis_mean, "nis_consistent": nis_ok, "n_eta": n_eta}


# ==========================================================================
# Synthetic self-check, n_eta=3, GENUINELY PE-rich IID input (does the
# generalised mechanism work AT ALL when the data supports it -- separate
# question from "does real data support it", answered by pe_report)
# ==========================================================================

def synthetic_selfcheck_n3(cfg, dual_cfg, n_epochs: int = 10, jac_eps: float = 1e-4) -> list[dict]:
    k_true = 80.0
    taus_true = np.array([0.03, 0.08, 0.25])
    phi_ar_true = np.array([0.6, -0.2, 0.1])
    sigma2_true = 0.05
    c0_true = 1.10
    N = 3000
    seeds = range(6)
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.004, 0.026, size=N)
        theta_true = params_to_theta_n(k_true, taus_true, cfg.Ts)
        eta_true = _simulate_arx_n(theta_true, u, len(taus_true))
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
        res = run_identification_logparam_n(u, y, phi_ar_true, sigma2_true, R_heur, q_eta_heur,
                                             cfg, dual_cfg, n_eta=3,
                                             tau_init=np.array([AVG_PATIENT_PARAMS["tau1"], 0.1,
                                                                 AVG_PATIENT_PARAMS["tau2"]]),
                                             n_epochs=n_epochs, jac_eps=jac_eps)
        k_err = abs(res["k"] - k_true) / k_true
        ok = res["k"] > 0 and k_err < 0.3
        results.append({"seed": seed, "k": res["k"], "taus": res["taus"], "k_err_pct": 100 * k_err,
                         "ok": ok, "whiteness_frac_exceed": res["whiteness_frac_exceed"]})
        print(f"seed={seed}: k={res['k']:.2f} (true 80, err {100*k_err:.1f}%) "
              f"taus={np.array2string(res['taus'], precision=4)} (true {taus_true}) "
              f"ok={ok} whiteness_exceed={res['whiteness_frac_exceed']:.2f}")
    n_ok = sum(r["ok"] for r in results)
    print(f"\nn_eta=3 synthetic self-check (PE-rich input): {n_ok}/{len(seeds)} within 30% of k_true")
    return results


def _simulate_arx_n(theta: np.ndarray, u: np.ndarray, n: int) -> np.ndarray:
    a, b = theta[:n], theta[n:]
    N = len(u)
    y = np.zeros(N)
    for kk in range(N):
        y[kk] = sum(a[i] * (y[kk - i - 1] if kk - i - 1 >= 0 else 0.0) for i in range(n)) \
            + sum(b[i] * (u[kk - i - 1] if kk - i - 1 >= 0 else 0.0) for i in range(n))
    return y


if __name__ == "__main__":
    print("=== regression self-check: n-generalised code matches validated n=2 case ===")
    _selfcheck_n2(CFG.Ts)
    print("\n=== synthetic self-check, n_eta=3, PE-rich input ===")
    synthetic_selfcheck_n3(CFG, DUAL_CFG)
