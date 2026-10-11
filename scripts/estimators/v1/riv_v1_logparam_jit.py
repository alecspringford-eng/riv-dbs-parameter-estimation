"""Numba-JIT log-parameterised dual KF+RIV core.

A pure-Python core would call the `control` package's params_to_theta once per
theta(logp) evaluation (~0.2ms/call, 6 calls/sample for the finite-difference
Jacobian), which is too slow for real-time use and not numba-compatible.
Here params_to_theta is replaced by a closed-form ZOH discretization of the
two-lag cascade, validated against the `control` version (max abs error
7.3e-12 over 2000 random draws, 5.0e-8 near tau1~=tau2, which needs a
separate limiting-case formula).

    p1 = exp(-Ts/tau1), p2 = exp(-Ts/tau2)
    a1 = p1+p2,  a2 = -p1*p2
    b1 = Bd2,    b2 = Ad21*Bd1 - p1*Bd2
  where (tau1 != tau2):
    Ad21 = tau1*(p1-p2)/(tau1-tau2)
    Bd1  = k*(1-p1)
    Bd2  = k*(tau2*(1-p2) - tau1*(1-p1))/(tau2-tau1)
  and the tau1==tau2==tau limit (L'Hopital):
    Ad21 = (Ts/tau)*p,  Bd2 = k*((1-p) - (Ts/tau)*p),  p=exp(-Ts/tau)

The rest (KF over eta/beta/offset, RIV update in logp space, instrument z,
clamps, epoch loop) is written in numba's array-preallocation style like riv_dual_estimation.py's
`_dual_estimator_core`.
"""

from __future__ import annotations

import time

import numba
import numpy as np

from riv_dual_estimation import (
    CFG, DUAL_CFG, POLE_MAX, AVG_PATIENT_PARAMS,
    load_all_recordings, fit_ar_beta_model_per_recording,
    params_to_theta, build_kf_setup, residual_whiteness_test,
    nis_consistency_check,
)

POLE_MIN = 1e-3
LOGK_BOUND = 10.0
DEGENERATE_REL_TOL = 1e-9


@numba.njit(cache=True)
def _theta_from_kt1t2(k: float, tau1: float, tau2: float, Ts: float) -> tuple[float, float, float, float]:
    if abs(tau1 - tau2) > DEGENERATE_REL_TOL * max(tau1, tau2):
        a = 1.0 / tau1
        b = 1.0 / tau2
        p1 = np.exp(-Ts * a)
        p2 = np.exp(-Ts * b)
        Ad11 = p1
        Ad21 = tau1 * (p1 - p2) / (tau1 - tau2)
        Bd1 = k * (1.0 - p1)
        Bd2 = k * (tau2 * (1.0 - p2) - tau1 * (1.0 - p1)) / (tau2 - tau1)
        a1 = p1 + p2
        a2 = -(p1 * p2)
    else:
        tau = 0.5 * (tau1 + tau2)
        a = 1.0 / tau
        p = np.exp(-Ts * a)
        Ad11 = p
        Ad21 = (Ts / tau) * p
        Bd1 = k * (1.0 - p)
        Bd2 = k * ((1.0 - p) - (Ts / tau) * p)
        a1 = 2.0 * p
        a2 = -(p * p)
    b1 = Bd2
    b2 = Ad21 * Bd1 - Ad11 * Bd2
    return a1, a2, b1, b2


@numba.njit(cache=True)
def _theta_jacobian(l0: float, l1: float, l2: float, Ts: float, u_scale: float,
                     eps: float, J: np.ndarray) -> None:
    """Central-difference d(theta)/d(logp) into the preallocated (4,3) J."""
    for i in range(3):
        dl0 = l0 + (eps if i == 0 else 0.0)
        dl1 = l1 + (eps if i == 1 else 0.0)
        dl2 = l2 + (eps if i == 2 else 0.0)
        a1p, a2p, b1p, b2p = _theta_from_kt1t2(np.exp(dl0), np.exp(dl1), np.exp(dl2), Ts)
        dl0 = l0 - (eps if i == 0 else 0.0)
        dl1 = l1 - (eps if i == 1 else 0.0)
        dl2 = l2 - (eps if i == 2 else 0.0)
        a1m, a2m, b1m, b2m = _theta_from_kt1t2(np.exp(dl0), np.exp(dl1), np.exp(dl2), Ts)
        J[0, i] = (a1p - a1m) / (2 * eps)
        J[1, i] = (a2p - a2m) / (2 * eps)
        J[2, i] = (b1p * u_scale - b1m * u_scale) / (2 * eps)
        J[3, i] = (b2p * u_scale - b2m * u_scale) / (2 * eps)


@numba.njit(cache=True)
def _clamp_logp(l0: float, l1: float, l2: float, tau_min: float, tau_max: float
                 ) -> tuple[float, float, float]:
    if l0 > LOGK_BOUND:
        l0 = LOGK_BOUND
    elif l0 < -LOGK_BOUND:
        l0 = -LOGK_BOUND
    log_tmin, log_tmax = np.log(tau_min), np.log(tau_max)
    if l1 > log_tmax:
        l1 = log_tmax
    elif l1 < log_tmin:
        l1 = log_tmin
    if l2 > log_tmax:
        l2 = log_tmax
    elif l2 < log_tmin:
        l2 = log_tmin
    return l0, l1, l2


@numba.njit(cache=True)
def _dual_estimator_core_logparam_jit(u, y, phi_ar, Q_diag, R, x0, P0_kf, logp0, P0_rls,
                                       lam, p, n_off, Ts, u_scale, jac_eps, tau_min, tau_max):
    n_eta = 2
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = y.shape[0]

    x = x0.copy()
    Pkf = P0_kf.copy()
    l0, l1, l2 = logp0[0], logp0[1], logp0[2]
    Prls = P0_rls.copy()
    z1 = 0.0
    z2 = 0.0

    x_hat = np.zeros((N, n))
    logp_hat = np.zeros((N, 3))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    a1, a2, b1u, b2u = _theta_from_kt1t2(np.exp(l0), np.exp(l1), np.exp(l2), Ts)
    b1 = b1u * u_scale
    b2 = b2u * u_scale

    x_new = np.empty(n)
    temp = np.empty((n, n))
    Pminus = np.empty((n, n))
    A = np.zeros((n, n))
    HP = np.empty(n)
    J = np.empty((4, 3))
    phi_vec = np.empty(4)
    z_vec = np.empty(4)
    psi_phi = np.empty(3)
    psi_z = np.empty(3)
    Pz = np.empty(3)
    Krls = np.empty(3)
    phiP = np.empty(3)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            eta1 = x[0]
            eta2 = x[1]
            x_new[0] = a1 * eta1 + eta2 + b1 * u_prev
            x_new[1] = a2 * eta1 + b2 * u_prev
            s = 0.0
            for l in range(p):
                s += phi_ar[l] * x[idx_beta + l]
            x_new[idx_beta] = s
            for m in range(1, p):
                x_new[idx_beta + m] = x[idx_beta + m - 1]
            if n_off == 1:
                x_new[idx_off] = x[idx_off]
            for i in range(n):
                x[i] = x_new[i]

            for i in range(n):
                for j in range(n):
                    A[i, j] = 0.0
            A[0, 0] = a1
            A[0, 1] = 1.0
            A[1, 0] = a2
            for l in range(p):
                A[idx_beta, idx_beta + l] = phi_ar[l]
            for m in range(1, p):
                A[idx_beta + m, idx_beta + m - 1] = 1.0
            if n_off == 1:
                A[idx_off, idx_off] = 1.0

            for i in range(n):
                for j in range(n):
                    s2 = 0.0
                    for kdx in range(n):
                        s2 += A[i, kdx] * Pkf[kdx, j]
                    temp[i, j] = s2
            for i in range(n):
                for j in range(n):
                    s3 = 0.0
                    for kdx in range(n):
                        s3 += temp[i, kdx] * A[j, kdx]
                    Pminus[i, j] = s3
            for i in range(n):
                Pminus[i, i] += Q_diag[i]
            for i in range(n):
                for j in range(i + 1, n):
                    avg = 0.5 * (Pminus[i, j] + Pminus[j, i])
                    Pminus[i, j] = avg
                    Pminus[j, i] = avg
            for i in range(n):
                for j in range(n):
                    Pkf[i, j] = Pminus[i, j]

        off = x[idx_off] if n_off == 1 else 0.0
        h_pred = x[idx_beta] - x[0] + off
        innov = y[kk] - h_pred

        for j in range(n):
            HP[j] = -Pkf[0, j] + Pkf[idx_beta, j]
            if n_off == 1:
                HP[j] += Pkf[idx_off, j]
        S = -HP[0] + HP[idx_beta] + R
        if n_off == 1:
            S += HP[idx_off]
        innov_hist[kk] = innov
        S_hist[kk] = S

        for i in range(n):
            Ki = HP[i] / S
            x[i] = x[i] + Ki * innov
        for i in range(n):
            Ki = HP[i] / S
            for j in range(n):
                Pkf[i, j] = Pkf[i, j] - Ki * HP[j]
        for i in range(n):
            for j in range(i + 1, n):
                avg = 0.5 * (Pkf[i, j] + Pkf[j, i])
                Pkf[i, j] = avg
                Pkf[j, i] = avg

        eta_hat_k = x[0]
        for i in range(n):
            x_hat[kk, i] = x[i]
        y_hat[kk] = x[idx_beta] - x[0] + (x[idx_off] if n_off == 1 else 0.0)

        eta_km1 = x_hat[kk - 1, 0] if kk - 1 >= 0 else 0.0
        eta_km2 = x_hat[kk - 2, 0] if kk - 2 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        u_km2 = u[kk - 2] if kk - 2 >= 0 else 0.0
        phi_vec[0] = eta_km1
        phi_vec[1] = eta_km2
        phi_vec[2] = u_km1
        phi_vec[3] = u_km2
        z_vec[0] = z1
        z_vec[1] = z2
        z_vec[2] = u_km1
        z_vec[3] = u_km2

        _theta_jacobian(l0, l1, l2, Ts, u_scale, jac_eps, J)
        for i in range(3):
            s4 = 0.0
            s5 = 0.0
            for j in range(4):
                s4 += J[j, i] * phi_vec[j]
                s5 += J[j, i] * z_vec[j]
            psi_phi[i] = s4
            psi_z[i] = s5

        pred = a1 * phi_vec[0] + a2 * phi_vec[1] + b1 * phi_vec[2] + b2 * phi_vec[3]
        err = eta_hat_k - pred

        z_new = a1 * z1 + a2 * z2 + b1 * u_km1 + b2 * u_km2
        z2 = z1
        z1 = z_new

        for i in range(3):
            s6 = 0.0
            for j in range(3):
                s6 += Prls[i, j] * psi_z[j]
            Pz[i] = s6
        denom = lam
        for i in range(3):
            denom += psi_phi[i] * Pz[i]
        for i in range(3):
            Krls[i] = Pz[i] / denom

        l0 = l0 + Krls[0] * err
        l1 = l1 + Krls[1] * err
        l2 = l2 + Krls[2] * err
        l0, l1, l2 = _clamp_logp(l0, l1, l2, tau_min, tau_max)

        for i in range(3):
            s7 = 0.0
            for j in range(3):
                s7 += psi_phi[j] * Prls[j, i]
            phiP[i] = s7
        for i in range(3):
            for j in range(3):
                Prls[i, j] = (Prls[i, j] - Krls[i] * phiP[j]) / lam
        for i in range(3):
            for j in range(i + 1, 3):
                avgp = 0.5 * (Prls[i, j] + Prls[j, i])
                Prls[i, j] = avgp
                Prls[j, i] = avgp

        a1, a2, b1u, b2u = _theta_from_kt1t2(np.exp(l0), np.exp(l1), np.exp(l2), Ts)
        b1 = b1u * u_scale
        b2 = b2u * u_scale
        logp_hat[kk, 0] = l0
        logp_hat[kk, 1] = l1
        logp_hat[kk, 2] = l2

    return x_hat, logp_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def run_dual_estimator_logparam_jit(u, y, phi_ar, Q_diag, R, x0, P0_kf, logp0, P0_rls,
                                     lam, p, n_off, Ts, u_scale, n_epochs, jac_eps=1e-4):
    tau_min = -Ts / np.log(POLE_MIN)
    tau_max = -Ts / np.log(POLE_MAX)
    logp_cur = logp0.copy()
    logp_epoch_hist = np.zeros((n_epochs, 3))
    x_hat = logp_hat = y_hat = Pkf_final = Prls_final = innov_final = S_final = None
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        x0_epoch = x0.copy()
        x_hat, logp_hat, y_hat, Pkf_final, Prls_final, innov_final, S_final = (
            _dual_estimator_core_logparam_jit(
                u, y, phi_ar, Q_diag, float(R), x0_epoch, P0_kf.copy(), logp_cur,
                P0_rls.copy(), float(lam), p, n_off, Ts, u_scale, jac_eps, tau_min, tau_max,
            )
        )
        logp_cur = logp_hat[-1].copy()
        logp_epoch_hist[epoch] = logp_cur
    elapsed = time.perf_counter() - t0
    return x_hat, logp_hat, y_hat, Pkf_final, Prls_final, logp_epoch_hist, elapsed, innov_final, S_final


def run_identification_logparam_jit(u, y, phi_coeffs, sigma2_ar, R, q_eta, cfg, dual_cfg,
                                     n_epochs=None, jac_eps=1e-4) -> dict:
    setup = build_kf_setup(u, y, phi_coeffs, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale = setup["u_norm"], setup["u_scale"]
    Q_c, x0, P0_kf = setup["Q_diag"], setup["x0"], setup["P0_kf"]
    phi_c, n_off, p = setup["phi_c"], setup["n_off"], setup["p"]

    logp0 = np.log(np.array([
        AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"], AVG_PATIENT_PARAMS["tau2"],
    ]))
    P0_rls = dual_cfg.p0_rls * np.eye(3)
    n_epochs = dual_cfg.n_epochs if n_epochs is None else n_epochs

    x_hat, logp_hat, y_hat, Pkf_final, Prls_final, logp_epoch_hist, elapsed, innov_final, S_final = (
        run_dual_estimator_logparam_jit(
            u_norm, y, phi_c, Q_c, R, x0, P0_kf, logp0, P0_rls, dual_cfg.lambda_ff,
            p, n_off, cfg.Ts, u_scale, n_epochs, jac_eps,
        )
    )

    logp_final = logp_hat[-1]
    k_f = float(np.exp(logp_final[0]))
    tau1_f = float(np.exp(logp_final[1]))
    tau2_f = float(np.exp(logp_final[2]))

    passed_white, frac_white = residual_whiteness_test(innov_final)
    _, nis_mean, nis_ok = nis_consistency_check(innov_final, S_final)

    epoch_k = np.exp(logp_epoch_hist[:, 0])
    epoch_tau1 = np.exp(logp_epoch_hist[:, 1])
    epoch_tau2 = np.exp(logp_epoch_hist[:, 2])

    return {
        "k": k_f, "tau1": tau1_f, "tau2": tau2_f, "elapsed": elapsed,
        "whiteness_pass": passed_white, "whiteness_frac_exceed": frac_white,
        "nis_mean": nis_mean, "nis_consistent": nis_ok,
        "epoch_k": epoch_k, "epoch_tau1": epoch_tau1, "epoch_tau2": epoch_tau2,
        "x_hat": x_hat, "logp_hat": logp_hat, "y_hat": y_hat,
        "innov": innov_final, "S": S_final, "u_norm": u_norm, "u_scale": u_scale,
        "n_off": n_off, "p": p,
    }


# Validation: the closed-form theta_from_kt1t2 must match the control-based
# params_to_theta

def _selfcheck_theta_formula(Ts: float, n: int = 5000) -> None:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n):
        k = rng.uniform(1, 300)
        tau1 = rng.uniform(0.005, 1.0)
        tau2 = rng.uniform(0.005, 1.0)
        ref = params_to_theta(k, tau1, tau2, Ts)
        approx = np.array(_theta_from_kt1t2(k, tau1, tau2, Ts))
        max_err = max(max_err, float(np.max(np.abs(ref - approx))))
    max_err_deg = 0.0
    for _ in range(1000):
        k = rng.uniform(1, 300)
        tau1 = rng.uniform(0.01, 0.5)
        tau2 = tau1 * (1 + rng.uniform(-1e-4, 1e-4))
        ref = params_to_theta(k, tau1, tau2, Ts)
        approx = np.array(_theta_from_kt1t2(k, tau1, tau2, Ts))
        max_err_deg = max(max_err_deg, float(np.max(np.abs(ref - approx))))
    print(f"[selfcheck] closed-form theta_from_kt1t2 vs control-based params_to_theta: "
          f"max_err={max_err:.2e} (generic, n={n}), max_err_deg={max_err_deg:.2e} (near-degenerate tau1~=tau2)")
    assert max_err < 1e-6 and max_err_deg < 1e-4, "closed-form ZOH formula disagrees with validated params_to_theta"


def _timing_benchmark(cfg, dual_cfg) -> None:
    """Post-compile speed at the production epoch count on one real recording."""
    recordings = load_all_recordings(cfg)
    ar_fits = fit_ar_beta_model_per_recording(recordings, cfg)
    name = "system_id_4.csv"
    u, y = recordings[name]
    f = ar_fits[name]
    R = 0.05 * f["sigma2_ar"]
    q_eta = 0.5 * float(np.var(y))
    # warm up (compiles)
    run_identification_logparam_jit(u, y, f["phi_coeffs"], f["sigma2_ar"], R, q_eta,
                                     cfg, dual_cfg, n_epochs=1)
    res = run_identification_logparam_jit(u, y, f["phi_coeffs"], f["sigma2_ar"], R, q_eta,
                                           cfg, dual_cfg, n_epochs=dual_cfg.n_epochs)
    print(f"[timing] {name}, n_epochs={dual_cfg.n_epochs} (production default), "
          f"post-warmup: {res['elapsed']:.3f}s")


if __name__ == "__main__":
    print("=== closed-form ZOH formula validation ===")
    _selfcheck_theta_formula(CFG.Ts)
    print("\n=== steady-state timing benchmark ===")
    _timing_benchmark(CFG, DUAL_CFG)
