"""Dual estimator (KF + recursive IV) for a single-pole plant
G(s) = k/(tau*s+1), theta=[a1,b1].

Alternative to riv_dual_estimation.py's two-pole cascade, which has an
intrinsic tau1/tau2 collinearity (corr=-0.87, FIM-confirmed) and a
marginal-stability attractor in the (a1,a2) subspace. Dropping the second
pole needs only a PE-order-2 regressor ([eta_{k-1}, u_{k-1}]) instead of 4.

Reuses the pole-count-agnostic pieces of riv_dual_estimation.py (CFG,
DUAL_CFG, ar_to_state_space, build_process_noise_diag, load_all_recordings,
fit_ar_beta_model_pooled, residual_whiteness_test, nis_consistency_check,
discretize_zoh, p_trace_plateau_check) and reimplements the rest.
"""
from __future__ import annotations

import time

import control
import numba
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import (
    Config, DualConfig, CFG, POLE_MAX, B_MAX, AVG_PATIENT_PARAMS,
    discretize_zoh, build_process_noise_diag,
)

N_ETA_1P = 1
N_THETA_1P = 2

# No single-pole prior exists in the literature (AVG_PATIENT_PARAMS is a
# two-lag fit), so anchor at the same gain with tau at the slower lag.
AVG_PATIENT_PARAMS_1P = {"k": AVG_PATIENT_PARAMS["k"], "tau": AVG_PATIENT_PARAMS["tau2"]}


# Plant model: continuous plant -> discrete ARX(1)

def build_true_ss_1p(k: float, tau: float) -> control.StateSpace:
    Ac = np.array([[-1.0 / tau]])
    Bc = np.array([[k / tau]])
    Cc = np.array([[1.0]])
    Dc = np.array([[0.0]])
    return control.ss(Ac, Bc, Cc, Dc)


def arx_theta_from_discrete_1p(sys_d: control.StateSpace) -> np.ndarray:
    """G(z) = b1/(z - d1) -> a1=d1, b1=b1. Only used to cross-check the closed
    form in the self-consistency test."""
    tf_d = control.tf(sys_d)
    num = np.atleast_1d(np.squeeze(tf_d.num[0][0]))
    den = np.atleast_1d(np.squeeze(tf_d.den[0][0]))
    den = den / den[0]
    num = np.pad(num, (2 - len(num), 0))
    den = np.pad(den, (2 - len(den), 0), constant_values=0.0)
    den[0] = 1.0
    a1 = -den[1]
    b1 = num[1]
    return np.array([a1, b1])


def params_to_theta_1p(k: float, tau: float, Ts: float) -> np.ndarray:
    """Closed-form ZOH discretization: a1=exp(-Ts/tau) is the pole and
    b1=k*(1-a1), so the DC gain b1/(1-a1)=k."""
    a1 = float(np.exp(-Ts / tau))
    b1 = float(k * (1.0 - a1))
    return np.array([a1, b1])


def theta_to_params_1p(theta: np.ndarray, Ts: float) -> tuple[float, float]:
    """Analytic inverse of params_to_theta_1p; a1 must lie in (0, 1)."""
    a1, b1 = theta
    if not (0.0 < a1 < 1.0):
        raise ValueError(f"theta_to_params_1p requires a1 in (0,1), got {a1}")
    tau = float(-Ts / np.log(a1))
    k = float(b1 / (1.0 - a1))
    return k, tau


def theta_to_params_safe_1p(theta: np.ndarray, Ts: float) -> tuple[float, float]:
    """theta_to_params_1p after clipping a1 into (0, POLE_MAX]."""
    a1, b1 = theta
    a1c = float(np.clip(a1, 1e-3, POLE_MAX))
    return theta_to_params_1p(np.array([a1c, b1]), Ts)


def theta_to_observer_canonical_1p(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a1, b1 = theta
    A_eta = np.array([[a1]])
    B_eta = np.array([[b1]])
    C_eta = np.array([[1.0]])
    return A_eta, B_eta, C_eta


def simulate_ss_1p(A: np.ndarray, B: np.ndarray, C: np.ndarray, u: np.ndarray,
                    x0: np.ndarray | None = None) -> np.ndarray:
    u = np.atleast_2d(u.reshape(-1, 1))
    n = A.shape[0]
    N = u.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    y = np.zeros(N)
    for kk in range(N):
        y[kk] = (C @ x)[0]
        x = A @ x + (B @ u[kk])
    return y


def simulate_arx_recursion_1p(theta: np.ndarray, u: np.ndarray) -> np.ndarray:
    a1, b1 = theta
    N = len(u)
    y = np.zeros(N)
    for kk in range(N):
        y_km1 = y[kk - 1] if kk - 1 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        y[kk] = a1 * y_km1 + b1 * u_km1
    return y


def run_self_consistency_checks_1p(cfg: Config) -> None:
    """ARX recursion == forced_response == observer-canonical simulation ==
    closed-form pole, for a plausible test plant."""
    k_test, tau_test = 60.0, 0.15
    sys_c = build_true_ss_1p(k_test, tau_test)
    sys_d = discretize_zoh(sys_c, cfg.Ts)
    theta_zoh = arx_theta_from_discrete_1p(sys_d)
    theta_closed = params_to_theta_1p(k_test, tau_test, cfg.Ts)

    assert np.allclose(theta_zoh, theta_closed, atol=1e-9, rtol=1e-9), \
        f"closed-form theta {theta_closed} does not match control.c2d ZOH theta {theta_zoh}"

    rng = np.random.default_rng(cfg.seed)
    u_test = rng.uniform(0, 2.5, size=500)

    y_arx = simulate_arx_recursion_1p(theta_closed, u_test)
    t = np.arange(len(u_test)) * cfg.Ts
    resp = control.forced_response(sys_d, T=t, U=u_test)
    y_ss = np.asarray(resp.outputs).squeeze()
    assert np.allclose(y_arx, y_ss, atol=1e-6, rtol=1e-6), \
        "ARX recursion does not match discretized state-space response"

    A_eta, B_eta, C_eta = theta_to_observer_canonical_1p(theta_closed)
    y_obs = simulate_ss_1p(A_eta, B_eta, C_eta, u_test)
    assert np.allclose(y_obs, y_arx, atol=1e-9), \
        "Observer-canonical simulation does not match ARX recursion"

    pole_analytic = np.exp(-cfg.Ts / tau_test)
    assert np.isclose(theta_closed[0], pole_analytic, atol=1e-8), \
        "theta pole does not match analytic ZOH pole"

    k_rt, tau_rt = theta_to_params_1p(theta_closed, cfg.Ts)
    assert np.isclose(k_rt, k_test, rtol=1e-9) and np.isclose(tau_rt, tau_test, rtol=1e-9), \
        "theta_to_params_1p is not the inverse of params_to_theta_1p"

    print("[1p.1] Self-consistency checks passed (closed-form/ZOH/ARX/state-space/pole/inverse).")


# Dual estimator core

@numba.njit(cache=True)
def _clamp_theta_1p(a1: float, b1: float) -> tuple[float, float]:
    """Clip a1 (the pole) into (pole_min, POLE_MAX]."""
    pole_min = 1e-3
    pole_max = POLE_MAX
    if a1 < pole_min:
        a1 = pole_min
    elif a1 > pole_max:
        a1 = pole_max
    if b1 > B_MAX:
        b1 = B_MAX
    elif b1 < -B_MAX:
        b1 = -B_MAX
    return a1, b1


@numba.njit(cache=True)
def _structured_kf_prop_1p(P: np.ndarray, Q_diag: np.ndarray, phi_ar: np.ndarray,
                            a1: float, n_eta: int, p: int, n_off: int,
                            temp: np.ndarray, Pminus: np.ndarray) -> None:
    """Structured P- = A P A^T + Q with the eta block reduced to the scalar
    recursion eta_k = a1*eta_{k-1} (b1*u enters the mean update only)."""
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off

    for j in range(n):
        temp[0, j] = a1 * P[0, j]
        s = 0.0
        for l in range(p):
            s += phi_ar[l] * P[idx_beta + l, j]
        temp[idx_beta, j] = s
        for m in range(1, p):
            temp[idx_beta + m, j] = P[idx_beta + m - 1, j]
        if n_off == 1:
            temp[idx_off, j] = P[idx_off, j]

    for i in range(n):
        Pminus[i, 0] = a1 * temp[i, 0]
        s2 = 0.0
        for l in range(p):
            s2 += phi_ar[l] * temp[i, idx_beta + l]
        Pminus[i, idx_beta] = s2
        for m in range(1, p):
            Pminus[i, idx_beta + m] = temp[i, idx_beta + m - 1]
        if n_off == 1:
            Pminus[i, idx_off] = temp[i, idx_off]

    for i in range(n):
        Pminus[i, i] += Q_diag[i]

    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (Pminus[i, j] + Pminus[j, i])
            Pminus[i, j] = avg
            Pminus[j, i] = avg


@numba.njit(cache=True)
def _dual_estimator_core_1p(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, Q_diag: np.ndarray,
                             R: float, x0: np.ndarray, P0_kf: np.ndarray, theta0: np.ndarray,
                             P0_rls: np.ndarray, lam: float, n_eta: int, p: int, n_off: int,
                             gamma_pole: float, a1_prior: float,
                             cov_stride: int,
                             Pkf_checks: np.ndarray, Prls_checks: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = y.shape[0]

    x = x0.copy()
    Pkf = P0_kf.copy()
    theta = theta0.copy()
    Prls = P0_rls.copy()
    z1 = 0.0

    x_hat = np.zeros((N, n))
    theta_hat = np.zeros((N, N_THETA_1P))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    x_new = np.empty(n)
    temp = np.empty((n, n))
    Pminus = np.empty((n, n))
    HP = np.empty(n)
    Kkf = np.empty(n)
    phi_vec = np.empty(N_THETA_1P)
    z_vec = np.empty(N_THETA_1P)
    Pz = np.empty(N_THETA_1P)
    phiP = np.empty(N_THETA_1P)
    Krls = np.empty(N_THETA_1P)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a1 = theta[0]
            b1 = theta[1]
            eta1 = x[0]

            x_new[0] = a1 * eta1 + b1 * u_prev
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

            _structured_kf_prop_1p(Pkf, Q_diag, phi_ar, a1, n_eta, p, n_off, temp, Pminus)
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
            Kkf[i] = HP[i] / S

        for i in range(n):
            x[i] = x[i] + Kkf[i] * innov
        for i in range(n):
            for j in range(n):
                Pkf[i, j] = Pkf[i, j] - Kkf[i] * HP[j]
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
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        phi_vec[0] = eta_km1
        phi_vec[1] = u_km1

        z_vec[0] = z1
        z_vec[1] = u_km1

        for i in range(N_THETA_1P):
            s3 = 0.0
            for j in range(N_THETA_1P):
                s3 += Prls[i, j] * z_vec[j]
            Pz[i] = s3
        denom = lam
        for i in range(N_THETA_1P):
            denom += phi_vec[i] * Pz[i]
        for i in range(N_THETA_1P):
            Krls[i] = Pz[i] / denom

        pred = 0.0
        for i in range(N_THETA_1P):
            pred += phi_vec[i] * theta[i]
        err = eta_hat_k - pred

        z_new = theta[0] * z1 + theta[1] * u_km1
        z1 = z_new

        for i in range(N_THETA_1P):
            theta[i] = theta[i] + Krls[i] * err
        theta[0], theta[1] = _clamp_theta_1p(theta[0], theta[1])
        if gamma_pole > 0.0:
            theta[0] += gamma_pole * (a1_prior - theta[0])
            theta[0], theta[1] = _clamp_theta_1p(theta[0], theta[1])
        for i in range(N_THETA_1P):
            s4 = 0.0
            for j in range(N_THETA_1P):
                s4 += phi_vec[j] * Prls[j, i]
            phiP[i] = s4
        for i in range(N_THETA_1P):
            for j in range(N_THETA_1P):
                Prls[i, j] = (Prls[i, j] - Krls[i] * phiP[j]) / lam

        for i in range(N_THETA_1P):
            theta_hat[kk, i] = theta[i]

        if cov_stride > 0 and kk % cov_stride == 0:
            idx_chk = kk // cov_stride
            if idx_chk < Pkf_checks.shape[0]:
                for i in range(n):
                    for j in range(n):
                        Pkf_checks[idx_chk, i, j] = Pkf[i, j]
                for i in range(N_THETA_1P):
                    for j in range(N_THETA_1P):
                        Prls_checks[idx_chk, i, j] = Prls[i, j]

    return x_hat, theta_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def run_dual_estimator_1p(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, Q_diag: np.ndarray,
                           R: float, x0: np.ndarray, P0_kf: np.ndarray, theta0: np.ndarray,
                           P0_rls: np.ndarray, lam: float, n_eta: int, p: int, n_off: int = 0,
                           gamma_pole: float = 0.0, a1_prior: float = 0.0,
                           n_epochs: int = 1, n_snapshots: int = 0,
                           save_epochs: set[int] | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], float,
                                      np.ndarray, np.ndarray]:
    n = n_eta + p + n_off
    N = len(y)
    cov_stride = max(1, N // n_snapshots) if n_snapshots > 0 else 0
    n_checks = (N // cov_stride + 1) if cov_stride > 0 else 1
    Pkf_checks = np.zeros((n_checks, n, n))
    Prls_checks = np.zeros((n_checks, N_THETA_1P, N_THETA_1P))
    save_epochs = save_epochs or set()

    theta_cur = np.ascontiguousarray(theta0, dtype=np.float64).copy()
    P0_kf_c = np.ascontiguousarray(P0_kf, dtype=np.float64)
    P0_rls_c = np.ascontiguousarray(P0_rls, dtype=np.float64)
    theta_epoch_hist = np.zeros((n_epochs, N_THETA_1P))
    theta_hat_by_epoch: dict[int, np.ndarray] = {}

    x_hat = theta_hat = y_hat = Pkf_final = Prls_final = innov_final = S_final = None
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        x0_epoch = np.ascontiguousarray(x0, dtype=np.float64).copy()
        x_hat, theta_hat, y_hat, Pkf_final, Prls_final, innov_final, S_final = (
            _dual_estimator_core_1p(
                u, y, phi_ar, Q_diag, float(R), x0_epoch, P0_kf_c.copy(), theta_cur, P0_rls_c.copy(),
                float(lam), n_eta, p, n_off, float(gamma_pole), float(a1_prior),
                cov_stride, Pkf_checks, Prls_checks,
            )
        )
        theta_cur = theta_hat[-1].copy()
        theta_epoch_hist[epoch] = theta_cur
        if epoch in save_epochs:
            theta_hat_by_epoch[epoch] = theta_hat.copy()
    elapsed = time.perf_counter() - t0

    return (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, Pkf_checks, Prls_checks,
            theta_epoch_hist, theta_hat_by_epoch, elapsed, innov_final, S_final)


# Dense (non-jitted) reference core

def riv_step_dense_1p(theta: np.ndarray, P: np.ndarray, phi_vec: np.ndarray, z_vec: np.ndarray,
                       target: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    Pz = P @ z_vec
    denom = lam + phi_vec @ Pz
    K = Pz / denom
    theta_new = theta + K * (target - phi_vec @ theta)
    P_new = P - np.outer(K, phi_vec @ P)
    P_new = P_new / lam
    return theta_new, P_new


def reference_dual_estimator_dense_1p(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray,
                                       Q_diag: np.ndarray, R: float, x0: np.ndarray,
                                       P0_kf: np.ndarray, theta0: np.ndarray, P0_rls: np.ndarray,
                                       lam: float, n_eta: int, p: int, n_off: int = 0,
                                       gamma_pole: float = 0.0, a1_prior: float = 0.0
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Dense-matmul reference dual estimator (plain KF + textbook RIV, no numba),
    used to validate the jitted core."""
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = len(y)

    x = x0.copy().astype(float)
    Pkf = P0_kf.copy().astype(float)
    theta = theta0.copy().astype(float)
    Prls = P0_rls.copy().astype(float)
    Qmat = np.diag(Q_diag)
    z1 = 0.0

    x_hat = np.zeros((N, n))
    theta_hat = np.zeros((N, N_THETA_1P))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a1, b1 = theta
            A_beta = np.zeros((p, p))
            A_beta[0, :] = phi_ar
            if p > 1:
                A_beta[1:, :-1] = np.eye(p - 1)
            A = np.zeros((n, n))
            A[0, 0] = a1
            A[n_eta:n_eta + p, n_eta:n_eta + p] = A_beta
            if n_off == 1:
                A[idx_off, idx_off] = 1.0
            B = np.zeros(n)
            B[0] = b1

            x = A @ x + B * u_prev
            Pkf = A @ Pkf @ A.T + Qmat
            Pkf = 0.5 * (Pkf + Pkf.T)

        H = np.zeros(n)
        H[0] = -1.0
        H[idx_beta] = 1.0
        if n_off == 1:
            H[idx_off] = 1.0
        innov = y[kk] - float(H @ x)
        S = float(H @ Pkf @ H + R)
        innov_hist[kk] = innov
        S_hist[kk] = S
        K = (Pkf @ H) / S
        x = x + K * innov
        Pkf = Pkf - np.outer(K, H @ Pkf)
        Pkf = 0.5 * (Pkf + Pkf.T)

        eta_hat_k = x[0]
        x_hat[kk] = x
        y_hat[kk] = float(H @ x)

        eta_km1 = x_hat[kk - 1, 0] if kk - 1 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        phi_vec = np.array([eta_km1, u_km1])

        z_vec = np.array([z1, u_km1])
        z_new = theta[0] * z1 + theta[1] * u_km1
        z1 = z_new

        theta, Prls = riv_step_dense_1p(theta, Prls, phi_vec, z_vec, eta_hat_k, lam)
        theta[0], theta[1] = _clamp_theta_1p(theta[0], theta[1])
        if gamma_pole > 0.0:
            theta[0] += gamma_pole * (a1_prior - theta[0])
            theta[0], theta[1] = _clamp_theta_1p(theta[0], theta[1])
        theta_hat[kk] = theta

    return x_hat, theta_hat, y_hat, Pkf, Prls, innov_hist, S_hist


# KF/RIV setup and top-level identification driver

def build_kf_setup_1p(u: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray, sigma2_ar: float,
                       q_eta: float, cfg: Config, dual_cfg: DualConfig,
                       theta0_params: dict | None = None) -> dict:
    """Single-pole analog of build_kf_setup. theta0_params overrides
    AVG_PATIENT_PARAMS_1P (k/tau keys, physical units)."""
    p = len(phi_coeffs)
    n_off = 1
    idx_off = N_ETA_1P + p
    n = N_ETA_1P + p + n_off

    u_scale = float(np.std(u))
    u_norm = u / u_scale

    prior = theta0_params or AVG_PATIENT_PARAMS_1P
    theta0 = params_to_theta_1p(prior["k"], prior["tau"], cfg.Ts)
    theta0[1] *= u_scale

    q_off = dual_cfg.q_off_frac * float(np.var(y))
    Q_diag = np.concatenate((
        build_process_noise_diag(N_ETA_1P, p, q_eta, sigma2_ar, 0.1 * sigma2_ar),
        [q_off],
    ))
    x0 = np.zeros(n)
    x0[idx_off] = float(np.mean(y))
    P0_kf = dual_cfg.p0_kf_state_var * np.eye(n)
    P0_rls = dual_cfg.p0_rls * np.eye(N_THETA_1P)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    Q_c = np.ascontiguousarray(Q_diag, dtype=np.float64)
    a1_prior = float(theta0[0])

    return {
        "u_norm": u_norm, "u_scale": u_scale, "theta0": theta0, "Q_diag": Q_c,
        "x0": x0, "P0_kf": P0_kf, "P0_rls": P0_rls, "phi_c": phi_c,
        "a1_prior": a1_prior, "n": n, "n_off": n_off, "p": p,
    }


def theta_to_params_physical_1p(theta: np.ndarray, u_scale: float, Ts: float) -> tuple[float, float]:
    k, tau = theta_to_params_safe_1p(theta, Ts)
    return k / u_scale, tau


def run_verification_suite_1p(cfg: Config) -> None:
    run_self_consistency_checks_1p(cfg)


if __name__ == "__main__":
    run_verification_suite_1p(CFG)
    print("riv_single_pole_estimation.py: self-consistency checks OK. "
          "See test_riv_single_pole_estimation.py for the full correctness suite.")
