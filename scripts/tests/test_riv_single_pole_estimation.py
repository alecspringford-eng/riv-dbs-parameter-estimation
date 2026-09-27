import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "estimators/single_pole"))
from riv_dual_estimation import CFG, ar_to_state_space
from riv_single_pole_estimation import (
    N_ETA_1P, N_THETA_1P,
    theta_to_observer_canonical_1p, simulate_ss_1p,
    _clamp_theta_1p, _structured_kf_prop_1p, _dual_estimator_core_1p,
    reference_dual_estimator_dense_1p, run_self_consistency_checks_1p,
    params_to_theta_1p, theta_to_params_1p,
)


def test_self_consistency_checks_pass():
    run_self_consistency_checks_1p(CFG)


def test_theta_to_params_1p_is_inverse():
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = rng.uniform(1.0, 200.0)
        tau = rng.uniform(0.01, 1.0)
        theta = params_to_theta_1p(k, tau, CFG.Ts)
        k_rt, tau_rt = theta_to_params_1p(theta, CFG.Ts)
        assert np.isclose(k_rt, k, rtol=1e-9)
        assert np.isclose(tau_rt, tau, rtol=1e-9)


def test_structured_kf_prop_1p_matches_dense():
    rng = np.random.default_rng(1)
    p = 7
    n = N_ETA_1P + p

    a1 = rng.uniform(0.01, 0.99)
    phi_ar = rng.normal(scale=0.3, size=p)
    M = rng.normal(size=(n, n))
    P = M @ M.T
    Q_diag = rng.uniform(1e-6, 1e-2, size=n)

    A_beta = np.zeros((p, p))
    A_beta[0, :] = phi_ar
    if p > 1:
        A_beta[1:, :-1] = np.eye(p - 1)
    A = np.zeros((n, n))
    A[0, 0] = a1
    A[N_ETA_1P:, N_ETA_1P:] = A_beta
    dense = A @ P @ A.T + np.diag(Q_diag)

    temp = np.empty((n, n))
    Pminus = np.empty((n, n))
    _structured_kf_prop_1p(P, Q_diag, phi_ar, a1, N_ETA_1P, p, 0, temp, Pminus)

    assert np.allclose(Pminus, 0.5 * (dense + dense.T), atol=1e-9)


def test_jit_core_1p_matches_dense_reference_and_covariance_sanity():
    rng = np.random.default_rng(2)
    p = 5
    N = 150

    phi_ar = np.array([0.6, -0.1, 0.05, 0.02, -0.01])
    theta_true = np.array([0.55, 0.3])
    u = rng.uniform(0, 2.5, size=N)

    A_eta_t, B_eta_t, C_eta_t = theta_to_observer_canonical_1p(theta_true)
    eta_true = simulate_ss_1p(A_eta_t, B_eta_t, C_eta_t, u)
    A_beta, C_beta = ar_to_state_space(phi_ar)
    x_beta = np.zeros(p)
    beta_sim = np.zeros(N)
    w = rng.normal(0.0, 0.05, size=N)
    for kk in range(N):
        beta_sim[kk] = (C_beta @ x_beta)[0]
        x_beta = A_beta @ x_beta
        x_beta[0] += w[kk]
    y = beta_sim - eta_true + rng.normal(0.0, 0.05, size=N)

    theta0 = theta_true * 1.1
    P0_rls = 1e3 * np.eye(N_THETA_1P)
    lam = 0.999
    cov_stride = 1
    n_checks = N // cov_stride + 1

    for n_off in (0, 1):
        n = N_ETA_1P + p + n_off
        Q_diag = np.concatenate((np.full(N_ETA_1P, 0.1), np.full(p, 0.1)))
        Q_diag[N_ETA_1P] = 5.0
        if n_off == 1:
            Q_diag = np.concatenate((Q_diag, [0.01]))
        x0 = np.zeros(n)
        P0_kf = 10.0 * np.eye(n)
        Pkf_checks = np.zeros((n_checks, n, n))
        Prls_checks = np.zeros((n_checks, N_THETA_1P, N_THETA_1P))
        gp = 0.0 if n_off == 0 else 0.05

        (x_hat_fast, theta_hat_fast, y_hat_fast, Pkf_final, Prls_final,
         innov_fast, S_fast) = _dual_estimator_core_1p(
            u, y, phi_ar, Q_diag, 1e-4, x0, P0_kf, theta0, P0_rls, lam, N_ETA_1P, p, n_off,
            gp, 0.5, cov_stride, Pkf_checks, Prls_checks)

        (x_hat_ref, theta_hat_ref, y_hat_ref, Pkf_final_ref, Prls_final_ref,
         innov_ref, S_ref) = reference_dual_estimator_dense_1p(
            u, y, phi_ar, Q_diag, 1e-4, x0, P0_kf, theta0, P0_rls, lam, N_ETA_1P, p, n_off, gp, 0.5)

        assert np.allclose(x_hat_fast, x_hat_ref, atol=1e-8), f"state mismatch (n_off={n_off})"
        assert np.allclose(theta_hat_fast, theta_hat_ref, atol=1e-8), f"theta mismatch (n_off={n_off})"
        assert np.allclose(y_hat_fast, y_hat_ref, atol=1e-8), f"y_hat mismatch (n_off={n_off})"
        assert np.allclose(Pkf_final, Pkf_final_ref, atol=1e-8), f"Pkf mismatch (n_off={n_off})"
        assert np.allclose(Prls_final, Prls_final_ref, atol=1e-8), f"Prls mismatch (n_off={n_off})"
        assert np.allclose(innov_fast, innov_ref, atol=1e-8), f"innov mismatch (n_off={n_off})"
        assert np.allclose(S_fast, S_ref, atol=1e-8), f"S mismatch (n_off={n_off})"

        for idx in range(n_checks):
            Pc = Pkf_checks[idx]
            if np.all(Pc == 0.0):
                continue
            assert np.allclose(Pc, Pc.T, atol=1e-8)
            assert np.linalg.eigvalsh(Pc).min() >= -1e-8


def test_clamp_theta_1p_bounds():
    a1, b1 = _clamp_theta_1p(5.0, 1e6)
    assert 0.0 < a1 <= 0.95
    assert b1 == 10.0
    a1, b1 = _clamp_theta_1p(-5.0, -1e6)
    assert a1 >= 1e-3
    assert b1 == -10.0
