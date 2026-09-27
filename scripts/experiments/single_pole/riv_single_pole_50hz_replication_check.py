"""riv_single_pole_50hz_replication_check.py -- faithful single-pole port of
the OLD prototype scenario that DID recover parameters reliably
(".../Internship/Code/Personal_Code/Linear-models/RIV Kalman filter/
riv_dual_estimation.py", predates this repo; INVESTIGATION.md Entry 16).

Per user: same setup as the one that worked, swapped to (k, tau) instead
of (k, tau1, tau2). Ported exactly, not reusing this repo's 2026-data-
amplitude-matched build_kf_setup_1p, since fidelity to what worked is the
point:
  - fs=50 Hz (Ts=0.02), N_sim=80000 (vs this repo's 25 Hz/N~3000)
  - u: rate-limited multisine, amplitude scale u_max=2.5 (own scale, not
    matched to real u -- so the noise constants below are self-consistent
    with it, not mismatched the way Entry 43 found the ported-verbatim
    constants to be when dropped into real-u-scaled code)
  - NO offset term in y_true and NO offset state in the KF (n_off=0) --
    the old script's synthetic y never had a baseline for eta to absorb,
    so root cause 1 (offset-hijack) cannot manifest here regardless
  - initial guess perturbed +-50% around the TRUE value, not this repo's
    fixed generic AVG_PATIENT_PARAMS prior (which can be arbitrarily far
    from an arbitrary true tau)
  - fixed (non-data-driven) R=0.1, q_eta=1e4, q_beta_nominal=5.0,
    q_beta_other=0.1

Reuses riv_single_pole_estimation.py's already-validated core
(_dual_estimator_core_1p, run_dual_estimator_1p, etc.) and
riv_dual_estimation.py's pole-count-agnostic ar_to_state_space /
build_process_noise_diag. Does not modify either file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/single_pole"))
from riv_dual_estimation import ar_to_state_space, build_process_noise_diag
from riv_single_pole_estimation import (
    N_ETA_1P, params_to_theta_1p, theta_to_params_1p,
    theta_to_observer_canonical_1p, simulate_ss_1p,
    run_dual_estimator_1p,
)


@dataclass(frozen=True)
class OldConfig:
    fs: float = 50.0
    Ts: float = 0.02
    k_true: float = 62.11
    tau_true: float = 0.05          # single pole -- old script's own tau1_true
    perturb_frac: float = 0.5
    seed: int = 0
    u_max: float = 2.5
    du_max: float = 1.5
    N_sim: int = 80000
    multisine_freq_fracs: tuple[float, ...] = (0.2, 0.5, 0.8)
    p_ar_fallback: int = 12
    sigma2_ar_fallback: float = 1e-2
    R: float = 0.1
    q_eta_default: float = 1e4
    q_beta_nominal: float = 5.0
    q_beta_other_default: float = 0.1
    lambda_ff: float = 0.9999
    p0_rls: float = 0.03
    p0_kf_state_var: float = 10.0
    n_epochs: int = 30


def make_multisine_input(cfg: OldConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim port of the old script's make_multisine_input."""
    u_max, du_max, Ts = cfg.u_max, cfg.du_max, cfg.Ts
    f_max = du_max / (2 * np.pi * (u_max / 2) * Ts)
    freqs = np.array(cfg.multisine_freq_fracs) * f_max
    t = np.arange(cfg.N_sim) * Ts
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    raw = np.mean([np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)], axis=0)
    u = (u_max / 2) * (1 + raw)
    return u, freqs


def synthesize_ar_beta_fallback(rng: np.random.Generator, p: int,
                                 sigma2: float) -> tuple[np.ndarray, float]:
    """Verbatim port of the old script's synthesize_ar_beta_fallback."""
    dominant = rng.uniform(0.90, 0.95)
    n_pairs = (p - 1) // 2
    n_real = p - 1 - 2 * n_pairs
    poles = [dominant]
    for _ in range(n_pairs):
        r = rng.uniform(0.3, 0.7)
        th = rng.uniform(0.1, 1.5)
        poles += [r * np.exp(1j * th), r * np.exp(-1j * th)]
    for _ in range(n_real):
        poles.append(rng.uniform(0.2, 0.6))
    char = np.real(np.poly(np.array(poles)))
    phi_coeffs = -char[1:]
    return phi_coeffs, sigma2


def simulate_ar_beta(A_beta: np.ndarray, C_beta: np.ndarray, sigma_beta2: float,
                      N: int, rng: np.random.Generator) -> np.ndarray:
    p_beta = A_beta.shape[0]
    x_beta = np.zeros(p_beta)
    beta_sim = np.zeros(N)
    w = rng.normal(0.0, np.sqrt(sigma_beta2), size=N)
    for kk in range(N):
        beta_sim[kk] = (C_beta @ x_beta)[0]
        x_beta = A_beta @ x_beta
        x_beta[0] += w[kk]
    return beta_sim


def perturb_true_params_1p(k: float, tau: float, rng: np.random.Generator,
                            frac: float) -> tuple[float, float]:
    delta = rng.uniform(-frac, frac, size=2)
    return k * (1 + delta[0]), tau * (1 + delta[1])


def run_replication(cfg: OldConfig = OldConfig(), seed: int | None = None) -> dict:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    theta_true = params_to_theta_1p(cfg.k_true, cfg.tau_true, cfg.Ts)
    k0, tau0 = perturb_true_params_1p(cfg.k_true, cfg.tau_true, rng, cfg.perturb_frac)
    theta_hat0 = params_to_theta_1p(k0, tau0, cfg.Ts)

    phi_coeffs, sigma2_ar = synthesize_ar_beta_fallback(rng, cfg.p_ar_fallback, cfg.sigma2_ar_fallback)
    p = len(phi_coeffs)
    n = N_ETA_1P + p  # NO offset state -- matches the old script exactly

    u, freqs = make_multisine_input(cfg, rng)

    A_eta_t, B_eta_t, C_eta_t = theta_to_observer_canonical_1p(theta_true)
    eta_true = simulate_ss_1p(A_eta_t, B_eta_t, C_eta_t, u)

    A_beta, C_beta = ar_to_state_space(phi_coeffs)
    beta_sim = simulate_ar_beta(A_beta, C_beta, sigma2_ar, cfg.N_sim, rng)
    v = rng.normal(0.0, np.sqrt(cfg.R), size=cfg.N_sim)
    y_true = beta_sim - eta_true + v   # NO offset term -- matches the old script exactly

    Q_diag = np.concatenate((build_process_noise_diag(N_ETA_1P, p, cfg.q_eta_default,
                                                        cfg.q_beta_nominal, cfg.q_beta_other_default),))
    x0 = np.zeros(n)
    P0_kf = cfg.p0_kf_state_var * np.eye(n)
    P0_rls = cfg.p0_rls * np.eye(2)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    Q_c = np.ascontiguousarray(Q_diag, dtype=np.float64)

    (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, Pkf_checks, Prls_checks,
     theta_epoch_hist, theta_hat_by_epoch, elapsed, innov_final, S_final) = run_dual_estimator_1p(
        u, y_true, phi_c, Q_c, cfg.R, x0, P0_kf, theta_hat0, P0_rls, cfg.lambda_ff,
        N_ETA_1P, p, n_off=0, gamma_pole=0.0, a1_prior=0.0,
        n_epochs=cfg.n_epochs, n_snapshots=0, save_epochs=set())

    theta_final = theta_hat[-1]
    final_err = float(np.linalg.norm(theta_final - theta_true))
    try:
        k_hat, tau_hat = theta_to_params_1p(theta_final, cfg.Ts)
    except ValueError:
        k_hat, tau_hat = float("nan"), float("nan")

    return dict(theta_true=theta_true, theta_hat0=theta_hat0, theta_final=theta_final,
                final_err=final_err, k_hat=k_hat, tau_hat=tau_hat, elapsed=elapsed,
                k_true=cfg.k_true, tau_true=cfg.tau_true)


if __name__ == "__main__":
    cfg = OldConfig()
    print(f"true (k,tau) = ({cfg.k_true}, {cfg.tau_true}); fs={cfg.fs}Hz, N={cfg.N_sim}, "
          f"no offset state, fixed R/q_eta/q_beta constants (old-prototype replication)")
    n_seeds = 10
    n_ok = 0
    for seed in range(n_seeds):
        r = run_replication(cfg, seed=seed)
        k_err = abs(r["k_hat"] - r["k_true"]) / r["k_true"] if np.isfinite(r["k_hat"]) else float("nan")
        tau_err = abs(r["tau_hat"] - r["tau_true"]) / r["tau_true"] if np.isfinite(r["tau_hat"]) else float("nan")
        ok = np.isfinite(k_err) and k_err < 0.3 and tau_err < 0.3
        n_ok += int(ok)
        print(f"seed={seed}: theta_err={r['final_err']:.5f}  k_hat={r['k_hat']:.3f} "
              f"(true {r['k_true']})  tau_hat={r['tau_hat']:.4f} (true {r['tau_true']})  ok={ok}  "
              f"[{r['elapsed']*1e3:.1f}ms]")
    print(f"\nok-rate: {n_ok}/{n_seeds}")
