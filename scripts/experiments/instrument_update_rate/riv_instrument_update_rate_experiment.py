"""riv_instrument_update_rate_experiment.py -- does the RIV auxiliary-model
instrument (eta_aux) need to be refreshed from the online theta estimate
every sample, or can it be updated less often -- down to fully offline
(frozen at one fixed theta, precomputed once before the estimation loop
starts at all)?

Base scenario: the SAME faithful 2-pole port of the old prototype's
"converging well" setup confirmed in INVESTIGATION.md Entry 47/48 (6/6 seeds
ok) -- fs=50Hz (Ts=0.02), N_sim=80000, rate-limited multisine, no offset
state, fixed AR(12) beta fallback, fixed noise constants (R=0.1, q_eta=1e4,
q_beta_nominal=5.0, q_beta_other=0.1), init perturbed +-50% around the true
(k,tau1,tau2)=(62.11,0.05,0.25). Every one of those constants is held
IDENTICAL across every arm below -- only the instrument-update mechanism
changes (per-user direction: this estimator is already tuned; retuning noise
constants per-arm would confound "does update rate matter" with "did we
compensate for it elsewhere").

Reused, unmodified: params_to_theta / theta_to_params /
theta_to_observer_canonical / simulate_ss / ar_to_state_space /
build_process_noise_diag / _structured_kf_prop, all from this repo's own
riv_dual_estimation.py. make_multisine_input / synthesize_ar_beta_fallback /
simulate_ar_beta reused unmodified from riv_single_pole_50hz_replication_
check.py (already a validated port of the same old-prototype scenario, and
pole-count-agnostic). Ported (not modified in place) from the old prototype
(".../RIV Kalman filter/riv_dual_estimation.py", read-only, per
MPC_linear/CLAUDE.md): perturb_true_params, theta_to_discrete_ss,
h2_model_error, and six plotting functions (plot_input, plot_output_fit,
plot_eta_fit, plot_param_convergence, plot_theta_epochs_stacked,
plot_theta_h2_epochs_stacked) -- so every arm below gets the SAME diagnostic
plot suite the original "converging well" run produced, per user direction,
not just a summary table.

Instrument-update arms (one dial M = instrument refresh period in samples,
plus an oracle check):
  adaptive          M=1    (baseline, current production behaviour)
  decimated_M{k}    M=k    (aux-model recursion still runs every sample, but
                             the theta values it uses are a snapshot
                             refreshed only every M samples)
  decimated_Mfull   M=N    (snapshot refreshed once, at t=0, then never
                             again -- cross-checked against frozen_init)
  frozen_init              (aux model fixed at the perturbed INITIAL guess
                             theta_hat0 for the whole run, eta_aux
                             precomputed once, vectorized, before the KF/RLS
                             loop starts -- the literal "fully offline" arm)
  frozen_true               (same, but fixed at the TRUE theta -- oracle
                             upper bound, isolates "offline is bad" from
                             "this offline guess was bad")
  lagged_u                  (classic delayed-input instrument, Entry 12 --
                             zero plant-knowledge anchor, already implemented
                             in riv_dual_estimation.py's dense reference)

Run: python3 scripts/riv_instrument_update_rate_experiment.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import control
import numba
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/single_pole"))
from riv_dual_estimation import (
    params_to_theta, theta_to_params, theta_to_observer_canonical,
    simulate_ss, ar_to_state_space, build_process_noise_diag,
    _structured_kf_prop,
)
from riv_single_pole_50hz_replication_check import (
    make_multisine_input, synthesize_ar_beta_fallback, simulate_ar_beta,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "instrument_update_rate"

AUX_DECIMATED = 0
AUX_EXTERNAL = 1
LAGGED_U = 2

DECIMATION_SWEEP = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
N_SEEDS = 10


@dataclass(frozen=True)
class Config:
    fs: float = 50.0
    Ts: float = 0.02
    k_true: float = 62.11
    tau1_true: float = 0.05
    tau2_true: float = 0.25
    perturb_frac: float = 0.5
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


# ==========================================================================
# Ported verbatim from the old prototype (read-only source, Entry 47/48)
# ==========================================================================

def perturb_true_params(k: float, tau1: float, tau2: float, rng: np.random.Generator,
                         frac: float) -> tuple[float, float, float]:
    delta = rng.uniform(-frac, frac, size=3)
    return k * (1 + delta[0]), tau1 * (1 + delta[1]), tau2 * (1 + delta[2])


def theta_to_discrete_ss(theta: np.ndarray, Ts: float) -> control.StateSpace:
    A_eta, B_eta, C_eta = theta_to_observer_canonical(theta)
    return control.ss(A_eta, B_eta, C_eta, 0.0, dt=Ts)


def h2_model_error(theta_true: np.ndarray, theta_hat: np.ndarray, Ts: float) -> float:
    sys_err = theta_to_discrete_ss(theta_true, Ts) - theta_to_discrete_ss(theta_hat, Ts)
    try:
        val = control.norm(sys_err, p=2)
        if np.isfinite(val):
            return float(val)
    except Exception:
        pass
    _, g = control.impulse_response(sys_err, T=np.arange(2000) * Ts)
    return float(np.sqrt(np.sum(np.asarray(g).squeeze() ** 2)))


def plot_input(t: np.ndarray, u: np.ndarray, cfg: Config, path: Path,
                title: str = "Rate-limited multisine input") -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, u, lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(cfg.u_max, color="r", ls="--", lw=0.8, label=f"u_max={cfg.u_max}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("u"); ax.set_title(title)
    ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_output_fit(t: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, y_true, label="y_true", lw=1)
    ax.plot(t, y_hat, label="y_hat (dual KF)", lw=1, ls="--")
    ax.set_xlabel("time (s)"); ax.set_ylabel("y = beta - eta"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_eta_fit(t: np.ndarray, eta_true: np.ndarray, eta_hat: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, eta_true, label="eta_true", lw=1)
    ax.plot(t, eta_hat, label="eta_hat (dual KF)", lw=1, ls="--")
    ax.set_xlabel("time (s)"); ax.set_ylabel("eta"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_param_convergence(t: np.ndarray, theta_traj: np.ndarray, theta_true: np.ndarray,
                            h2_t: np.ndarray, h2_err: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    labels = ["a1", "a2", "b1", "b2"]
    for i, lab in enumerate(labels):
        axes[0].plot(t, theta_traj[:, i], label=f"{lab}_hat", lw=1)
        axes[0].axhline(theta_true[i], color=axes[0].lines[-1].get_color(), ls=":", lw=1)
    axes[0].set_ylabel("theta_hat_k"); axes[0].set_title("Parameter convergence (dotted = true)")
    axes[0].legend(ncol=4, fontsize=8)

    err_norm = np.linalg.norm(theta_traj - theta_true[None, :], axis=1)
    axes[1].semilogy(t, err_norm, color="tab:blue", label="||theta_hat_k - theta_true||")
    ax2 = axes[1].twinx()
    ax2.semilogy(h2_t, h2_err, color="tab:red", marker="o", ms=3, lw=1,
                 label="||G_true-G_hat||_H2")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("||theta err||", color="tab:blue")
    ax2.set_ylabel("H2 model error", color="tab:red")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_theta_epochs_stacked(t: np.ndarray, epochs: list[int],
                               theta_hat_by_epoch: dict[int, np.ndarray],
                               theta_true: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(len(epochs), 1, figsize=(8, 2.6 * len(epochs)), sharex=True)
    labels = ["a1", "a2", "b1", "b2"]
    for row, epoch in enumerate(epochs):
        ax = axes[row]
        traj = theta_hat_by_epoch[epoch]
        for i, lab in enumerate(labels):
            ax.plot(t, traj[:, i], label=f"{lab}_hat", lw=1)
            ax.axhline(theta_true[i], color=ax.lines[-1].get_color(), ls=":", lw=1)
        ax.set_ylabel("theta_hat_k")
        ax.set_title(f"epoch {epoch}")
        if row == 0:
            ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Parameter trajectories across selected epochs (dotted = true)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_theta_h2_epochs_stacked(t: np.ndarray, epochs: list[int],
                                  theta_hat_by_epoch: dict[int, np.ndarray], theta_true: np.ndarray,
                                  h2_t: np.ndarray, h2_err_by_epoch: dict[int, np.ndarray],
                                  path: Path) -> None:
    fig, axes = plt.subplots(len(epochs), 1, figsize=(8, 2.6 * len(epochs)), sharex=True)
    for row, epoch in enumerate(epochs):
        ax = axes[row]
        traj = theta_hat_by_epoch[epoch]
        err_norm = np.linalg.norm(traj - theta_true[None, :], axis=1)
        ax.semilogy(t, err_norm, color="tab:blue", label="||theta_hat_k - theta_true||")
        ax2 = ax.twinx()
        ax2.semilogy(h2_t, h2_err_by_epoch[epoch], color="tab:red", marker="o", ms=3, lw=1,
                     label="||G_true-G_hat||_H2")
        ax.set_ylabel("||theta err||", color="tab:blue")
        ax2.set_ylabel("H2 error", color="tab:red")
        ax.set_title(f"epoch {epoch}")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Error convergence across selected epochs")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ==========================================================================
# Instrument-aware jitted core -- copy of the old prototype's
# _dual_estimator_core (no clamp/no offset state, matching Entry 47's
# faithful replication) with the z-update block replaced by a mode switch.
# ==========================================================================

@numba.njit(cache=True)
def _dual_estimator_core_instrument(u, y, phi_ar, Q_diag, R, x0, P0_kf, theta0, P0_rls, lam,
                                     n_eta, p, instrument_mode, M, eta_aux_full,
                                     cov_stride, Pkf_checks, Prls_checks):
    idx_beta = n_eta
    n = n_eta + p
    N = y.shape[0]

    x = x0.copy()
    Pkf = P0_kf.copy()
    theta = theta0.copy()
    Prls = P0_rls.copy()
    z1 = 0.0
    z2 = 0.0
    theta_aux0, theta_aux1, theta_aux2, theta_aux3 = theta0[0], theta0[1], theta0[2], theta0[3]

    x_hat = np.zeros((N, n))
    theta_hat = np.zeros((N, 4))
    y_hat = np.zeros(N)

    x_new = np.empty(n)
    temp = np.empty((n, n))
    Pminus = np.empty((n, n))
    HP = np.empty(n)
    Kkf = np.empty(n)
    phi_vec = np.empty(4)
    z_vec = np.empty(4)
    Pz = np.empty(4)
    phiP = np.empty(4)
    Krls = np.empty(4)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a1 = theta[0]; a2 = theta[1]; b1 = theta[2]; b2 = theta[3]
            eta1 = x[0]; eta2 = x[1]

            x_new[0] = a1 * eta1 + eta2 + b1 * u_prev
            x_new[1] = a2 * eta1 + b2 * u_prev
            s = 0.0
            for l in range(p):
                s += phi_ar[l] * x[idx_beta + l]
            x_new[idx_beta] = s
            for m in range(1, p):
                x_new[idx_beta + m] = x[idx_beta + m - 1]
            for i in range(n):
                x[i] = x_new[i]

            _structured_kf_prop(Pkf, Q_diag, phi_ar, a1, a2, n_eta, p, 0, temp, Pminus)
            for i in range(n):
                for j in range(n):
                    Pkf[i, j] = Pminus[i, j]

        h_pred = x[idx_beta] - x[0]
        innov = y[kk] - h_pred

        for j in range(n):
            HP[j] = -Pkf[0, j] + Pkf[idx_beta, j]
        S = -HP[0] + HP[idx_beta] + R
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
        y_hat[kk] = x[idx_beta] - x[0]

        eta_km1 = x_hat[kk - 1, 0] if kk - 1 >= 0 else 0.0
        eta_km2 = x_hat[kk - 2, 0] if kk - 2 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        u_km2 = u[kk - 2] if kk - 2 >= 0 else 0.0
        phi_vec[0] = eta_km1
        phi_vec[1] = eta_km2
        phi_vec[2] = u_km1
        phi_vec[3] = u_km2

        if instrument_mode == LAGGED_U:
            u_km3 = u[kk - 3] if kk - 3 >= 0 else 0.0
            u_km4 = u[kk - 4] if kk - 4 >= 0 else 0.0
            z_vec[0] = u_km3
            z_vec[1] = u_km4
            z_vec[2] = u_km1
            z_vec[3] = u_km2
        elif instrument_mode == AUX_EXTERNAL:
            ze1 = eta_aux_full[kk - 1] if kk - 1 >= 0 else 0.0
            ze2 = eta_aux_full[kk - 2] if kk - 2 >= 0 else 0.0
            z_vec[0] = ze1
            z_vec[1] = ze2
            z_vec[2] = u_km1
            z_vec[3] = u_km2
        else:  # AUX_DECIMATED (M=1 reduces exactly to the adaptive baseline)
            z_vec[0] = z1
            z_vec[1] = z2
            z_vec[2] = u_km1
            z_vec[3] = u_km2

        for i in range(4):
            s3 = 0.0
            for j in range(4):
                s3 += Prls[i, j] * z_vec[j]
            Pz[i] = s3
        denom = lam
        for i in range(4):
            denom += phi_vec[i] * Pz[i]
        for i in range(4):
            Krls[i] = Pz[i] / denom

        pred = 0.0
        for i in range(4):
            pred += phi_vec[i] * theta[i]
        err = eta_hat_k - pred

        if instrument_mode == AUX_DECIMATED:
            if kk % M == 0:
                theta_aux0, theta_aux1, theta_aux2, theta_aux3 = theta[0], theta[1], theta[2], theta[3]
            z_new = theta_aux0 * z1 + theta_aux1 * z2 + theta_aux2 * u_km1 + theta_aux3 * u_km2
            z2 = z1
            z1 = z_new

        for i in range(4):
            theta[i] = theta[i] + Krls[i] * err
        for i in range(4):
            s4 = 0.0
            for j in range(4):
                s4 += phi_vec[j] * Prls[j, i]
            phiP[i] = s4
        for i in range(4):
            for j in range(4):
                Prls[i, j] = (Prls[i, j] - Krls[i] * phiP[j]) / lam

        for i in range(4):
            theta_hat[kk, i] = theta[i]

        if cov_stride > 0 and kk % cov_stride == 0:
            idx_chk = kk // cov_stride
            if idx_chk < Pkf_checks.shape[0]:
                for i in range(n):
                    for j in range(n):
                        Pkf_checks[idx_chk, i, j] = Pkf[i, j]
                for i in range(4):
                    for j in range(4):
                        Prls_checks[idx_chk, i, j] = Prls[i, j]

    return x_hat, theta_hat, y_hat, Pkf, Prls


def run_dual_estimator_instrument(u, y, phi_ar, Q_diag, R, x0, P0_kf, theta0, P0_rls, lam,
                                   n_eta, p, instrument_mode, M, eta_aux_full,
                                   n_epochs=1, n_snapshots=0, save_epochs=None):
    """Epoch-loop wrapper, same reset-each-epoch convention as this repo's
    own run_dual_estimator (INVESTIGATION.md Entry 22: reset beats
    carry-forward)."""
    n = n_eta + p
    N = len(y)
    cov_stride = max(1, N // n_snapshots) if n_snapshots > 0 else 0
    n_checks = (N // cov_stride + 1) if cov_stride > 0 else 1
    Pkf_checks = np.zeros((n_checks, n, n))
    Prls_checks = np.zeros((n_checks, 4, 4))
    save_epochs = save_epochs or set()

    theta_cur = np.ascontiguousarray(theta0, dtype=np.float64).copy()
    P0_kf_c = np.ascontiguousarray(P0_kf, dtype=np.float64)
    P0_rls_c = np.ascontiguousarray(P0_rls, dtype=np.float64)
    theta_epoch_hist = np.zeros((n_epochs, 4))
    theta_hat_by_epoch: dict[int, np.ndarray] = {}

    x_hat = theta_hat = y_hat = Pkf_final = Prls_final = None
    for epoch in range(n_epochs):
        x0_epoch = np.ascontiguousarray(x0, dtype=np.float64).copy()
        x_hat, theta_hat, y_hat, Pkf_final, Prls_final = _dual_estimator_core_instrument(
            u, y, phi_ar, Q_diag, float(R), x0_epoch, P0_kf_c.copy(), theta_cur, P0_rls_c.copy(),
            float(lam), n_eta, p, instrument_mode, M, eta_aux_full,
            cov_stride, Pkf_checks, Prls_checks)
        theta_cur = theta_hat[-1].copy()
        theta_epoch_hist[epoch] = theta_cur
        if epoch in save_epochs:
            theta_hat_by_epoch[epoch] = theta_hat.copy()

    return (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, theta_epoch_hist, theta_hat_by_epoch)


# ==========================================================================
# Scenario generation (Entry 47's exact setup) + arms
# ==========================================================================

def make_scenario(cfg: Config, rng: np.random.Generator) -> dict:
    theta_true = params_to_theta(cfg.k_true, cfg.tau1_true, cfg.tau2_true, cfg.Ts)
    k0, tau1_0, tau2_0 = perturb_true_params(cfg.k_true, cfg.tau1_true, cfg.tau2_true,
                                              rng, cfg.perturb_frac)
    theta_hat0 = params_to_theta(k0, tau1_0, tau2_0, cfg.Ts)

    phi_coeffs, sigma2_ar = synthesize_ar_beta_fallback(rng, cfg.p_ar_fallback, cfg.sigma2_ar_fallback)
    p = len(phi_coeffs)
    n = 2 + p

    u, freqs = make_multisine_input(cfg, rng)

    A_eta_t, B_eta_t, C_eta_t = theta_to_observer_canonical(theta_true)
    eta_true = simulate_ss(A_eta_t, B_eta_t, C_eta_t, u)

    A_beta, C_beta = ar_to_state_space(phi_coeffs)
    beta_sim = simulate_ar_beta(A_beta, C_beta, sigma2_ar, cfg.N_sim, rng)
    v = rng.normal(0.0, np.sqrt(cfg.R), size=cfg.N_sim)
    y_true = beta_sim - eta_true + v  # NO offset term -- matches Entry 47 exactly

    Q_diag = build_process_noise_diag(2, p, cfg.q_eta_default, cfg.q_beta_nominal,
                                       cfg.q_beta_other_default)
    x0 = np.zeros(n)
    P0_kf = cfg.p0_kf_state_var * np.eye(n)
    P0_rls = cfg.p0_rls * np.eye(4)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    Q_c = np.ascontiguousarray(Q_diag, dtype=np.float64)

    return dict(theta_true=theta_true, theta_hat0=theta_hat0, phi_c=phi_c, Q_c=Q_c,
                u=u, y_true=y_true, eta_true=eta_true, x0=x0, P0_kf=P0_kf, P0_rls=P0_rls, p=p)


def arm_eta_aux_full(theta_fixed: np.ndarray, u: np.ndarray) -> np.ndarray:
    A, B, C = theta_to_observer_canonical(theta_fixed)
    return simulate_ss(A, B, C, u)


def build_arms() -> list[dict]:
    arms = []
    for M in DECIMATION_SWEEP:
        name = "adaptive" if M == 1 else f"decimated_M{M:04d}"
        arms.append(dict(name=name, instrument_mode=AUX_DECIMATED, M=M, theta_source=None))
    arms.append(dict(name="decimated_Mfull", instrument_mode=AUX_DECIMATED, M=None, theta_source=None))
    arms.append(dict(name="frozen_init", instrument_mode=AUX_EXTERNAL, M=1, theta_source="init"))
    arms.append(dict(name="frozen_true", instrument_mode=AUX_EXTERNAL, M=1, theta_source="true"))
    arms.append(dict(name="lagged_u", instrument_mode=LAGGED_U, M=1, theta_source=None))
    return arms


def run_arm_for_seed(arm: dict, cfg: Config, seed: int, save_epochs: set[int] | None = None):
    rng = np.random.default_rng(seed)
    scn = make_scenario(cfg, rng)
    M = cfg.N_sim if arm["M"] is None else arm["M"]
    if arm["theta_source"] == "init":
        eta_aux_full = arm_eta_aux_full(scn["theta_hat0"], scn["u"])
    elif arm["theta_source"] == "true":
        eta_aux_full = arm_eta_aux_full(scn["theta_true"], scn["u"])
    else:
        eta_aux_full = np.zeros(cfg.N_sim)  # unused placeholder for this mode

    result = run_dual_estimator_instrument(
        scn["u"], scn["y_true"], scn["phi_c"], scn["Q_c"], cfg.R, scn["x0"], scn["P0_kf"],
        scn["theta_hat0"], scn["P0_rls"], cfg.lambda_ff, 2, scn["p"],
        arm["instrument_mode"], M, eta_aux_full,
        n_epochs=cfg.n_epochs, n_snapshots=20, save_epochs=save_epochs or set())
    return scn, result


# ==========================================================================
# Per-arm ok-rate sweep + detailed (plotted) single-seed run
# ==========================================================================

def sweep_arm(arm: dict, cfg: Config, n_seeds: int = N_SEEDS) -> list[dict]:
    rows = []
    for seed in range(n_seeds):
        scn, (x_hat, theta_hat, y_hat, Pkf_final, Prls_final,
              theta_epoch_hist, _) = run_arm_for_seed(arm, cfg, seed)
        theta_final = theta_hat[-1]
        theta_true = scn["theta_true"]
        final_err = float(np.linalg.norm(theta_final - theta_true))
        try:
            k_hat, tau1_hat, tau2_hat = theta_to_params(theta_final, cfg.Ts)
        except ValueError:
            k_hat, tau1_hat, tau2_hat = float("nan"), float("nan"), float("nan")
        k_err = abs(k_hat - cfg.k_true) / cfg.k_true if np.isfinite(k_hat) else float("nan")
        tau1_err = abs(tau1_hat - cfg.tau1_true) / cfg.tau1_true if np.isfinite(tau1_hat) else float("nan")
        tau2_err = abs(tau2_hat - cfg.tau2_true) / cfg.tau2_true if np.isfinite(tau2_hat) else float("nan")
        ok = bool(np.isfinite(k_err) and k_err < 0.3 and tau1_err < 0.3 and tau2_err < 0.3)
        min_eig_rls = float(np.linalg.eigvals(Prls_final).real.min())
        rows.append(dict(seed=seed, theta_err=final_err, k_hat=k_hat, tau1_hat=tau1_hat,
                          tau2_hat=tau2_hat, k_err=k_err, tau1_err=tau1_err, tau2_err=tau2_err,
                          ok=ok, min_eig_rls=min_eig_rls))
    return rows


def run_arm_detailed(arm: dict, cfg: Config, seed: int = 0) -> None:
    n_epochs = cfg.n_epochs
    selected_epochs = sorted(set(range(min(4, n_epochs))) | {n_epochs - 1})
    scn, (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, theta_epoch_hist,
          theta_hat_by_epoch) = run_arm_for_seed(arm, cfg, seed, save_epochs=set(selected_epochs))

    t = np.arange(cfg.N_sim) * cfg.Ts
    eta_hat = x_hat[:, 0]
    theta_true = scn["theta_true"]

    n_h2 = 150
    idxs = np.unique(np.linspace(0, cfg.N_sim - 1, n_h2).astype(int))
    h2_traj = np.array([h2_model_error(theta_true, theta_hat[i], cfg.Ts) for i in idxs])
    h2_traj_by_epoch = {
        epoch: np.array([h2_model_error(theta_true, theta_hat_by_epoch[epoch][i], cfg.Ts) for i in idxs])
        for epoch in selected_epochs
    }

    out = OUT_DIR / arm["name"]
    out.mkdir(parents=True, exist_ok=True)
    plot_input(t, scn["u"], cfg, out / "multisine.png",
               title=f"Rate-limited multisine input ({arm['name']})")
    plot_output_fit(t, scn["y_true"], y_hat, out / "output_fit.png")
    plot_eta_fit(t, scn["eta_true"], eta_hat, out / "eta_fit.png")
    plot_param_convergence(t, theta_hat, theta_true, t[idxs], h2_traj, out / "convergence.png")
    plot_theta_epochs_stacked(t, selected_epochs, theta_hat_by_epoch, theta_true,
                               out / "theta_epochs_stacked.png")
    plot_theta_h2_epochs_stacked(t, selected_epochs, theta_hat_by_epoch, theta_true,
                                  t[idxs], h2_traj_by_epoch, out / "theta_h2_epochs_stacked.png")

    final_err = float(np.linalg.norm(theta_hat[-1] - theta_true))
    print(f"  [{arm['name']}] seed=0 detailed run (plots -> {out}): theta_err={final_err:.5f}")


def plot_ok_rate_vs_M(summary: list[dict], path: Path) -> None:
    dec = [s for s in summary if s["is_decimated"]]
    dec.sort(key=lambda s: s["M_eff"])
    Ms = [s["M_eff"] for s in dec]
    ok_rates = [s["ok_rate"] for s in dec]
    med_err = [s["median_theta_err"] for s in dec]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].semilogx(Ms, ok_rates, marker="o")
    axes[0].set_ylabel(f"ok-rate ({N_SEEDS} seeds)"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Instrument update-rate sweep (aux-model, decimated)")
    axes[1].loglog(Ms, med_err, marker="o", color="tab:orange")
    axes[1].set_ylabel("median theta_err"); axes[1].set_xlabel("instrument refresh period M (samples)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> None:
    cfg = Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms()

    print(f"true (k,tau1,tau2)=({cfg.k_true},{cfg.tau1_true},{cfg.tau2_true}); "
          f"fs={cfg.fs}Hz N={cfg.N_sim}, no offset, fixed noise constants "
          f"(Entry 47 replication base, {len(arms)} instrument-update arms, {N_SEEDS} seeds each)")

    summary = []
    for arm in arms:
        print(f"\n=== arm: {arm['name']} ===")
        rows = sweep_arm(arm, cfg)
        n_ok = sum(r["ok"] for r in rows)
        med_err = float(np.median([r["theta_err"] for r in rows]))
        for r in rows:
            print(f"  seed={r['seed']}: theta_err={r['theta_err']:.5f} k_hat={r['k_hat']:.3f} "
                  f"tau1_hat={r['tau1_hat']:.4f} tau2_hat={r['tau2_hat']:.4f} "
                  f"min_eig_Prls={r['min_eig_rls']:.3e} ok={r['ok']}")
        print(f"  ok-rate: {n_ok}/{N_SEEDS}  median theta_err={med_err:.5f}")

        M_eff = cfg.N_sim if arm["M"] is None else arm["M"]
        summary.append(dict(name=arm["name"], M_eff=M_eff, ok_rate=n_ok / N_SEEDS,
                             median_theta_err=med_err,
                             is_decimated=(arm["instrument_mode"] == AUX_DECIMATED)))

        run_arm_detailed(arm, cfg, seed=0)

    plot_ok_rate_vs_M(summary, OUT_DIR / "ok_rate_vs_M.png")

    # Cross-check: decimated(M=N_sim) and frozen_init are two different code
    # paths (recursive-but-never-refreshed vs. precomputed-once-vectorized)
    # that should compute the SAME thing -- verify they agree.
    arm_dec_full = next(a for a in arms if a["name"] == "decimated_Mfull")
    arm_frozen_init = next(a for a in arms if a["name"] == "frozen_init")
    _, (_, theta_hat_a, *_rest_a) = run_arm_for_seed(arm_dec_full, cfg, seed=0)
    _, (_, theta_hat_b, *_rest_b) = run_arm_for_seed(arm_frozen_init, cfg, seed=0)
    max_diff = float(np.max(np.abs(theta_hat_a - theta_hat_b)))
    print(f"\ncross-check decimated(M=N_sim) vs frozen_init (seed=0): "
          f"max|theta_hat diff| = {max_diff:.3e}")

    print(f"\n{'arm':20s} {'M':>8s} {'ok-rate':>10s} {'median theta_err':>18s}")
    for s in summary:
        print(f"{s['name']:20s} {s['M_eff']:8d} {s['ok_rate']:10.2f} {s['median_theta_err']:18.5f}")


if __name__ == "__main__":
    main()
