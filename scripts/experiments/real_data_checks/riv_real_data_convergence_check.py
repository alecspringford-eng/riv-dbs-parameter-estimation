
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import control
import numba
import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import (
    CFG, N_ETA, AVG_PATIENT_PARAMS,
    params_to_theta, theta_to_observer_canonical, simulate_arx_recursion,
    ar_to_state_space, load_all_recordings, fit_ar_beta_model_pooled,
    build_process_noise_diag,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "real_data_convergence_check"

# reference DualConfig defaults, ported verbatim (this repo's DUAL_CFG has diverged:
# p0_rls=0.3 not 0.03, plus an offset/clamping-era Q_diag scheme)
LAMBDA_FF = 0.9999
P0_RLS = 0.03
P0_KF_STATE_VAR = 10.0
N_EPOCHS = 30
H2_EVAL_POINTS = 150
PERTURB_FRAC = 150

# R / q_eta / q_beta_nominal / q_beta_other are computed in main() (see module
# docstring) and passed as parameters, since they depend on runtime values


# Perturbed initial guess (reference's perturb_true_params, verbatim)

def perturb_true_params(k: float, tau1: float, tau2: float, rng: np.random.Generator,
                         frac: float) -> tuple[float, float, float]:
    delta = rng.uniform(-frac, frac, size=3)
    return k * (1 + delta[0]), tau1 * (1 + delta[1]), tau2 * (1 + delta[2])


# H2 model error (reference's h2_model_error, verbatim)

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


def nll_innovations(innov: np.ndarray, S: np.ndarray) -> float:
    """Gaussian one-step-ahead prediction-error NLL, used to pick P0_B from the
    data rather than theta_true. Evaluated post-hoc on the dual estimator's own
    KF innovations at the converged theta, not with theta held fixed."""
    return float(0.5 * np.sum(innov ** 2 / S + np.log(2 * np.pi * S)))


def simulate_ar_beta(A_beta: np.ndarray, C_beta: np.ndarray, sigma2: float,
                      N: int, rng: np.random.Generator) -> np.ndarray:
    p_beta = A_beta.shape[0]
    x_beta = np.zeros(p_beta)
    beta_sim = np.zeros(N)
    w = rng.normal(0.0, np.sqrt(sigma2), size=N)
    for kk in range(N):
        beta_sim[kk] = (C_beta @ x_beta)[0]
        x_beta = A_beta @ x_beta
        x_beta[0] += w[kk]
    return beta_sim


# Bare KF+RIV core, a standalone port of the reference _structured_kf_prop /
# _dual_estimator_core / run_dual_estimator (no offset state, clamping or
# u-normalization), kept separate from riv_dual_estimation.py's core

@numba.njit(cache=True)
def _structured_kf_prop(P: np.ndarray, Q_diag: np.ndarray, phi_ar: np.ndarray,
                         a1: float, a2: float, n_eta: int, p: int,
                         temp: np.ndarray, Pminus: np.ndarray) -> None:
    idx_beta = n_eta
    n = n_eta + p

    for j in range(n):
        temp[0, j] = a1 * P[0, j] + P[1, j]
        temp[1, j] = a2 * P[0, j]
        s = 0.0
        for l in range(p):
            s += phi_ar[l] * P[idx_beta + l, j]
        temp[idx_beta, j] = s
        for m in range(1, p):
            temp[idx_beta + m, j] = P[idx_beta + m - 1, j]

    for i in range(n):
        Pminus[i, 0] = a1 * temp[i, 0] + temp[i, 1]
        Pminus[i, 1] = a2 * temp[i, 0]
        s2 = 0.0
        for l in range(p):
            s2 += phi_ar[l] * temp[i, idx_beta + l]
        Pminus[i, idx_beta] = s2
        for m in range(1, p):
            Pminus[i, idx_beta + m] = temp[i, idx_beta + m - 1]

    for i in range(n):
        Pminus[i, i] += Q_diag[i]

    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (Pminus[i, j] + Pminus[j, i])
            Pminus[i, j] = avg
            Pminus[j, i] = avg


@numba.njit(cache=True)
def _dual_estimator_core(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, Q_diag: np.ndarray,
                          R: float, x0: np.ndarray, P0_kf: np.ndarray, theta0: np.ndarray,
                          P0_rls: np.ndarray, lam: float, n_eta: int, p: int, cov_stride: int,
                          Pkf_checks: np.ndarray, Prls_checks: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    idx_beta = n_eta
    n = n_eta + p
    N = y.shape[0]

    x = x0.copy()
    Pkf = P0_kf.copy()
    theta = theta0.copy()
    Prls = P0_rls.copy()
    z1 = 0.0
    z2 = 0.0

    x_hat = np.zeros((N, n))
    theta_hat = np.zeros((N, 4))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

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
            a1 = theta[0]
            a2 = theta[1]
            b1 = theta[2]
            b2 = theta[3]
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
            for i in range(n):
                x[i] = x_new[i]

            _structured_kf_prop(Pkf, Q_diag, phi_ar, a1, a2, n_eta, p, temp, Pminus)
            for i in range(n):
                for j in range(n):
                    Pkf[i, j] = Pminus[i, j]

        h_pred = x[idx_beta] - x[0]
        innov = y[kk] - h_pred

        for j in range(n):
            HP[j] = -Pkf[0, j] + Pkf[idx_beta, j]
        S = -HP[0] + HP[idx_beta] + R
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
        y_hat[kk] = x[idx_beta] - x[0]

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

        z_new = theta[0] * z1 + theta[1] * z2 + theta[2] * u_km1 + theta[3] * u_km2
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

    return x_hat, theta_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def run_dual_estimator(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, Q_diag: np.ndarray,
                        R: float, x0: np.ndarray, P0_kf: np.ndarray, theta0: np.ndarray,
                        P0_rls: np.ndarray, lam: float, n_eta: int, p: int,
                        n_epochs: int = 1, n_snapshots: int = 0,
                        save_epochs: set[int] | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], float,
                                   np.ndarray, np.ndarray]:
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

    x_hat = theta_hat = y_hat = Pkf_final = Prls_final = innov_hist = S_hist = None
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        x0_epoch = np.ascontiguousarray(x0, dtype=np.float64) if epoch == 0 else np.zeros(n)
        x_hat, theta_hat, y_hat, Pkf_final, Prls_final, innov_hist, S_hist = _dual_estimator_core(
            u, y, phi_ar, Q_diag, float(R), x0_epoch, P0_kf_c.copy(), theta_cur, P0_rls_c.copy(),
            float(lam), n_eta, p, cov_stride, Pkf_checks, Prls_checks)
        theta_cur = theta_hat[-1].copy()
        theta_epoch_hist[epoch] = theta_cur
        if epoch in save_epochs:
            theta_hat_by_epoch[epoch] = theta_hat.copy()
    elapsed = time.perf_counter() - t0

    return (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, Pkf_checks, Prls_checks,
            theta_epoch_hist, theta_hat_by_epoch, elapsed, innov_hist, S_hist)


# Plotting suite

def plot_input(t, u, path, title="Real stimulation input") -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, u, lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("u"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_output_fit(t, y_true, y_hat, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, y_true, label="y_true (semisynthetic)", lw=1)
    ax.plot(t, y_hat, label="y_hat (dual KF)", lw=1, ls="--")
    ax.set_xlabel("time (s)"); ax.set_ylabel("y = beta_log_fluct - eta"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_eta_fit(t, eta_true, eta_hat, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, eta_true, label="eta_true", lw=1)
    ax.plot(t, eta_hat, label="eta_hat (dual KF)", lw=1, ls="--")
    ax.set_xlabel("time (s)"); ax.set_ylabel("eta"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_param_convergence(t, theta_traj, theta_true, h2_t, h2_err, path) -> None:
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


def plot_theta_epochs_stacked(t, epochs, theta_hat_by_epoch, theta_true, path) -> None:
    fig, axes = plt.subplots(len(epochs), 1, figsize=(8, 2.6 * len(epochs)), sharex=True)
    if len(epochs) == 1:
        axes = [axes]
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


def plot_theta_h2_epochs_stacked(t, epochs, theta_hat_by_epoch, theta_true,
                                  h2_t, h2_err_by_epoch, path) -> None:
    fig, axes = plt.subplots(len(epochs), 1, figsize=(8, 2.6 * len(epochs)), sharex=True)
    if len(epochs) == 1:
        axes = [axes]
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


# Semisynthetic generator + per-recording driver

def generate_semisynthetic(u_real: np.ndarray, phi_coeffs: np.ndarray, sigma2_ar: float,
                            theta_true: np.ndarray, rng: np.random.Generator, R: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y = beta_true - eta_true + v with no offset term, as in the reference's
    generate_true_data(idealised=False). beta_true is a zero-mean log-domain AR
    fluctuation, not exponentiated back to linear power."""
    N = len(u_real)
    eta_true = simulate_arx_recursion(theta_true, u_real)
    A_beta, C_beta = ar_to_state_space(phi_coeffs)
    beta_true = simulate_ar_beta(A_beta, C_beta, sigma2_ar, N, rng)
    y = beta_true - eta_true + rng.normal(0.0, np.sqrt(R), N)
    return y, eta_true, beta_true


def build_kf_inputs(phi_coeffs: np.ndarray, p0_b: float, q_eta: float,
                     q_beta_nominal: float, q_beta_other: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = len(phi_coeffs)
    Q_diag = build_process_noise_diag(N_ETA, p, q_eta, q_beta_nominal, q_beta_other)
    x0 = np.zeros(N_ETA + p)
    P0_kf = P0_KF_STATE_VAR * np.eye(N_ETA + p)
    P0_rls = np.diag([P0_RLS, P0_RLS, p0_b, p0_b])
    return np.ascontiguousarray(Q_diag), x0, P0_kf, P0_rls


def calibrate_p0_b(fit_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                    names: list[str], R: float, q_eta: float, q_beta_nominal: float,
                    q_beta_other: float,
                    grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5,
                                                1.0, 2.0, 5.0)) -> float:
    print("\ncalibrating P0_B against innovation NLL (truth-blind, not theta_true)...")
    finite = np.zeros((len(fit_inputs), len(grid)), dtype=bool)
    nll = np.full((len(fit_inputs), len(grid)), np.nan)
    for gi, p0_b in enumerate(grid):
        for ri, (u_real, y, phi_coeffs, theta_hat0) in enumerate(fit_inputs):
            p = len(phi_coeffs)
            Q_c, x0, P0_kf, P0_rls = build_kf_inputs(phi_coeffs, p0_b, q_eta, q_beta_nominal,
                                                       q_beta_other)
            phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
            *_, innov_hist, S_hist = run_dual_estimator(
                u_real, y, phi_c, Q_c, R, x0, P0_kf, theta_hat0, P0_rls, LAMBDA_FF, N_ETA, p,
                n_epochs=N_EPOCHS, n_snapshots=0, save_epochs=set())
            if np.all(np.isfinite(innov_hist)) and np.all(S_hist > 0):
                finite[ri, gi] = True
                nll[ri, gi] = nll_innovations(innov_hist, S_hist)

    always_diverges = ~finite.any(axis=1)
    if always_diverges.any():
        bad_names = [names[i] for i in np.flatnonzero(always_diverges)]
        print(f"  excluded from calibration (diverge at every P0_B tested, not a tuning "
              f"issue -- check their perturbed initial guess for a sign-flipped tau): {bad_names}")
    included = ~always_diverges

    best_p0b, best_nll = None, np.inf
    for gi, p0_b in enumerate(grid):
        col_finite = finite[included, gi]
        if not col_finite.all():
            print(f"  P0_B={p0_b:5.2f}: some included recording diverged -- skipped")
            continue
        total_nll = float(np.sum(nll[included, gi]))
        print(f"  P0_B={p0_b:5.2f}: total innovation NLL = {total_nll:.2f}")
        if total_nll < best_nll:
            best_p0b, best_nll = p0_b, total_nll

    if best_p0b is None:
        raise RuntimeError("calibrate_p0_b: no grid value was finite for every included "
                            "recording -- widen the grid, this is not a case to silently fall back on")
    print(f"selected P0_B={best_p0b} (min total innovation NLL={best_nll:.2f})")
    return best_p0b


def run_semisynthetic_recording(name: str, u_real: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray,
                                 theta_true: np.ndarray, theta_hat0: np.ndarray, p0_b: float,
                                 R: float, q_eta: float, q_beta_nominal: float, q_beta_other: float,
                                 cfg, out_dir: Path) -> dict:
    print(f"\n=== {name} (semisynthetic, reference structure: real u, "
          f"log-domain real-fitted beta AR model, P0_B={p0_b} calibrated truth-blind) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    eta_true = simulate_arx_recursion(theta_true, u_real)
    p = len(phi_coeffs)
    Q_c, x0, P0_kf, P0_rls = build_kf_inputs(phi_coeffs, p0_b, q_eta, q_beta_nominal, q_beta_other)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)

    n_epochs = N_EPOCHS
    selected_epochs = sorted(set(range(min(4, n_epochs))) | {n_epochs - 1})

    (x_hat, theta_hat, y_hat, _, _, _, _, theta_epoch_hist,
     theta_hat_by_epoch, elapsed, _, _) = run_dual_estimator(
        u_real, y, phi_c, Q_c, R, x0, P0_kf, theta_hat0, P0_rls, LAMBDA_FF, N_ETA, p,
        n_epochs=n_epochs, n_snapshots=20, save_epochs=set(selected_epochs))

    # theta_true is used below only to evaluate the fit; it does not pick p0_b or
    # the reported epoch (decided by epoch count alone)
    epoch_errs = np.linalg.norm(theta_epoch_hist - theta_true[None, :], axis=1)
    print(f"epoch-to-epoch parameter error ({name}):")
    for epoch in np.unique(np.linspace(0, n_epochs - 1, min(6, n_epochs)).astype(int)):
        print(f"  epoch {epoch:>4d}: theta={np.array2string(theta_epoch_hist[epoch], precision=4)}"
              f"  err={epoch_errs[epoch]:.5f}")

    best_epoch = int(np.argmin(epoch_errs))
    if best_epoch != n_epochs - 1:
        print(f"  WARNING ({name}): error is NOT monotonically settled -- epoch {best_epoch} "
              f"(err={epoch_errs[best_epoch]:.5f}) is closer to truth than the final epoch "
              f"{n_epochs - 1} (err={epoch_errs[-1]:.5f}). Reporting the final epoch anyway "
              f"(that's what a blind run would deliver), not the best one.")

    eta_hat = x_hat[:, 0]
    theta_final = theta_hat[-1]
    final_err = float(np.linalg.norm(theta_final - theta_true))
    h2_final = h2_model_error(theta_true, theta_final, cfg.Ts)
    print(f"final parameter error ({name}): {final_err:.6f}")
    print(f"final H2 model error ({name}): {h2_final:.6f}")
    print(f"dual-estimator wall-clock time ({name}, {len(u_real)} samples): {elapsed * 1e3:.2f} ms")

    N = len(u_real)
    t = np.arange(N) * cfg.Ts
    n_h2 = min(H2_EVAL_POINTS, N)
    idxs = np.unique(np.linspace(0, N - 1, n_h2).astype(int))
    h2_traj = np.array([h2_model_error(theta_true, theta_hat[i], cfg.Ts) for i in idxs])

    h2_traj_by_epoch = {
        ep: np.array([h2_model_error(theta_true, theta_hat_by_epoch[ep][i], cfg.Ts) for i in idxs])
        for ep in selected_epochs
    }

    plot_input(t, u_real, out_dir / f"{name}_input.png", title=f"Real stimulation input ({name})")
    plot_output_fit(t, y, y_hat, out_dir / f"{name}_output_fit.png")
    plot_eta_fit(t, eta_true, eta_hat, out_dir / f"{name}_eta_fit.png")
    plot_param_convergence(t, theta_hat, theta_true, t[idxs], h2_traj,
                            out_dir / f"{name}_convergence.png")
    plot_theta_epochs_stacked(t, selected_epochs, theta_hat_by_epoch, theta_true,
                               out_dir / f"{name}_theta_epochs_stacked.png")
    plot_theta_h2_epochs_stacked(t, selected_epochs, theta_hat_by_epoch, theta_true,
                                  t[idxs], h2_traj_by_epoch,
                                  out_dir / f"{name}_theta_h2_epochs_stacked.png")

    return dict(final_err=final_err, h2_final=h2_final, theta_final=theta_final)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recordings = load_all_recordings(CFG)
    cfg_log = replace(CFG, ar_log_domain=True)  # local override
    # Pooled across every recording's no-stim periods: one AR model shared by all
    ar_order, phi_coeffs, sigma2_ar = fit_ar_beta_model_pooled(recordings, cfg_log)
    print(f"pooled AR({ar_order}) sigma2_ar={sigma2_ar:.4e}")

    # R / q_beta_nominal / q_beta_other use riv_dual_estimation.py's heuristic rather
    # than the reference's amplitude-mismatched constants. q_eta is set below, once
    # the semisynthetic y exists (0.5*var(y_all_system_id), pooled as in main())
    R = 0.05 * sigma2_ar
    q_beta_nominal = sigma2_ar
    q_beta_other = 0.1 * sigma2_ar
    print(f"data-driven R={R:.4e}, q_beta_nominal={q_beta_nominal:.4e}, "
          f"q_beta_other={q_beta_other:.4e} (was reference-verbatim R=0.1, "
          f"Q_BETA_NOMINAL=5.0, Q_BETA_OTHER_DEFAULT=0.1 -- see module docstring)")

    theta_true = params_to_theta(AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"],
                                  AVG_PATIENT_PARAMS["tau2"], CFG.Ts)
    print(f"theta_true = {np.array2string(theta_true, precision=5)}")

    # Generate each recording's semisynthetic y and initial guess once, so the P0_B
    # calibration and the final run see the same data
    per_recording = {}
    for idx, name in enumerate(CFG.system_id_files):
        u_real, _ = recordings[name]
        rng = np.random.default_rng(idx)

        k0, tau1_0, tau2_0 = perturb_true_params(
            AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"], AVG_PATIENT_PARAMS["tau2"],
            rng, PERTURB_FRAC)
        theta_hat0 = params_to_theta(k0, tau1_0, tau2_0, CFG.Ts)
        print(f"{name}: guess (k,tau1,tau2) = ({k0:.4f}, {tau1_0:.4f}, {tau2_0:.4f})")

        y, _, _ = generate_semisynthetic(u_real, phi_coeffs, sigma2_ar, theta_true, rng, R)
        per_recording[name] = (u_real, y, phi_coeffs, theta_hat0)

    q_eta = 0.5 * float(np.var(np.concatenate([per_recording[name][1]
                                                for name in CFG.system_id_files])))
    print(f"data-driven q_eta={q_eta:.4e} (pooled var(y), was reference-verbatim "
          f"Q_ETA_DEFAULT=1e4)")

    p0_b = calibrate_p0_b([per_recording[name] for name in CFG.system_id_files],
                          list(CFG.system_id_files), R, q_eta, q_beta_nominal, q_beta_other)

    results = {}
    for name in CFG.system_id_files:
        u_real, y, phi_coeffs, theta_hat0 = per_recording[name]
        stem = name.replace(".csv", "")
        results[name] = run_semisynthetic_recording(
            stem, u_real, y, phi_coeffs, theta_true, theta_hat0, p0_b,
            R, q_eta, q_beta_nominal, q_beta_other, CFG, OUT_DIR / stem)

    print(f"\n=== Summary (semisynthetic: known truth k={AVG_PATIENT_PARAMS['k']}, "
          f"tau1={AVG_PATIENT_PARAMS['tau1']}, tau2={AVG_PATIENT_PARAMS['tau2']}, "
          f"P0_B={p0_b} calibrated truth-blind via innovation NLL) ===")
    for name, r in results.items():
        print(f"{name}: final_err={r['final_err']:.4f}  H2_err={r['h2_final']:.4f}  "
              f"theta_final={np.array2string(r['theta_final'], precision=4)}")
    print(f"\nFigures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
