"""riv_mpc_param_export.py -- exports theta/logparam identification trajectories
against a known-truth synthetic plant and the 6 real recordings, for
MPC_linear's closed-loop parameterization comparison to consume.

No changes to riv_dual_estimation.py or riv_v1_logparam_jit.py -- this file
only calls their existing, unmodified public functions.

This module implements the synthetic ground-truth generator, the
theta-arm/logparam-arm estimator wrappers (Tasks 1-2), and the real-recordings
arm + full export driver (Task 3) of the export plan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import (
    CFG, DUAL_CFG, N_ETA, AVG_PATIENT_PARAMS,
    params_to_theta, theta_to_params_physical, build_kf_setup,
    run_dual_estimator, simulate_arx_recursion, ar_to_state_space,
    load_all_recordings, fit_ar_beta_model_pooled,
)
from riv_v1_logparam_jit import run_dual_estimator_logparam_jit

N_SYNTH = 3000
# 30, not the smaller values (4/6/10) used by some earlier one-off scripts in
# this repo -- matches this repo's own resolved seed-count decision
# (INVESTIGATION.md Entry 39: 6->30 to get a stable ok-rate estimate).
N_SEEDS = 30
OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "mpc_param_export"
# Disturbance (beta) truth used by the synthetic generator -- reproduces
# riv_v1_logparam_riv_experiment.py's synthetic_selfcheck exactly, truth
# parameterized instead of hardcoded.
PHI_AR_TRUE = np.array([0.6, -0.2, 0.1])
SIGMA2_TRUE = 0.05
C0_TRUE = 1.10


def generate_synthetic_iv(seed: int, k_true: float, tau1_true: float, tau2_true: float,
                           N: int = N_SYNTH) -> tuple[np.ndarray, np.ndarray]:
    """Known-truth synthetic (u, y) pair: PRBS-like uniform u, ARX-simulated
    eta from the truth plant, AR(3) beta random walk (fixed disturbance
    truth). Same generator as synthetic_selfcheck in
    riv_v1_logparam_riv_experiment.py, parameterized by (k_true, tau1_true,
    tau2_true) instead of hardcoded, so it can also be run at
    AVG_PATIENT_PARAMS."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.004, 0.026, size=N)
    eta_true = simulate_arx_recursion(params_to_theta(k_true, tau1_true, tau2_true, CFG.Ts), u)

    beta_transition, beta_output = ar_to_state_space(PHI_AR_TRUE)
    beta_state = np.zeros(len(PHI_AR_TRUE))
    beta_true = np.zeros(N)
    for i in range(N):
        beta_true[i] = (beta_output @ beta_state)[0]
        beta_state = beta_transition @ beta_state
        beta_state[0] += rng.normal(0, np.sqrt(SIGMA2_TRUE))

    y = C0_TRUE + beta_true - eta_true + rng.normal(0, 0.05, N)
    return u, y


def run_theta_arm(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, sigma2_ar: float,
                   q_eta: float, cfg, dual_cfg, n_epochs_online: int = 1) -> dict:
    """Runs the theta=[a1,a2,b1,b2] (ARX) dual estimator two ways on the same
    (u, y): a full-convergence pass (dual_cfg.n_epochs epochs) for a single
    "final" estimate, and a single causal pass (n_epochs_online, default 1)
    to get the per-sample online (k, tau1, tau2) trajectory. Both go through
    theta_to_params_physical to undo the u-normalization build_kf_setup
    applies -- theta_hat/theta_epoch_hist from run_dual_estimator are in
    u-normalized space.

    phi_ar/sigma2_ar/q_eta are the disturbance-model AR coefficients,
    innovation variance, and eta process-noise scale to use -- the synthetic
    arm passes the known truth (PHI_AR_TRUE, SIGMA2_TRUE, 0.5*var(y)), the
    real-recordings arm passes each recording's own fitted AR model."""
    R = 0.05 * sigma2_ar

    setup = build_kf_setup(u, y, phi_ar, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale, theta0 = setup["u_norm"], setup["u_scale"], setup["theta0"]
    Q_c, x0, P0_kf, P0_rls = setup["Q_diag"], setup["x0"], setup["P0_kf"], setup["P0_rls"]
    phi_c, a1_prior, a2_prior = setup["phi_c"], setup["a1_prior"], setup["a2_prior"]
    n_off, p = setup["n_off"], setup["p"]

    def run_pass(n_epochs):
        _, theta_hat, *_ = run_dual_estimator(
            u_norm, y, phi_c, Q_c, R, x0, P0_kf, theta0, P0_rls, dual_cfg.lambda_ff,
            N_ETA, p, n_off=n_off, gamma_pole=dual_cfg.gamma_pole,
            a1_prior=a1_prior, a2_prior=a2_prior, n_epochs=n_epochs,
            n_snapshots=0, save_epochs=set(),
        )
        return theta_hat

    # Once-per-episode: full n_epochs convergence (dual_cfg.n_epochs, default 30).
    theta_hat_conv = run_pass(dual_cfg.n_epochs)
    k_f, tau1_f, tau2_f = theta_to_params_physical(theta_hat_conv[-1], u_scale, cfg.Ts)

    # Online: single causal pass (n_epochs=1), per-sample trajectory.
    theta_hat_online = run_pass(n_epochs_online)
    N = len(u)
    k_on = np.empty(N)
    tau1_on = np.empty(N)
    tau2_on = np.empty(N)
    for i in range(N):
        k_on[i], tau1_on[i], tau2_on[i] = theta_to_params_physical(theta_hat_online[i], u_scale, cfg.Ts)

    ok = bool(k_f > 0 and abs(k_f - AVG_PATIENT_PARAMS["k"]) / AVG_PATIENT_PARAMS["k"] < 0.3)
    return dict(
        k_final=k_f, tau1_final=tau1_f, tau2_final=tau2_f, ok=ok,
        k_online=k_on, tau1_online=tau1_on, tau2_online=tau2_on,
    )


def run_logparam_arm(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, sigma2_ar: float,
                      q_eta: float, cfg, dual_cfg, n_epochs_online: int = 1) -> dict:
    """Runs the logparam=[log k, log tau1, log tau2] dual estimator two ways
    on the same (u, y): a full-convergence pass (dual_cfg.n_epochs epochs)
    for a single "final" estimate, and a single causal pass (n_epochs_online,
    default 1) to get the per-sample online (k, tau1, tau2) trajectory.
    Unlike theta's path, logp_hat is already in physical (Ts-independent)
    log-space -- np.exp() is the only conversion needed, no u-scale
    correction (build_kf_setup's u-normalization only affects the ARX
    regression the logparam Jacobian is linearized around, not the reported
    parameters). P0_rls here is 3x3 (log k, log tau1, log tau2), not
    build_kf_setup's 4x4 theta-space P0_rls -- mirrors
    riv_v1_logparam_jit.run_identification_logparam_jit's own setup.

    phi_ar/sigma2_ar/q_eta are the disturbance-model AR coefficients,
    innovation variance, and eta process-noise scale to use -- the synthetic
    arm passes the known truth (PHI_AR_TRUE, SIGMA2_TRUE, 0.5*var(y)), the
    real-recordings arm passes each recording's own fitted AR model."""
    R = 0.05 * sigma2_ar

    setup = build_kf_setup(u, y, phi_ar, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale = setup["u_norm"], setup["u_scale"]
    Q_c, x0, P0_kf = setup["Q_diag"], setup["x0"], setup["P0_kf"]
    phi_c, n_off, p = setup["phi_c"], setup["n_off"], setup["p"]

    logp0 = np.log(np.array([
        AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"], AVG_PATIENT_PARAMS["tau2"],
    ]))
    P0_rls = dual_cfg.p0_rls * np.eye(3)

    def run_pass(n_epochs):
        _, logp_hat, *_ = run_dual_estimator_logparam_jit(
            u_norm, y, phi_c, Q_c, R, x0, P0_kf, logp0, P0_rls, dual_cfg.lambda_ff,
            p, n_off, cfg.Ts, u_scale, n_epochs,
        )
        return logp_hat

    # Once-per-episode: full n_epochs convergence (dual_cfg.n_epochs, default 30).
    logp_hat_conv = run_pass(dual_cfg.n_epochs)
    logp_final = logp_hat_conv[-1]
    k_f = float(np.exp(logp_final[0]))
    tau1_f = float(np.exp(logp_final[1]))
    tau2_f = float(np.exp(logp_final[2]))

    # Online: single causal pass (n_epochs=1), per-sample trajectory.
    logp_hat_online = run_pass(n_epochs_online)
    k_on = np.exp(logp_hat_online[:, 0])
    tau1_on = np.exp(logp_hat_online[:, 1])
    tau2_on = np.exp(logp_hat_online[:, 2])

    ok = bool(k_f > 0 and abs(k_f - AVG_PATIENT_PARAMS["k"]) / AVG_PATIENT_PARAMS["k"] < 0.3)
    return dict(
        k_final=k_f, tau1_final=tau1_f, tau2_final=tau2_f, ok=ok,
        k_online=k_on, tau1_online=tau1_on, tau2_online=tau2_on,
    )


def export_synthetic(n_seeds: int = N_SEEDS, N: int = N_SYNTH, out_dir: Path = OUT_DIR) -> Path:
    """Runs both estimator arms on n_seeds known-truth synthetic (u, y)
    realizations (fixed AVG_PATIENT_PARAMS truth, PHI_AR_TRUE/SIGMA2_TRUE
    disturbance truth), and saves per-seed final + online-trajectory
    estimates to out_dir/synthetic_params.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k_true = AVG_PATIENT_PARAMS["k"]
    tau1_true = AVG_PATIENT_PARAMS["tau1"]
    tau2_true = AVG_PATIENT_PARAMS["tau2"]

    theta_runs = []
    logp_runs = []
    for seed in range(n_seeds):
        u, y = generate_synthetic_iv(seed, k_true, tau1_true, tau2_true, N=N)
        q_eta = 0.5 * float(np.var(y))
        th = run_theta_arm(u, y, PHI_AR_TRUE, SIGMA2_TRUE, q_eta, CFG, DUAL_CFG)
        lp = run_logparam_arm(u, y, PHI_AR_TRUE, SIGMA2_TRUE, q_eta, CFG, DUAL_CFG)
        print(f"seed={seed}: theta k={th['k_final']:.2f} ok={th['ok']}  "
              f"logparam k={lp['k_final']:.2f} ok={lp['ok']}")
        theta_runs.append(th)
        logp_runs.append(lp)

    theta_ok = [r["ok"] for r in theta_runs]
    logp_ok = [r["ok"] for r in logp_runs]

    path = out_dir / "synthetic_params.npz"
    np.savez(
        path,
        truth_k=k_true, truth_tau1=tau1_true, truth_tau2=tau2_true,
        theta_k_final=np.array([r["k_final"] for r in theta_runs]),
        theta_tau1_final=np.array([r["tau1_final"] for r in theta_runs]),
        theta_tau2_final=np.array([r["tau2_final"] for r in theta_runs]),
        theta_ok=np.array(theta_ok),
        theta_k_online=np.array([r["k_online"] for r in theta_runs]),
        theta_tau1_online=np.array([r["tau1_online"] for r in theta_runs]),
        theta_tau2_online=np.array([r["tau2_online"] for r in theta_runs]),
        logparam_k_final=np.array([r["k_final"] for r in logp_runs]),
        logparam_tau1_final=np.array([r["tau1_final"] for r in logp_runs]),
        logparam_tau2_final=np.array([r["tau2_final"] for r in logp_runs]),
        logparam_ok=np.array(logp_ok),
        logparam_k_online=np.array([r["k_online"] for r in logp_runs]),
        logparam_tau1_online=np.array([r["tau1_online"] for r in logp_runs]),
        logparam_tau2_online=np.array([r["tau2_online"] for r in logp_runs]),
    )
    print(f"theta ok-rate: {sum(theta_ok)}/{n_seeds}   logparam ok-rate: {sum(logp_ok)}/{n_seeds}")
    print(f"Saved -> {path}")
    return path


def export_real(out_dir: Path = OUT_DIR) -> Path:
    """Runs both estimator arms once (full-convergence only -- no online
    trajectory) on each of the 6 real system_id recordings, each against the
    SAME pooled AR beta disturbance model (fit across every recording's
    no-stimulation periods together, not the synthetic truth), and saves
    the once-per-episode estimates to out_dir/real_params.npz.

    This is the design's secondary/caveat arm: real-data numbers here are
    confounded by known identifiability pathologies in both
    parameterizations (documented elsewhere in this repo) -- it is not the
    driver of the "online" cadence comparison, which uses the controlled
    synthetic arm. `ok`/status is not computed here (no known truth to check
    against for real recordings); every recording's estimate is reported,
    none silently dropped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = load_all_recordings(CFG)
    _, phi_coeffs, sigma2_ar = fit_ar_beta_model_pooled(recordings, CFG)
    y_all = np.concatenate([recordings[f][1] for f in CFG.system_id_files])
    q_eta_global = 0.5 * float(np.var(y_all))

    names = list(CFG.system_id_files)
    theta_runs = []
    logp_runs = []

    for name in names:
        u, y = recordings[name]
        # pooled AR fit + innovation variance (not the synthetic
        # PHI_AR_TRUE/SIGMA2_TRUE, and not a per-recording fit) drive R and
        # the KF disturbance model, shared across every recording.
        th = run_theta_arm(u, y, phi_coeffs, sigma2_ar, q_eta_global, CFG, DUAL_CFG)
        lp = run_logparam_arm(u, y, phi_coeffs, sigma2_ar, q_eta_global, CFG, DUAL_CFG)
        print(f"{name}: theta k={th['k_final']:.2f}  logparam k={lp['k_final']:.2f}")
        theta_runs.append(th)
        logp_runs.append(lp)

    path = out_dir / "real_params.npz"
    np.savez(
        path,
        recording_names=np.array(names),
        theta_k_final=np.array([r["k_final"] for r in theta_runs]),
        theta_tau1_final=np.array([r["tau1_final"] for r in theta_runs]),
        theta_tau2_final=np.array([r["tau2_final"] for r in theta_runs]),
        logparam_k_final=np.array([r["k_final"] for r in logp_runs]),
        logparam_tau1_final=np.array([r["tau1_final"] for r in logp_runs]),
        logparam_tau2_final=np.array([r["tau2_final"] for r in logp_runs]),
    )
    print(f"Saved -> {path}")
    return path


if __name__ == "__main__":
    export_synthetic()
    export_real()
