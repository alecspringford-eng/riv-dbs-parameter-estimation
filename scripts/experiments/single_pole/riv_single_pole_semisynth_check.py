"""Semi-synthetic validation gate for the single-pole (k, tau) RIV estimator
(riv_single_pole_estimation.py).

A known-truth plant is driven by a wide persistently exciting multisine, with
the same AR(3) beta-disturbance truth as the two-pole synthetic check
(riv_mpc_param_export.py's generate_synthetic_iv), Monte Carlo over seed and a
(k, tau) grid. This is the go/no-go before touching real data: run
riv_single_pole_real_data_check.py only if truth is recovered reliably.

Reuses public functions of riv_dual_estimation.py, riv_single_pole_estimation.py
and riv_excitation_protocol_design_check.py without modifying them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/excitation_design"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/mpc_export"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/single_pole"))
from riv_dual_estimation import CFG, DUAL_CFG, ar_to_state_space, pe_order_check
from riv_excitation_protocol_design_check import make_multisine
from riv_mpc_param_export import PHI_AR_TRUE, SIGMA2_TRUE, C0_TRUE
from riv_single_pole_estimation import (
    N_ETA_1P, AVG_PATIENT_PARAMS_1P,
    params_to_theta_1p, simulate_arx_recursion_1p,
    build_kf_setup_1p, run_dual_estimator_1p, theta_to_params_physical_1p,
)

N_SYNTH = 3000
N_SEEDS = 30
OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "single_pole_semisynth"
U_LO, U_HI = 0.004, 0.026  # real u's amplitude range
# Wide, log-spaced tones from well below the dominant lag's bandwidth to near-Nyquist
# (Ts=0.04s -> 12.5 Hz)
MULTISINE_FREQS_HZ = list(np.geomspace(0.1, 10.0, 10))
OK_REL_TOL = 0.3  # as in run_theta_arm


def make_wide_multisine(N: int, rng: np.random.Generator) -> np.ndarray:
    return make_multisine(N, U_LO, U_HI, CFG.Ts, MULTISINE_FREQS_HZ, rng)


def generate_synthetic_iv_1p(seed: int, k_true: float, tau_true: float,
                              N: int = N_SYNTH) -> tuple[np.ndarray, np.ndarray]:
    """Known-truth (u, y): wide multisine u, ARX(1)-simulated eta, and the AR(3)
    beta disturbance of riv_mpc_param_export.generate_synthetic_iv
    (PHI_AR_TRUE/SIGMA2_TRUE/C0_TRUE)."""
    rng = np.random.default_rng(seed)
    u = make_wide_multisine(N, rng)
    eta_true = simulate_arx_recursion_1p(params_to_theta_1p(k_true, tau_true, CFG.Ts), u)
    A_beta, C_beta = ar_to_state_space(PHI_AR_TRUE)
    p = len(PHI_AR_TRUE)
    xb = np.zeros(p)
    beta_true = np.zeros(N)
    for kk in range(N):
        beta_true[kk] = (C_beta @ xb)[0]
        xb = A_beta @ xb
        xb[0] += rng.normal(0, np.sqrt(SIGMA2_TRUE))
    y = C0_TRUE + beta_true - eta_true + rng.normal(0, 0.05, N)
    return u, y


def run_1p_arm(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, sigma2_ar: float,
                q_eta: float, cfg, dual_cfg, k_true: float, tau_true: float,
                n_epochs_online: int = 1, r_mult: float = 0.05) -> dict:
    """Run the single-pole dual estimator on (u, y): a full-convergence pass
    (dual_cfg.n_epochs) for the final estimate and a single causal pass for the
    online trajectory. The warm start stays at the AVG_PATIENT_PARAMS_1P prior
    regardless of truth, as on real data, so recovery under prior mismatch is
    tested too. r_mult is the R=0.05*sigma2_ar multiplier for the noise
    calibration sweeps (0.05 matches run_theta_arm)."""
    R = r_mult * sigma2_ar

    setup = build_kf_setup_1p(u, y, phi_ar, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale, theta0 = setup["u_norm"], setup["u_scale"], setup["theta0"]
    Q_c, x0, P0_kf, P0_rls = setup["Q_diag"], setup["x0"], setup["P0_kf"], setup["P0_rls"]
    phi_c, a1_prior = setup["phi_c"], setup["a1_prior"]
    n_off, p = setup["n_off"], setup["p"]

    (_, theta_hat_conv, _, _, _, _, _, theta_epoch_hist, _, _, _, _) = run_dual_estimator_1p(
        u_norm, y, phi_c, Q_c, R, x0, P0_kf, theta0, P0_rls, dual_cfg.lambda_ff,
        N_ETA_1P, p, n_off=n_off, gamma_pole=dual_cfg.gamma_pole, a1_prior=a1_prior,
        n_epochs=dual_cfg.n_epochs, n_snapshots=0, save_epochs=set())
    k_f, tau_f = theta_to_params_physical_1p(theta_hat_conv[-1], u_scale, cfg.Ts)

    (_, theta_hat_online, _, _, _, _, _, _, _, _, _, _) = run_dual_estimator_1p(
        u_norm, y, phi_c, Q_c, R, x0, P0_kf, theta0, P0_rls, dual_cfg.lambda_ff,
        N_ETA_1P, p, n_off=n_off, gamma_pole=dual_cfg.gamma_pole, a1_prior=a1_prior,
        n_epochs=n_epochs_online, n_snapshots=0, save_epochs=set())

    N = len(u)
    k_on = np.empty(N)
    tau_on = np.empty(N)
    for kk in range(N):
        k_on[kk], tau_on[kk] = theta_to_params_physical_1p(theta_hat_online[kk], u_scale, cfg.Ts)

    ok = bool(k_f > 0 and tau_f > 0
              and abs(k_f - k_true) / k_true < OK_REL_TOL
              and abs(tau_f - tau_true) / tau_true < OK_REL_TOL)
    return dict(k_final=k_f, tau_final=tau_f, ok=ok, k_online=k_on, tau_online=tau_on)


def check_excitation_pe_order() -> None:
    """The 1-pole regressor [eta_{k-1}, u_{k-1}] needs only PE order 2 (the
    2-pole model needs 4), so even a modest multisine should suffice."""
    rng = np.random.default_rng(CFG.seed)
    u = make_wide_multisine(N_SYNTH, rng)
    pe2 = pe_order_check(u, 2)
    print(f"multisine PE(2): cond={pe2['cond']:.3e} is_pe={pe2['is_pe']}")


def export_semisynth(n_seeds: int = N_SEEDS, N: int = N_SYNTH, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    check_excitation_pe_order()

    k_ref = AVG_PATIENT_PARAMS_1P["k"]
    tau_grid = [0.05, 0.1, 0.15, 0.25, 0.4]
    k_grid = [30.0, k_ref, 100.0]
    tau_ref = 0.15  # representative tau for the k-sweep

    results = {}

    def run_grid(label: str, k_true: float, tau_true: float) -> None:
        k_f, tau_f, ok = [], [], []
        for seed in range(n_seeds):
            u, y = generate_synthetic_iv_1p(seed, k_true, tau_true, N=N)
            q_eta = 0.5 * float(np.var(y))
            r = run_1p_arm(u, y, PHI_AR_TRUE, SIGMA2_TRUE, q_eta, CFG, DUAL_CFG, k_true, tau_true)
            k_f.append(r["k_final"]); tau_f.append(r["tau_final"]); ok.append(r["ok"])
        ok_rate = sum(ok) / n_seeds
        print(f"{label}: k_true={k_true:.2f} tau_true={tau_true:.3f}  ok-rate={sum(ok)}/{n_seeds}  "
              f"median k_hat={np.median(k_f):.2f}  median tau_hat={np.median(tau_f):.3f}")
        results[label] = dict(k_true=k_true, tau_true=tau_true,
                               k_final=np.array(k_f), tau_final=np.array(tau_f),
                               ok=np.array(ok), ok_rate=ok_rate)

    print("tau sweep (k fixed at avg-patient reference)")
    for tau_true in tau_grid:
        run_grid(f"tau={tau_true}", k_ref, tau_true)

    print("k sweep (tau fixed at representative value)")
    for k_true in k_grid:
        run_grid(f"k={k_true}", k_true, tau_ref)

    path = out_dir / "semisynth_params.npz"
    save_kwargs = {}
    for label, d in results.items():
        for key, val in d.items():
            save_kwargs[f"{label}__{key}"] = val
    np.savez(path, **save_kwargs)

    overall_ok = np.concatenate([d["ok"] for d in results.values()])
    print(f"overall ok-rate: {overall_ok.sum()}/{len(overall_ok)} ({100 * overall_ok.mean():.1f}%)")
    print(f"Saved -> {path}")
    return path


if __name__ == "__main__":
    export_semisynth()
