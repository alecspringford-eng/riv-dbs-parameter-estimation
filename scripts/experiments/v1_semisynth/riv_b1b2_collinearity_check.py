"""riv_b1b2_collinearity_check.py -- verification experiments for the b1/b2
non-separation found in riv_real_data_convergence_check.py (that script's
output shows a1_hat,a2_hat converging close to truth while b1_hat,b2_hat
collapse to a shared ~1.85-1.9 instead of separating to their true 2.9/2.1,
with ||theta_err|| plateauing ~0.94-0.97 across all 30 epochs).

Leading hypothesis: real u's [u_k-1,u_k-2] RLS regressor is near-collinear
(lag-1 autocorrelation 0.98-0.997, Gram-matrix condition number 800-3800 on
all 6 recordings -- directly measured), so the RLS can only resolve the
well-conditioned combination of (b1,b2), not each individually. This is
consistent with (not contradicted by) SEMISYNTHETIC_VS_REAL_REPORT.md
Entry 15, which found production V1 (offset state, clamping) recovers k
fine on the same real u -- k = (b1+b2)/(1-a1-a2) only needs the
well-conditioned sum, never separates b1 from b2.

Per INVESTIGATION.md Entries 41/42 (an excitation-based explanation that
looked clean at N=1 and reversed under replication), this hypothesis is
NOT accepted on the collinearity evidence alone. Four experiments, see
~/.claude/plans/why-is-convergence-failing-iterative-jellyfish.md:

  Exp 0: eigen-projection of the b1,b2 error onto the Gram-matrix
         eigenbasis (near-free, real-u run only).
  Exp 1: causal swap test -- same semisynthetic pipeline, real u replaced
         by a PRBS/multisine input AMPLITUDE-MATCHED to real u's own
         [min,max] range (not OneDrive's own u_max=2.5/du_max=1.5 verbatim
         -- that would also jump amplitude ~100x and confound the test
         with a Q/R-suitability change, not just a switching-pattern
         change), rate-limited at OneDrive's own relative slew fraction
         (du_max/u_max = 0.6) scaled onto that range. Uses a SEPARATE rng
         stream for u-synthesis only, so perturb_true_params's and
         generate_semisynthetic's random draws stay identical in sequence
         to the real-u baseline -- only u itself differs.
  Exp 2: PE(2) diagnostic side by side, real vs synthetic u (reuses
         riv_v1_logparam_diagnostics.pe_order_check).
  Exp 3: P0_B sweep -- does |b1_hat-b2_hat| close at ANY prior variance in
         the existing calibration grid (rules out prior mistuning, the
         competing hypothesis riv_real_data_convergence_check.py's own
         docstring flags: P0_B was retuned 0.03->0.5 for real u's much
         smaller amplitude).

Ts is pinned throughout to this repo's CFG.Ts=0.04 (NOT OneDrive's own
Config.Ts=0.02) -- a third, previously undocumented difference between the
two scripts that would otherwise confound the swap test.

Does not modify riv_dual_estimation.py or riv_real_data_convergence_check.py
-- imports and reuses their already-validated functions only.

Run: python3 scripts/riv_b1b2_collinearity_check.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import signal

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/real_data_checks"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/v1_logparam"))
from riv_dual_estimation import (
    CFG, N_ETA, AVG_PATIENT_PARAMS, params_to_theta, load_all_recordings,
    fit_ar_beta_model_pooled, simulate_arx_recursion, ar_to_state_space,
)
from riv_real_data_convergence_check import (
    R, LAMBDA_FF, N_EPOCHS, PERTURB_FRAC, P0_KF_STATE_VAR, P0_RLS,
    perturb_true_params, generate_semisynthetic, build_kf_inputs,
    run_dual_estimator, calibrate_p0_b, simulate_ar_beta,
)
from riv_v1_logparam_diagnostics import pe_order_check

# ==========================================================================
# Amplitude-matched PRBS / multisine excitation (ported from the OneDrive
# reference script's generate_prbs_raw/rate_limit/make_prbs_input/
# make_multisine_input, but re-parametrized to match real u's own
# amplitude range instead of OneDrive's own u_max=2.5/du_max=1.5).
# ==========================================================================

ONEDRIVE_N_BITS = 12
ONEDRIVE_MULTISINE_FRACS = (0.2, 0.5, 0.8)
ONEDRIVE_DU_OVER_UMAX = 1.5 / 2.5  # OneDrive's own relative slew fraction


def generate_prbs_raw(n_bits: int, length: int, u_max: float,
                       rng: np.random.Generator) -> np.ndarray:
    state = rng.integers(0, 2, size=n_bits)
    if not state.any():
        state[0] = 1
    seq, _ = signal.max_len_seq(n_bits, state=state)
    seq = np.resize(seq, length)
    return np.where(seq == 1, u_max, 0.0)


def rate_limit(u_raw: np.ndarray, du_max: float) -> np.ndarray:
    u = np.empty_like(u_raw)
    u[0] = u_raw[0]
    for kk in range(1, len(u_raw)):
        step = np.clip(u_raw[kk] - u[kk - 1], -du_max, du_max)
        u[kk] = u[kk - 1] + step
    return u


def make_amplitude_matched_prbs(u_real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u_min, u_max = float(u_real.min()), float(u_real.max())
    span = u_max - u_min
    du_max = ONEDRIVE_DU_OVER_UMAX * span
    u_raw = generate_prbs_raw(ONEDRIVE_N_BITS, len(u_real), span, rng)
    return rate_limit(u_raw, du_max) + u_min


def make_amplitude_matched_multisine(u_real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u_min, u_max = float(u_real.min()), float(u_real.max())
    span = u_max - u_min
    du_max = ONEDRIVE_DU_OVER_UMAX * span
    Ts = CFG.Ts
    f_max = du_max / (2 * np.pi * (span / 2) * Ts)
    freqs = np.array(ONEDRIVE_MULTISINE_FRACS) * f_max
    t = np.arange(len(u_real)) * Ts
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    raw = np.mean([np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)], axis=0)
    return (span / 2) * (1 + raw) + u_min


# ==========================================================================
# Exp 0: eigen-projection of the b1,b2 error onto the [u_k-1,u_k-2]
# Gram-matrix eigenbasis
# ==========================================================================

def gram_matrix_b_block(u: np.ndarray) -> np.ndarray:
    Z = np.column_stack([u[1:-1], u[:-2]])
    return Z.T @ Z


def eigen_projection(theta_final: np.ndarray, theta_true: np.ndarray, u: np.ndarray) -> dict:
    """Projects the (b1,b2) error onto the eigenbasis of [u_k-1,u_k-2]'s
    Gram matrix. eigh returns ascending eigenvalues, so index 0 is the
    ILL-conditioned direction, index 1 the WELL-conditioned one (which,
    for near-collinear columns, is close to (1,1)/sqrt2 -- i.e. close to
    the b1+b2 sum direction)."""
    G = gram_matrix_b_block(u)
    eigvals, eigvecs = np.linalg.eigh(G)
    d = theta_final[2:4] - theta_true[2:4]
    proj = eigvecs.T @ d
    return {
        "cond": float(eigvals[-1] / max(eigvals[0], 1e-300)),
        "ill_cond_err": float(abs(proj[0])),
        "well_cond_err": float(abs(proj[1])),
        "well_cond_dir": eigvecs[:, 1].tolist(),
    }


# ==========================================================================
# Exp 3: P0_B sweep -- does |b1_hat - b2_hat| close at any prior variance?
# ==========================================================================

def p0b_sweep(u: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray, theta_hat0: np.ndarray,
              grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
              ) -> list[tuple[float, float, float]]:
    p = len(phi_coeffs)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    out = []
    for p0_b in grid:
        Q_c, x0, P0_kf, P0_rls = build_kf_inputs(phi_coeffs, p0_b)
        theta_hat = run_dual_estimator(u, y, phi_c, Q_c, R, x0, P0_kf, theta_hat0, P0_rls,
                                        LAMBDA_FF, N_ETA, p, n_epochs=N_EPOCHS)[1]
        b1, b2 = theta_hat[-1, 2], theta_hat[-1, 3]
        out.append((p0_b, float(b1), float(b2)))
    return out


# ==========================================================================
# Exp 4 -- mirror-image control (Entry 43's flagged next step): hold u
# REAL (no swap), but rescale R/Q_ETA/Q_BETA_NOMINAL/Q_BETA_OTHER from
# OneDrive's verbatim constants (R=0.1, Q_ETA_DEFAULT=1e4, Q_BETA_NOMINAL=
# 5.0, Q_BETA_OTHER_DEFAULT=0.1 -- tuned for OneDrive's own ~100-600x
# larger u/y amplitude) to values derived from THIS recording's actual
# data, the same treatment P0_B already got. R and Q_BETA_NOMINAL are set
# to the production heuristic's own logic (riv_dual_estimation.py's
# R=0.05*sigma2_ar, q_eta=0.5*var(y)) rather than arbitrary rescaling
# factors -- q_beta_nominal = sigma2_ar directly, since that's exactly the
# AR innovation variance generate_semisynthetic already uses to simulate
# beta_true (a matched, not just amplitude-scaled, process-noise
# assumption). q_beta_other keeps OneDrive's own 2% ratio to nominal
# (0.1/5.0) rather than introducing a new unjustified constant.
# ==========================================================================

def build_kf_inputs_scaled(phi_coeffs: np.ndarray, p0_b: float, q_eta: float,
                            q_beta_nominal: float, q_beta_other: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = len(phi_coeffs)
    n = N_ETA + p
    Q_diag = np.full(n, q_eta)
    Q_diag[N_ETA:] = q_beta_other
    Q_diag[N_ETA] = q_beta_nominal
    x0 = np.zeros(n)
    P0_kf = P0_KF_STATE_VAR * np.eye(n)
    P0_rls = np.diag([P0_RLS, P0_RLS, p0_b, p0_b])
    return np.ascontiguousarray(Q_diag), x0, P0_kf, P0_rls


def generate_semisynthetic_scaled(u_real: np.ndarray, phi_coeffs: np.ndarray, sigma2_ar: float,
                                   theta_true: np.ndarray, rng: np.random.Generator,
                                   R_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(u_real)
    eta_true = simulate_arx_recursion(theta_true, u_real)
    A_beta, C_beta = ar_to_state_space(phi_coeffs)
    beta_true = simulate_ar_beta(A_beta, C_beta, sigma2_ar, N, rng)
    y = beta_true - eta_true + rng.normal(0.0, np.sqrt(R_val), N)
    return y, eta_true, beta_true


def p0b_sweep_scaled(u: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray, theta_hat0: np.ndarray,
                      q_eta: float, q_beta_nominal: float, q_beta_other: float, R_val: float,
                      grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
                      ) -> list[tuple[float, float, float]]:
    p = len(phi_coeffs)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    out = []
    for p0_b in grid:
        Q_c, x0, P0_kf, P0_rls = build_kf_inputs_scaled(phi_coeffs, p0_b, q_eta, q_beta_nominal,
                                                          q_beta_other)
        theta_hat = run_dual_estimator(u, y, phi_c, Q_c, R_val, x0, P0_kf, theta_hat0, P0_rls,
                                        LAMBDA_FF, N_ETA, p, n_epochs=N_EPOCHS)[1]
        b1, b2 = theta_hat[-1, 2], theta_hat[-1, 3]
        out.append((p0_b, float(b1), float(b2)))
    return out


# ==========================================================================
# Main experiment loop
# ==========================================================================

def main() -> None:
    recordings = load_all_recordings(CFG)
    cfg_log = replace(CFG, ar_log_domain=True)
    ar_order, phi_coeffs, sigma2_ar = fit_ar_beta_model_pooled(recordings, cfg_log)
    print(f"pooled AR({ar_order}) sigma2_ar={sigma2_ar:.4e}\n")

    theta_true = params_to_theta(AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"],
                                  AVG_PATIENT_PARAMS["tau2"], CFG.Ts)
    print(f"theta_true = {np.array2string(theta_true, precision=4)} "
          f"(a1,a2,b1,b2)\n")

    per_recording = {}
    for idx, name in enumerate(CFG.system_id_files):
        u_real, _ = recordings[name]
        rng = np.random.default_rng(idx)
        k0, tau1_0, tau2_0 = perturb_true_params(
            AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"], AVG_PATIENT_PARAMS["tau2"],
            rng, PERTURB_FRAC)
        theta_hat0 = params_to_theta(k0, tau1_0, tau2_0, CFG.Ts)
        y, _, _ = generate_semisynthetic(u_real, phi_coeffs, sigma2_ar, theta_true, rng)
        per_recording[name] = (u_real, y, phi_coeffs, theta_hat0)

    p0_b = calibrate_p0_b([per_recording[name] for name in CFG.system_id_files],
                           list(CFG.system_id_files))

    for idx, name in enumerate(CFG.system_id_files):
        u_real, y, _, theta_hat0 = per_recording[name]
        print(f"\n=== {name} ===")

        # --- baseline (real u) ---
        p = len(phi_coeffs)
        phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
        Q_c, x0, P0_kf, P0_rls = build_kf_inputs(phi_coeffs, p0_b)
        theta_hat_real = run_dual_estimator(u_real, y, phi_c, Q_c, R, x0, P0_kf, theta_hat0,
                                             P0_rls, LAMBDA_FF, N_ETA, p, n_epochs=N_EPOCHS)[1]
        theta_final_real = theta_hat_real[-1]

        # --- Exp 0: eigen-projection (real u) ---
        exp0 = eigen_projection(theta_final_real, theta_true, u_real)
        a_err = float(np.linalg.norm(theta_final_real[:2] - theta_true[:2]))
        print(f"Exp0 (real u): a1,a2 err={a_err:.4f}  "
              f"b-block cond={exp0['cond']:.3e}  "
              f"well-cond-dir err={exp0['well_cond_err']:.4f}  "
              f"ill-cond-dir err={exp0['ill_cond_err']:.4f}")

        # --- Exp 2: PE(2) diagnostic, real vs synthetic ---
        rng_u = np.random.default_rng(1000 + idx)
        u_prbs = make_amplitude_matched_prbs(u_real, rng_u)
        u_multisine = make_amplitude_matched_multisine(u_real, rng_u)
        pe_real = pe_order_check(u_real, order=2)
        pe_prbs = pe_order_check(u_prbs, order=2)
        pe_multi = pe_order_check(u_multisine, order=2)
        print(f"Exp2 PE(2) cond: real={pe_real['cond']:.3e}  "
              f"prbs={pe_prbs['cond']:.3e}  multisine={pe_multi['cond']:.3e}")

        # --- Exp 1: causal swap (fixed p0_b, fixed theta_hat0, only u differs) ---
        # Replays the SAME rng(idx) stream used to build the real-u baseline
        # (perturb_true_params's 3 draws, then generate_semisynthetic's beta/
        # noise draws) so those random draws land bit-identical to the
        # baseline -- only u itself (and what's deterministically derived
        # from it: eta_true, beta_true, y) differs.
        for tag, u_synth in (("prbs", u_prbs), ("multisine", u_multisine)):
            rng_replay = np.random.default_rng(idx)
            perturb_true_params(AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"],
                                 AVG_PATIENT_PARAMS["tau2"], rng_replay, PERTURB_FRAC)
            y_synth, _, _ = generate_semisynthetic(u_synth, phi_coeffs, sigma2_ar, theta_true,
                                                     rng_replay)
            theta_hat_synth = run_dual_estimator(u_synth, y_synth, phi_c, Q_c, R, x0, P0_kf,
                                                  theta_hat0, P0_rls, LAMBDA_FF, N_ETA, p,
                                                  n_epochs=N_EPOCHS)[1]
            tf = theta_hat_synth[-1]
            print(f"Exp1 ({tag}): theta_final={np.array2string(tf, precision=4)}  "
                  f"|b1-b2|={abs(tf[2] - tf[3]):.4f}  "
                  f"(real-u |b1-b2|={abs(theta_final_real[2] - theta_final_real[3]):.4f}, "
                  f"true |b1-b2|={abs(theta_true[2] - theta_true[3]):.4f})")

        # --- Exp 3: P0_B sweep (unscaled OneDrive constants) ---
        sweep = p0b_sweep(u_real, y, phi_coeffs, theta_hat0)
        gaps = [abs(b1 - b2) for _, b1, b2 in sweep]
        print(f"Exp3 P0_B sweep |b1-b2|: min={min(gaps):.4f} max={max(gaps):.4f} "
              f"(true |b1-b2|={abs(theta_true[2] - theta_true[3]):.4f})")
        for p0b_val, b1, b2 in sweep:
            print(f"    P0_B={p0b_val:5.2f}: b1={b1:7.4f} b2={b2:7.4f} |b1-b2|={abs(b1-b2):.4f}")

        # --- Exp 4: mirror-image control -- real u, rescaled R/Q_ETA/Q_BETA ---
        R_scaled = 0.05 * sigma2_ar
        q_beta_nominal_scaled = sigma2_ar
        q_beta_other_scaled = 0.02 * sigma2_ar
        rng_replay4 = np.random.default_rng(idx)
        perturb_true_params(AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"],
                             AVG_PATIENT_PARAMS["tau2"], rng_replay4, PERTURB_FRAC)
        y_scaled, _, _ = generate_semisynthetic_scaled(u_real, phi_coeffs, sigma2_ar, theta_true,
                                                         rng_replay4, R_scaled)
        q_eta_scaled = 0.5 * float(np.var(y_scaled))
        print(f"Exp4 scaled constants: R={R_scaled:.4e} (was {R:.4e})  "
              f"q_eta={q_eta_scaled:.4e} (was {1e4:.4e})  "
              f"q_beta_nom={q_beta_nominal_scaled:.4e} (was 5.0)  "
              f"q_beta_other={q_beta_other_scaled:.4e} (was 0.1)")

        Q_c4, x0_4, P0_kf_4, P0_rls_4 = build_kf_inputs_scaled(
            phi_coeffs, p0_b, q_eta_scaled, q_beta_nominal_scaled, q_beta_other_scaled)
        theta_hat_scaled = run_dual_estimator(u_real, y_scaled, phi_c, Q_c4, R_scaled, x0_4,
                                               P0_kf_4, theta_hat0, P0_rls_4, LAMBDA_FF, N_ETA, p,
                                               n_epochs=N_EPOCHS)[1]
        theta_final_scaled = theta_hat_scaled[-1]
        exp4_proj = eigen_projection(theta_final_scaled, theta_true, u_real)
        a_err_scaled = float(np.linalg.norm(theta_final_scaled[:2] - theta_true[:2]))
        print(f"Exp4 (real u, scaled): theta_final={np.array2string(theta_final_scaled, precision=4)}  "
              f"a1,a2 err={a_err_scaled:.4f} (was {a_err:.4f})  "
              f"|b1-b2|={abs(theta_final_scaled[2] - theta_final_scaled[3]):.4f} "
              f"(unscaled real-u was {abs(theta_final_real[2] - theta_final_real[3]):.4f}, "
              f"true={abs(theta_true[2] - theta_true[3]):.4f})  "
              f"well-cond-dir err={exp4_proj['well_cond_err']:.4f} (was {exp0['well_cond_err']:.4f})  "
              f"ill-cond-dir err={exp4_proj['ill_cond_err']:.4f} (was {exp0['ill_cond_err']:.4f})")

        sweep4 = p0b_sweep_scaled(u_real, y_scaled, phi_coeffs, theta_hat0, q_eta_scaled,
                                   q_beta_nominal_scaled, q_beta_other_scaled, R_scaled)
        gaps4 = [abs(b1 - b2) for _, b1, b2 in sweep4]
        print(f"Exp4 P0_B sweep (scaled) |b1-b2|: min={min(gaps4):.4f} max={max(gaps4):.4f} "
              f"(unscaled sweep was min={min(gaps):.4f} max={max(gaps):.4f})")
        for p0b_val, b1, b2 in sweep4:
            print(f"    P0_B={p0b_val:5.2f}: b1={b1:7.4f} b2={b2:7.4f} |b1-b2|={abs(b1-b2):.4f}")


if __name__ == "__main__":
    main()
