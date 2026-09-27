"""riv_wide_multisine_fs50_check.py -- reruns the fs=50Hz, 10-seed "adaptive"
arm baseline (INVESTIGATION.md Entry 50 / riv_sampling_rate_sweep_experiment.py)
with the input excitation swapped from the current 3-tone rate-limited
multisine to a 10-tone log-spaced "wide persistently exciting multisine",
the same pattern already used for a different (single-pole) model in
riv_single_pole_semisynth_check.py (there: geomspace(0.1,10,10) Hz, fixed,
tuned for that script's own fs=25Hz/Ts=0.04s setup -- attributed to "the
tutor's suggested method").

Frequency scaling: the tutor's tones as a fraction of THEIR OWN Nyquist
(12.5Hz) span geomspace(0.008, 0.8, 10) (0.8%-80% of Nyquist). This script
preserves that log-spaced relative content but anchors it to cfg.fs's own
Nyquist instead of the fixed 25Hz reference, so it doesn't alias at other
sampling rates (see wide_multisine_freqs_hz). At fs=50Hz this gives tones
at geomspace(0.2, 20, 10) Hz.

Amplitude: reuses make_multisine's rescale-to-[lo,hi] behavior with
lo=0, hi=cfg.u_max -- the same envelope as the current signal (NOT the
single-pole script's U_LO/U_HI=(0.004,0.026), which is tuned for a
different, unrelated real-data amplitude range).

Reused, unmodified (no source file below is edited in place): Config,
AUX_DECIMATED, perturb_true_params, run_dual_estimator_instrument (all from
riv_instrument_update_rate_experiment.py), synthesize_ar_beta_fallback,
simulate_ar_beta (riv_single_pole_50hz_replication_check.py), make_multisine
(riv_excitation_protocol_design_check.py), and the theta/state-space/AR
helpers from riv_dual_estimation.py.

make_scenario itself is not parameterized/injectable, so make_scenario_wide
below is a byte-for-byte duplicate of
riv_instrument_update_rate_experiment.make_scenario (lines 427-456) with
only the input-generation line swapped (marked inline).

CAVEAT (not a bug, must be disclosed): make_scenario's rng draws happen in
order (a) perturb_true_params, (b) synthesize_ar_beta_fallback, (c) u
generation [swap point], (d) simulate_ar_beta's process noise, (e)
measurement noise v. Steps (a)/(b) precede the swap, so for the same seed
theta_true/theta_hat0/AR(12) coefficients are bit-identical between the
old- and new-signal runs. But make_multisine draws one phase per tone via
10 separate scalar rng.uniform() calls, vs. the original's single
vectorized 3-draw call -- a different draw count that shifts the rng
stream for everything after it. So beta_sim and v's noise realizations
differ too, not just u. The two runs are therefore NOT a strict
seed-paired ablation (noise realizations become independent draws, not
matched pairs) -- valid for a median-over-10-seeds comparison, not for a
seed-by-seed diff.

Run: python3 scripts/riv_wide_multisine_fs50_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root, for riv_* module imports

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/single_pole"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
from riv_dual_estimation import (
    params_to_theta, theta_to_params, theta_to_observer_canonical,
    simulate_ss, ar_to_state_space, build_process_noise_diag,
)
from riv_single_pole_50hz_replication_check import (
    synthesize_ar_beta_fallback, simulate_ar_beta,
)
from riv_instrument_update_rate_experiment import (
    Config, AUX_DECIMATED, perturb_true_params, run_dual_estimator_instrument,
)
from riv_excitation_protocol_design_check import make_multisine

# Tutor's wide multisine (riv_single_pole_semisynth_check.py) uses
# geomspace(0.1, 10, 10) Hz at its own fs_ref=25Hz (Nyquist=12.5Hz).
# As a fraction of THAT Nyquist: geomspace(0.1,10,10)/12.5 = geomspace(0.008, 0.8, 10)
# -> 0.8%-80% of Nyquist, log-spaced, 10 tones. Anchor to whatever fs is in
# play instead of the fixed 25Hz setup so it doesn't alias at low fs.
_REL_FREQS = np.geomspace(0.008, 0.8, 10)


def wide_multisine_freqs_hz(fs: float) -> np.ndarray:
    freqs = _REL_FREQS * (fs / 2.0)
    assert np.all(freqs < fs / 2.0), "tone at/above Nyquist -- would alias"
    return freqs


def make_scenario_wide(cfg: Config, rng: np.random.Generator) -> dict:
    """Duplicate of riv_instrument_update_rate_experiment.make_scenario
    (lines 427-456), identical except the marked SWAP block."""
    theta_true = params_to_theta(cfg.k_true, cfg.tau1_true, cfg.tau2_true, cfg.Ts)
    k0, tau1_0, tau2_0 = perturb_true_params(cfg.k_true, cfg.tau1_true, cfg.tau2_true,
                                              rng, cfg.perturb_frac)
    theta_hat0 = params_to_theta(k0, tau1_0, tau2_0, cfg.Ts)

    phi_coeffs, sigma2_ar = synthesize_ar_beta_fallback(rng, cfg.p_ar_fallback, cfg.sigma2_ar_fallback)
    p = len(phi_coeffs)
    n = 2 + p

    # --- SWAP (was: u, freqs = make_multisine_input(cfg, rng)) ---
    freqs = wide_multisine_freqs_hz(cfg.fs)
    u = make_multisine(cfg.N_sim, 0.0, cfg.u_max, cfg.Ts, list(freqs), rng)
    # --- end SWAP ---

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


def run_seed(cfg: Config, seed: int) -> dict:
    """Mirrors sweep_arm's per-seed body (riv_instrument_update_rate_
    experiment.py:499-519) for the fixed adaptive arm, with make_scenario_wide
    in place of make_scenario. All run_dual_estimator_instrument arguments
    match run_arm_for_seed's call exactly (n_snapshots=20, save_epochs=set())."""
    rng = np.random.default_rng(seed)
    scn = make_scenario_wide(cfg, rng)
    eta_aux_full = np.zeros(cfg.N_sim)  # unused placeholder, AUX_DECIMATED ignores it
    x_hat, theta_hat, y_hat, Pkf_final, Prls_final, theta_epoch_hist, _ = run_dual_estimator_instrument(
        scn["u"], scn["y_true"], scn["phi_c"], scn["Q_c"], cfg.R, scn["x0"], scn["P0_kf"],
        scn["theta_hat0"], scn["P0_rls"], cfg.lambda_ff, 2, scn["p"],
        AUX_DECIMATED, 1, eta_aux_full, n_epochs=cfg.n_epochs, n_snapshots=20, save_epochs=set())

    theta_final = theta_hat[-1]
    theta_err = float(np.linalg.norm(theta_final - scn["theta_true"]))
    try:
        k_hat, tau1_hat, tau2_hat = theta_to_params(theta_final, cfg.Ts)
    except ValueError:
        k_hat, tau1_hat, tau2_hat = float("nan"), float("nan"), float("nan")
    k_err = abs(k_hat - cfg.k_true) / cfg.k_true if np.isfinite(k_hat) else float("nan")
    tau1_err = abs(tau1_hat - cfg.tau1_true) / cfg.tau1_true if np.isfinite(tau1_hat) else float("nan")
    tau2_err = abs(tau2_hat - cfg.tau2_true) / cfg.tau2_true if np.isfinite(tau2_hat) else float("nan")
    ok = bool(np.isfinite(k_err) and k_err < 0.3 and tau1_err < 0.3 and tau2_err < 0.3)
    return dict(seed=seed, k_hat=k_hat, tau1_hat=tau1_hat, tau2_hat=tau2_hat,
                theta_err=theta_err, k_err=k_err, tau1_err=tau1_err, tau2_err=tau2_err, ok=ok)


N_SEEDS = 10
CFG = Config()  # fs=50Hz, Ts=0.02 -- identical anchor cfg to Entry 50

BASELINE = dict(k_hat=62.125, tau1_hat=0.05094, tau2_hat=0.24423,
                 ok_rate="10/10", theta_err=0.0344)  # Entry 50 / this-session rerun


def main() -> None:
    rows = [run_seed(CFG, seed) for seed in range(N_SEEDS)]
    print(f"{'seed':>4}{'k_hat':>10}{'tau1_hat':>12}{'tau2_hat':>12}{'ok':>6}")
    for r in rows:
        print(f"{r['seed']:>4}{r['k_hat']:>10.4f}{r['tau1_hat']:>12.5f}{r['tau2_hat']:>12.5f}{str(r['ok']):>6}")
    n_ok = sum(r["ok"] for r in rows)
    print("-" * 44)
    print(f"median  k_hat={np.median([r['k_hat'] for r in rows]):.4f}  "
          f"tau1_hat={np.median([r['tau1_hat'] for r in rows]):.5f}  "
          f"tau2_hat={np.median([r['tau2_hat'] for r in rows]):.5f}  "
          f"ok_rate={n_ok}/{N_SEEDS}  "
          f"median_theta_err={np.median([r['theta_err'] for r in rows]):.4f}")
    print(f"baseline (current 3-tone signal, Entry 50): {BASELINE}")


if __name__ == "__main__":
    main()
