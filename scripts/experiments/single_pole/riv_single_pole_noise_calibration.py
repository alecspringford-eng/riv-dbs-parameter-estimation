"""riv_single_pole_noise_calibration.py -- re-tune (q_off_frac, q_eta
multiplier, R multiplier) for the single-pole estimator specifically.

INVESTIGATION.md Entry 44: the single-pole model inherited these three
noise constants unchanged from the two-pole DualConfig (q_off_frac=1e-2,
q_eta=0.5*var(y), R=0.05*sigma2_ar), none re-derived for the single-pole
model's own dynamics. The semi-synthetic gate failed (6.7% ok-rate, a1
converging to the POLE_MAX clamp corner regardless of true tau) with a
clean isolating-experiment signature pointing at the beta-disturbance/eta
joint-observability confound, not PE order. This script does a truth-
informed grid search (the same methodology DualConfig.q_off_frac's own
docstring describes for the two-pole model: "Tuned by synthetic sweep") to
find calibration values for which the single-pole model actually
identifies (k, tau), before re-running the full semi-synthetic gate.

Does not modify riv_dual_estimation.py or riv_single_pole_estimation.py.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/mpc_export"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/single_pole"))
from riv_dual_estimation import CFG, DUAL_CFG
from riv_mpc_param_export import PHI_AR_TRUE, SIGMA2_TRUE
from riv_single_pole_estimation import AVG_PATIENT_PARAMS_1P
from riv_single_pole_semisynth_check import generate_synthetic_iv_1p, run_1p_arm

N_CAL = 1500          # shorter than the N_SYNTH=3000 gate recordings -- cheap search
SEEDS_CAL = range(10)  # subset of the gate's N_SEEDS=30, for search speed only
TAU_TRUTHS_CAL = (0.05, 0.15, 0.4)   # spans the gate's full tau grid
K_TRUE_CAL = AVG_PATIENT_PARAMS_1P["k"]

Q_OFF_FRAC_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 0.5)
Q_ETA_MULT_GRID = (0.1, 0.5, 1.0, 2.0)
R_MULT_GRID = (0.01, 0.05, 0.2, 1.0)

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "single_pole_semisynth"


def evaluate_combo(q_off_frac: float, q_eta_mult: float, r_mult: float) -> dict:
    dual_cfg = dataclasses.replace(DUAL_CFG, q_off_frac=q_off_frac)
    k_err, tau_err, ok = [], [], []
    for tau_true in TAU_TRUTHS_CAL:
        for seed in SEEDS_CAL:
            u, y = generate_synthetic_iv_1p(seed, K_TRUE_CAL, tau_true, N=N_CAL)
            q_eta = q_eta_mult * float(np.var(y))
            r = run_1p_arm(u, y, PHI_AR_TRUE, SIGMA2_TRUE, q_eta, CFG, dual_cfg,
                            K_TRUE_CAL, tau_true, r_mult=r_mult)
            k_err.append(abs(r["k_final"] - K_TRUE_CAL) / K_TRUE_CAL)
            tau_err.append(abs(r["tau_final"] - tau_true) / tau_true)
            ok.append(r["ok"])
    n = len(ok)
    return dict(q_off_frac=q_off_frac, q_eta_mult=q_eta_mult, r_mult=r_mult,
                ok_rate=sum(ok) / n, median_k_err=float(np.median(k_err)),
                median_tau_err=float(np.median(tau_err)), n=n)


def run_calibration_search() -> list[dict]:
    results = []
    total = len(Q_OFF_FRAC_GRID) * len(Q_ETA_MULT_GRID) * len(R_MULT_GRID)
    i = 0
    for q_off_frac in Q_OFF_FRAC_GRID:
        for q_eta_mult in Q_ETA_MULT_GRID:
            for r_mult in R_MULT_GRID:
                i += 1
                res = evaluate_combo(q_off_frac, q_eta_mult, r_mult)
                results.append(res)
                print(f"[{i}/{total}] q_off_frac={q_off_frac:<7g} q_eta_mult={q_eta_mult:<5g} "
                      f"r_mult={r_mult:<5g}  ok-rate={res['ok_rate']:.2f}  "
                      f"med|k_err|={res['median_k_err']:.2f}  med|tau_err|={res['median_tau_err']:.2f}")
    results.sort(key=lambda r: (-r["ok_rate"], r["median_k_err"] + r["median_tau_err"]))
    return results


if __name__ == "__main__":
    results = run_calibration_search()
    print("\n--- top 10 combos (by ok-rate, tie-broken by median error) ---")
    for r in results[:10]:
        print(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "noise_calibration_results.npz",
              q_off_frac=np.array([r["q_off_frac"] for r in results]),
              q_eta_mult=np.array([r["q_eta_mult"] for r in results]),
              r_mult=np.array([r["r_mult"] for r in results]),
              ok_rate=np.array([r["ok_rate"] for r in results]),
              median_k_err=np.array([r["median_k_err"] for r in results]),
              median_tau_err=np.array([r["median_tau_err"] for r in results]))
    print(f"Saved -> {OUT_DIR / 'noise_calibration_results.npz'}")
