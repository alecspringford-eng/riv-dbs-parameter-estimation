"""riv_raw_real_replay_sweep_check.py -- same fs sweep as
riv_real_recon_multisine_sweep_check.py, but instead of reconstructing a
10-tone multisine FROM the real recording's spectrum, this replays the
ACTUAL recorded waveform itself (real_recordings/delay_5/
system_id_long_run.csv, N=6120 samples, native Ts=0.04s/fs=25Hz),
resampled to whatever Ts the current sweep point needs and tiled to fill
N_sim=80000 samples (the recording is confirmed near-exactly periodic --
see riv_real_recon_multisine_sweep_check.py's FFT: a clean harmonic
series, fundamental ~0.249Hz with 9 exact harmonics and no broadband
content -- so periodic tiling introduces no discontinuity beyond what's
already in the recording itself).

Resampling: linear interpolation of the recorded u(t) against continuous
time, wrapped with period = (recording length) * native Ts, via
np.interp(..., period=...). This is literally "what would this sampling
rate see if the real stimulation hardware ran at this fs" -- not a
frequency-content approximation like the multisine reconstruction.

Amplitude: same convention as riv_real_recon_multisine_sweep_check.py --
the raw real trace (native range ~0.004-0.026) is linearly rescaled to
[0, cfg.u_max=2.5] (same rescale-to-envelope mechanism as make_multisine's
own (x-min)/(max-min) step) so the comparison isolates waveform SHAPE
(what the real recording actually looks like, unweighted-by-decomposition)
from amplitude budget, exactly matching the other two signal variants
already tested this session.

Reused, unmodified (no source file below is edited in place): Config,
AUX_DECIMATED, perturb_true_params, run_dual_estimator_instrument (from
riv_instrument_update_rate_experiment.py), synthesize_ar_beta_fallback,
simulate_ar_beta (riv_single_pole_50hz_replication_check.py), theta/
state-space/AR helpers (riv_dual_estimation.py). make_scenario_raw_real is
a byte-for-byte duplicate of riv_instrument_update_rate_experiment.
make_scenario (lines 427-456) with only the input-generation line swapped.
eval_fs_raw/coarse_descent_raw/coarse_ascent_raw/bisect_boundary_raw
duplicate riv_sampling_rate_sweep_experiment.py's sweep algorithm (lines
68-121), same constants (COARSE_RATIO=1/1.4, OK_RATE_THRESHOLD=0.5,
ANCHOR_OK_RATE_MIN=0.8, BISECT_TOL_FRAC=0.10, N_SEEDS=30) for direct
comparability with Entry 50 and the multisine-reconstruction run.

Difference from the other two signal scripts: u itself is now fully
deterministic given cfg.Ts/cfg.N_sim (no rng draws at all for u -- there's
no phase/tone parameterization to randomize). So run_seed_raw's rng stream
is NOT shifted relative to the ORIGINAL make_multisine_input's 3-phase
draw the way the multisine-reconstruction scripts' rng stream was -- u
generation here consumes zero rng draws, so for a given seed,
theta_true/theta_hat0/AR(12) coefficients match the current-signal run
exactly, but beta_sim/v still differ from it (different number of
subsequent draws consumed: 0 here vs 3 in the original vs 10 in the
multisine reconstruction) -- so this is, again, not a seed-paired
ablation against either prior run, only against itself across fs.

Run: python3 scripts/riv_raw_real_replay_sweep_check.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root, for riv_* module imports

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REAL_CSV = REPO_ROOT / "real_recordings" / "delay_5" / "system_id_long_run.csv"
REAL_TS = 0.04  # native Ts of the real-recordings dataset (riv_dual_estimation.py Config.Ts)

_U_REAL = np.genfromtxt(REAL_CSV, delimiter=",", skip_header=1, usecols=0)
_NATIVE_T = np.arange(len(_U_REAL)) * REAL_TS
_PERIOD_S = len(_U_REAL) * REAL_TS  # whole recording tiled as one repeating block


def raw_real_u(Ts: float, N: int, lo: float, hi: float) -> np.ndarray:
    """Resample the real recorded u to Ts via linear interpolation (period-
    wrapped over the recording length), tile to N samples, rescale to
    [lo,hi] (same min-max rescale mechanism as make_multisine)."""
    t_query = np.arange(N) * Ts
    u = np.interp(t_query, _NATIVE_T, _U_REAL, period=_PERIOD_S)
    return lo + (u - u.min()) / (u.max() - u.min()) * (hi - lo)


def make_scenario_raw_real(cfg: Config, rng: np.random.Generator) -> dict:
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
    u = raw_real_u(cfg.Ts, cfg.N_sim, 0.0, cfg.u_max)
    # --- end SWAP (no rng draw here, unlike either multisine variant) ---

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


def run_seed_raw(cfg: Config, seed: int) -> dict:
    """Mirrors sweep_arm's per-seed body (riv_instrument_update_rate_
    experiment.py:499-519) for the fixed adaptive arm, with
    make_scenario_raw_real in place of make_scenario."""
    rng = np.random.default_rng(seed)
    scn = make_scenario_raw_real(cfg, rng)
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


# ==========================================================================
# fs sweep -- mirrors riv_sampling_rate_sweep_experiment.py's eval_fs /
# coarse_descent / bisect_boundary (lines 68-121), same constants.
# ==========================================================================

BASE_CFG = Config()  # fs=50Hz, Ts=0.02 -- Entry 49's confirmed "adaptive" baseline
N_SEEDS = 30
COARSE_RATIO = 1 / 1.4
OK_RATE_THRESHOLD = 0.5
ANCHOR_OK_RATE_MIN = 0.8
BISECT_TOL_FRAC = 0.10
MIN_FS = 0.5
MAX_FS = 100.0


def cfg_at_fs(fs: float) -> Config:
    return dataclasses.replace(BASE_CFG, fs=fs, Ts=1.0 / fs)


def eval_fs_raw(fs: float, n_seeds: int = N_SEEDS) -> dict:
    cfg = cfg_at_fs(fs)
    rows = [run_seed_raw(cfg, seed) for seed in range(n_seeds)]
    n_ok = sum(r["ok"] for r in rows)
    result = dict(
        fs=fs, Ts=cfg.Ts, n_seeds=n_seeds, n_ok=n_ok, ok_rate=n_ok / n_seeds,
        median_theta_err=float(np.median([r["theta_err"] for r in rows])),
        median_k_err=float(np.nanmedian([r["k_err"] for r in rows])),
        median_tau1_err=float(np.nanmedian([r["tau1_err"] for r in rows])),
        median_tau2_err=float(np.nanmedian([r["tau2_err"] for r in rows])),
        samples_per_tau1=cfg.tau1_true / cfg.Ts,
        samples_per_tau2=cfg.tau2_true / cfg.Ts,
        rows=rows,
    )
    print(f"  fs={fs:7.3f}Hz (Ts={cfg.Ts:.4f}s): ok-rate={n_ok}/{n_seeds}  "
          f"median theta_err={result['median_theta_err']:.4f}")
    return result


def coarse_descent_raw(n_seeds: int = N_SEEDS) -> list[dict]:
    curve = []
    fs = BASE_CFG.fs
    anchor = eval_fs_raw(fs, n_seeds=n_seeds)
    curve.append(anchor)
    if anchor["ok_rate"] < ANCHOR_OK_RATE_MIN:
        # ponytail: unlike riv_sampling_rate_sweep_experiment.py's ANCHOR_OK_RATE_MIN
        # check (which guards "does the pipeline still reproduce Entry 49 with the
        # SAME signal"), this script deliberately swaps the signal itself -- a
        # degraded fs=50 anchor here is the finding, not a pipeline bug. Warn and
        # keep going instead of aborting.
        print(f"  WARNING: fs={fs}Hz anchor ok-rate={anchor['ok_rate']:.2f} is below "
              f"the original-signal sanity floor ({ANCHOR_OK_RATE_MIN}) -- expected, "
              "this signal is already degraded at the baseline rate. Continuing anyway.")

    while curve[-1]["ok_rate"] >= OK_RATE_THRESHOLD and fs > MIN_FS:
        fs *= COARSE_RATIO
        curve.append(eval_fs_raw(fs, n_seeds=n_seeds))

    return curve


def coarse_ascent_raw(n_seeds: int = N_SEEDS) -> list[dict]:
    curve = []
    fs = BASE_CFG.fs
    while True:
        fs /= COARSE_RATIO
        if fs > MAX_FS:
            break
        curve.append(eval_fs_raw(fs, n_seeds=n_seeds))
    return curve


def bisect_boundary_raw(fs_pass: float, fs_fail: float,
                         n_seeds: int = N_SEEDS) -> tuple[float, float, list[dict]]:
    trace = []
    while fs_pass / fs_fail - 1.0 > BISECT_TOL_FRAC:
        fs_mid = float(np.sqrt(fs_pass * fs_fail))
        r = eval_fs_raw(fs_mid, n_seeds=n_seeds)
        trace.append(r)
        if r["ok_rate"] >= OK_RATE_THRESHOLD:
            fs_pass = fs_mid
        else:
            fs_fail = fs_mid
    return fs_pass, fs_fail, trace


OUT_DIR = REPO_ROOT / "results" / "raw_real_replay_sweep"


def plot_ok_rate_vs_fs_raw(curve: list[dict], boundary_fs: float, n_seeds: int, path: Path,
                            logx: bool = False, err_logy: bool = True) -> None:
    """Duplicate of riv_sampling_rate_sweep_experiment.plot_ok_rate_vs_fs
    (lines 124-157), title corrected for this signal."""
    pts = sorted(curve, key=lambda r: r["fs"])
    fss = [r["fs"] for r in pts]
    ok_rates = [r["ok_rate"] for r in pts]
    med_err = [r["median_theta_err"] for r in pts]
    spt1 = [r["samples_per_tau1"] for r in pts]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(fss, ok_rates, marker="o")
    axes[0].axvline(boundary_fs, color="r", ls="--", lw=1, label=f"boundary~{boundary_fs:.2f}Hz")
    axes[0].axhline(OK_RATE_THRESHOLD, color="gray", ls=":", lw=1)
    axes[0].set_ylabel(f"ok-rate ({n_seeds} seeds)"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Sampling-rate sweep: 2-pole 50Hz baseline, raw real-recording replay input")
    axes[0].legend()

    axes[1].plot(fss, med_err, marker="o", color="tab:orange")
    axes[1].set_ylabel("median theta_err")
    if err_logy:
        axes[1].set_yscale("log")

    axes[2].plot(fss, spt1, marker="o", color="tab:green", label="samples/tau1 (fastest pole)")
    axes[2].axhline(1.0, color="k", ls=":", lw=0.8, label="1 sample/tau1")
    axes[2].set_ylabel("samples per time constant"); axes[2].set_xlabel("fs (Hz)")
    axes[2].legend()

    if logx:
        for ax in axes:
            ax.set_xscale("log")

    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> None:
    print(f"Raw real-recording replay from {REAL_CSV.name} "
          f"(native fs={1/REAL_TS:.1f}Hz, N={len(_U_REAL)}, period={_PERIOD_S:.1f}s), "
          f"resampled+tiled per fs, rescaled to [0, {BASE_CFG.u_max}]\n")

    print("coarse descent (raw real-replay input):")
    curve = coarse_descent_raw()

    if curve[-1]["ok_rate"] >= OK_RATE_THRESHOLD:
        print(f"\nfs floor {MIN_FS}Hz reached without crossing ok-rate={OK_RATE_THRESHOLD} "
              "-- no failure boundary found in range.")
        full_curve = curve
        boundary_fs = float("nan")
    else:
        fs_pass, fs_fail = curve[-2]["fs"], curve[-1]["fs"]
        print(f"\nbisecting between fs_pass={fs_pass:.3f}Hz and fs_fail={fs_fail:.3f}Hz ...")
        fs_pass, fs_fail, trace = bisect_boundary_raw(fs_pass, fs_fail)
        boundary_fs = float(np.sqrt(fs_pass * fs_fail))
        full_curve = curve + trace

        print(f"\nboundary bracket: [{fs_fail:.3f}, {fs_pass:.3f}]Hz "
              f"(width {100*(fs_pass/fs_fail-1):.1f}%), geometric-mid estimate ~{boundary_fs:.3f}Hz")
        print("reference: Entry 50 (current 3-tone signal) boundary bracket ~[5.6,6.1]Hz")
        print("reference: real-recon weighted-multisine boundary ~12.48Hz")

    print(f"\ncoarse ascent above the fs=50Hz anchor (up to {MAX_FS}Hz):")
    ascent = coarse_ascent_raw()
    full_curve += ascent

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "ok_rate_vs_fs.png"
    plot_ok_rate_vs_fs_raw(full_curve, boundary_fs, N_SEEDS, plot_path)
    print(f"\nplot saved: {plot_path}")


if __name__ == "__main__":
    main()
