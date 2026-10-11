
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root

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
REAL_TS = 0.04  # native Ts of the real recordings (Config.Ts)
N_TONES = 10


def extract_dominant_freqs_hz(csv_path: Path, Ts_real: float,
                               n_tones: int = N_TONES) -> tuple[np.ndarray, np.ndarray]:
    u = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=0)
    n = len(u)
    u_detrended = u - u.mean()
    spectrum = np.fft.rfft(u_detrended * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=Ts_real)
    power = np.abs(spectrum) ** 2
    df = 1.0 / (n * Ts_real)
    power[freqs < 3 * df] = 0.0

    maxima = [i for i in range(1, len(power) - 1) if power[i] > power[i - 1] and power[i] > power[i + 1]]
    maxima.sort(key=lambda i: -power[i])
    top = sorted(maxima[:n_tones])
    assert len(top) == n_tones, f"expected {n_tones} spectral peaks, found {len(top)}"
    amps = np.sqrt(power[top])
    return freqs[top], amps / amps.max()


_EXTRACTED_FREQS_HZ, _TONE_WEIGHTS = extract_dominant_freqs_hz(REAL_CSV, REAL_TS)
_REAL_NYQUIST_HZ = 1.0 / (2 * REAL_TS)
_REL_FREQS = _EXTRACTED_FREQS_HZ / _REAL_NYQUIST_HZ


def real_recon_freqs_hz(fs: float) -> np.ndarray:
    freqs = _REL_FREQS * (fs / 2.0)
    assert np.all(freqs < fs / 2.0), "tone at/above Nyquist -- would alias"
    return freqs


def make_weighted_multisine(n: int, lo: float, hi: float, Ts: float, freqs_hz: np.ndarray,
                             weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """riv_excitation_protocol_design_check.make_multisine with a per-tone
    amplitude weight, so relative tone power matches the real recording."""
    t = np.arange(n) * Ts
    x = np.zeros(n)
    for f, w in zip(freqs_hz, weights):
        x += w * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x = (x - x.min()) / (x.max() - x.min())
    return lo + x * (hi - lo)


def make_scenario_real_recon(cfg: Config, rng: np.random.Generator) -> dict:
    """riv_instrument_update_rate_experiment.make_scenario with the marked SWAP block changed."""
    theta_true = params_to_theta(cfg.k_true, cfg.tau1_true, cfg.tau2_true, cfg.Ts)
    k0, tau1_0, tau2_0 = perturb_true_params(cfg.k_true, cfg.tau1_true, cfg.tau2_true,
                                              rng, cfg.perturb_frac)
    theta_hat0 = params_to_theta(k0, tau1_0, tau2_0, cfg.Ts)

    phi_coeffs, sigma2_ar = synthesize_ar_beta_fallback(rng, cfg.p_ar_fallback, cfg.sigma2_ar_fallback)
    p = len(phi_coeffs)
    n = 2 + p

    # SWAP (was: u, freqs = make_multisine_input(cfg, rng))
    freqs = real_recon_freqs_hz(cfg.fs)
    u = make_weighted_multisine(cfg.N_sim, 0.0, cfg.u_max, cfg.Ts, freqs, _TONE_WEIGHTS, rng)

    A_eta_t, B_eta_t, C_eta_t = theta_to_observer_canonical(theta_true)
    eta_true = simulate_ss(A_eta_t, B_eta_t, C_eta_t, u)

    A_beta, C_beta = ar_to_state_space(phi_coeffs)
    beta_sim = simulate_ar_beta(A_beta, C_beta, sigma2_ar, cfg.N_sim, rng)
    v = rng.normal(0.0, np.sqrt(cfg.R), size=cfg.N_sim)
    y_true = beta_sim - eta_true + v  # no offset term

    Q_diag = build_process_noise_diag(2, p, cfg.q_eta_default, cfg.q_beta_nominal,
                                       cfg.q_beta_other_default)
    x0 = np.zeros(n)
    P0_kf = cfg.p0_kf_state_var * np.eye(n)
    P0_rls = cfg.p0_rls * np.eye(4)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    Q_c = np.ascontiguousarray(Q_diag, dtype=np.float64)

    return dict(theta_true=theta_true, theta_hat0=theta_hat0, phi_c=phi_c, Q_c=Q_c,
                u=u, y_true=y_true, eta_true=eta_true, x0=x0, P0_kf=P0_kf, P0_rls=P0_rls, p=p)


def run_seed_real(cfg: Config, seed: int) -> dict:
    """sweep_arm's per-seed body for the adaptive arm, using make_scenario_real_recon."""
    rng = np.random.default_rng(seed)
    scn = make_scenario_real_recon(cfg, rng)
    eta_aux_full = np.zeros(cfg.N_sim)  # unused, AUX_DECIMATED ignores it
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


# fs sweep: eval_fs / coarse_descent / bisect_boundary of
# riv_sampling_rate_sweep_experiment.py, same constants

BASE_CFG = Config()  # fs=50Hz, Ts=0.02, the "adaptive" baseline
N_SEEDS = 30  # SE ~9% on ok-rate at p=0.5
COARSE_RATIO = 1 / 1.4
OK_RATE_THRESHOLD = 0.5
ANCHOR_OK_RATE_MIN = 0.8
BISECT_TOL_FRAC = 0.10
MIN_FS = 0.5
MAX_FS = 100.0  # upper end of the ascent above the 50Hz anchor


def cfg_at_fs(fs: float) -> Config:
    import dataclasses
    return dataclasses.replace(BASE_CFG, fs=fs, Ts=1.0 / fs)


def eval_fs_real(fs: float, n_seeds: int = N_SEEDS) -> dict:
    cfg = cfg_at_fs(fs)
    rows = [run_seed_real(cfg, seed) for seed in range(n_seeds)]
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


def coarse_descent_real(n_seeds: int = N_SEEDS) -> list[dict]:
    curve = []
    fs = BASE_CFG.fs
    anchor = eval_fs_real(fs, n_seeds=n_seeds)
    curve.append(anchor)
    if anchor["ok_rate"] < ANCHOR_OK_RATE_MIN:
        raise RuntimeError(
            f"fs={fs}Hz anchor ok-rate={anchor['ok_rate']:.2f} does not reproduce "
            f"the confirmed baseline (expected >={ANCHOR_OK_RATE_MIN}) -- "
            "sweep would not be trustworthy, stopping.")

    while curve[-1]["ok_rate"] >= OK_RATE_THRESHOLD and fs > MIN_FS:
        fs *= COARSE_RATIO
        curve.append(eval_fs_real(fs, n_seeds=n_seeds))

    return curve


def coarse_ascent_real(n_seeds: int = N_SEEDS) -> list[dict]:
    """coarse_descent_real stepping up from the fs=50Hz anchor (fs *= 1/COARSE_RATIO)
    to MAX_FS, for plot coverage only (ok-rate only rises, so no boundary search)."""
    curve = []
    fs = BASE_CFG.fs
    while True:
        fs /= COARSE_RATIO
        if fs > MAX_FS:
            break
        curve.append(eval_fs_real(fs, n_seeds=n_seeds))
    return curve


def bisect_boundary_real(fs_pass: float, fs_fail: float,
                          n_seeds: int = N_SEEDS) -> tuple[float, float, list[dict]]:
    trace = []
    while fs_pass / fs_fail - 1.0 > BISECT_TOL_FRAC:
        fs_mid = float(np.sqrt(fs_pass * fs_fail))
        r = eval_fs_real(fs_mid, n_seeds=n_seeds)
        trace.append(r)
        if r["ok_rate"] >= OK_RATE_THRESHOLD:
            fs_pass = fs_mid
        else:
            fs_fail = fs_mid
    return fs_pass, fs_fail, trace


OUT_DIR = REPO_ROOT / "results" / "real_recon_multisine_sweep"


def plot_ok_rate_vs_fs_real(curve: list[dict], boundary_fs: float, n_seeds: int, path: Path,
                             logx: bool = False, err_logy: bool = True) -> None:
    """riv_sampling_rate_sweep_experiment.plot_ok_rate_vs_fs with its hardcoded
    3-tone title replaced."""
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
    axes[0].set_title("Sampling-rate sweep: 2-pole 50Hz baseline, real-recon weighted-multisine input")
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
    print(f"Reconstructed {N_TONES}-tone weighted multisine from {REAL_CSV.name} "
          f"(native fs={1/REAL_TS:.1f}Hz): tones (at native fs) = "
          f"{np.round(_EXTRACTED_FREQS_HZ, 4).tolist()} Hz "
          f"({np.round(_REL_FREQS, 4).tolist()} of Nyquist)")
    print(f"amplitude weights (relative to fundamental): {np.round(_TONE_WEIGHTS, 4).tolist()}")
    print(f"At fs=50Hz: {np.round(real_recon_freqs_hz(50.0), 3).tolist()} Hz\n")

    print("coarse descent (real-recon-multisine input):")
    curve = coarse_descent_real()

    if curve[-1]["ok_rate"] >= OK_RATE_THRESHOLD:
        print(f"\nfs floor {MIN_FS}Hz reached without crossing ok-rate={OK_RATE_THRESHOLD} "
              "-- no failure boundary found in range.")
        return

    fs_pass, fs_fail = curve[-2]["fs"], curve[-1]["fs"]
    print(f"\nbisecting between fs_pass={fs_pass:.3f}Hz and fs_fail={fs_fail:.3f}Hz ...")
    fs_pass, fs_fail, trace = bisect_boundary_real(fs_pass, fs_fail)
    boundary_fs = float(np.sqrt(fs_pass * fs_fail))
    full_curve = curve + trace

    print(f"\nboundary bracket: [{fs_fail:.3f}, {fs_pass:.3f}]Hz "
          f"(width {100*(fs_pass/fs_fail-1):.1f}%), geometric-mid estimate ~{boundary_fs:.3f}Hz")
    print("reference: an earlier experiment (current 3-tone signal) boundary bracket ~[5.6,6.1]Hz")

    print(f"\ncoarse ascent above the fs=50Hz anchor (up to {MAX_FS}Hz):")
    ascent = coarse_ascent_real()
    full_curve += ascent

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "ok_rate_vs_fs.png"
    plot_ok_rate_vs_fs_real(full_curve, boundary_fs, N_SEEDS, plot_path)
    print(f"\nplot saved: {plot_path}")


if __name__ == "__main__":
    main()
