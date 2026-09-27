"""riv_real_recon_multisine_sweep_check.py -- repeats the fs sweep (Entry 50 /
riv_sampling_rate_sweep_experiment.py: fs=50Hz "adaptive" arm baseline,
coarse geometric descent + bisection to find the failure boundary) with the
input excitation replaced by a multisine RECONSTRUCTED from the real
recorded data in real_recordings/delay_5/system_id_long_run.csv (the
longest of the real recordings, N=6120 samples at the recordings'
native Ts=0.04s / fs=25Hz).

Reconstruction method: Hann-windowed FFT of the (mean-removed) real u,
DC/leakage bins below 3 frequency-resolution-bins dropped, top-10 local
power maxima kept as the reconstructed tone set, EACH TONE WEIGHTED BY
sqrt(its power) (normalized to the fundamental) -- i.e. amplitude-matched
to the real signal's own spectrum, not equal-amplitude. This is a clean
harmonic series (confirmed by direct inspection): fundamental ~0.249Hz,
harmonics spaced ~0.249-0.253Hz apart out to the 10th at 2.5Hz -- i.e. the
real system-id stimulation is itself a ~4.01s-period repeating sequence
(~100 samples/period at 25Hz), not noise-like, so a small number of
harmonic lines captures effectively all of its structure, and that
structure is heavily fundamental-dominated (2nd harmonic ~27% of the
fundamental's amplitude, everything past the 6th harmonic under ~10%).
An earlier equal-amplitude version of this script (all 10 tones same
amplitude, see riv_wide_multisine_fs50_check.py's convention) was found to
fail much earlier (fs~9.7Hz) than the current 3-tone signal's ~5.6-6.1Hz
boundary -- this weighted version tests whether that was an artifact of
forcing equal power into harmonics the real signal barely uses. Tone
frequencies as a fraction of the recording's own Nyquist (12.5Hz) span
~0.02-0.20 -- much narrower than the "tutor's" wide multisine (0.008-0.8)
used in riv_wide_multisine_fs50_check.py, since the real stimulation is
comparatively low-bandwidth.

Same frequency-scaling convention as riv_wide_multisine_fs50_check.py:
preserve the extracted tones' fractions of Nyquist, re-anchor to whatever
fs is in play in the sweep (so it doesn't alias as fs drops), rather than
holding the extracted Hz values fixed at their 25Hz-native values.

Amplitude: make_weighted_multisine rescales the weighted sum to [lo,hi]
with lo=0, hi=cfg.u_max -- the current 2-pole synthetic scenario's own
envelope, NOT the real recording's tiny native amplitude (~0.004-0.026,
~100x smaller, tuned for an unrelated single-pole patient model) -- only the frequency
CONTENT is borrowed from the real data, the known-truth 2-pole plant
(k_true=62.11, tau1_true=0.05, tau2_true=0.25) and its own amplitude scale
are unchanged, exactly as in riv_wide_multisine_fs50_check.py.

Reused, unmodified (no source file below is edited in place): Config,
AUX_DECIMATED, perturb_true_params, run_dual_estimator_instrument (from
riv_instrument_update_rate_experiment.py), synthesize_ar_beta_fallback,
simulate_ar_beta (riv_single_pole_50hz_replication_check.py), make_multisine
(riv_excitation_protocol_design_check.py), theta/state-space/AR helpers
(riv_dual_estimation.py). make_scenario_real_recon below is a byte-for-byte
duplicate of riv_instrument_update_rate_experiment.make_scenario (lines
427-456) with only the input-generation line swapped (marked inline), and
eval_fs_real/coarse_descent_real/bisect_boundary_real duplicate
riv_sampling_rate_sweep_experiment.py's eval_fs/coarse_descent/
bisect_boundary (lines 68-121) with sweep_arm's per-seed body swapped to
call make_scenario_real_recon -- same algorithm/constants (COARSE_RATIO=
1/1.4, OK_RATE_THRESHOLD=0.5, ANCHOR_OK_RATE_MIN=0.8, BISECT_TOL_FRAC=0.10,
N_SEEDS=10) for direct comparability with Entry 50's documented ~5.6-6.1Hz
boundary.

CAVEAT (same as riv_wide_multisine_fs50_check.py, see its docstring): the
reused make_multisine draws one rng.uniform() phase per tone (10 scalar
draws) vs. the original make_multisine_input's single vectorized 3-draw
call, shifting the rng stream for every draw after the swap point within
make_scenario for a given seed -- so noise/disturbance realizations differ
from the original-signal runs even at "the same seed", not just u. Valid
for a median/ok-rate comparison, not a seed-by-seed diff.

Run: python3 scripts/riv_real_recon_multisine_sweep_check.py
"""
from __future__ import annotations

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
N_TONES = 10


def extract_dominant_freqs_hz(csv_path: Path, Ts_real: float,
                               n_tones: int = N_TONES) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed FFT of the real recorded input u, top-n_tones local
    power maxima (DC/leakage bins below 3 frequency bins excluded).
    Returns (freqs_hz, weights) where weights = sqrt(power) at each peak,
    normalized so the largest = 1 -- i.e. each tone's amplitude relative
    to the fundamental, matching the real signal's actual power
    distribution (heavily fundamental-dominated: ratio ~14:1 in power,
    ~3.7:1 in amplitude, over the 2nd harmonic, and >1000:1 in power over
    the smallest harmonics -- confirmed by direct inspection of this
    recording). Deterministic -- no rng involved, computed once at import
    time."""
    u = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=0)
    n = len(u)
    u_detrended = u - u.mean()
    spectrum = np.fft.rfft(u_detrended * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=Ts_real)
    power = np.abs(spectrum) ** 2
    df = 1.0 / (n * Ts_real)
    power[freqs < 3 * df] = 0.0  # drop DC + near-DC leakage

    maxima = [i for i in range(1, len(power) - 1) if power[i] > power[i - 1] and power[i] > power[i + 1]]
    maxima.sort(key=lambda i: -power[i])
    top = sorted(maxima[:n_tones])
    assert len(top) == n_tones, f"expected {n_tones} spectral peaks, found {len(top)}"
    amps = np.sqrt(power[top])
    return freqs[top], amps / amps.max()


_EXTRACTED_FREQS_HZ, _TONE_WEIGHTS = extract_dominant_freqs_hz(REAL_CSV, REAL_TS)
_REAL_NYQUIST_HZ = 1.0 / (2 * REAL_TS)
_REL_FREQS = _EXTRACTED_FREQS_HZ / _REAL_NYQUIST_HZ  # fractions of the recording's own Nyquist


def real_recon_freqs_hz(fs: float) -> np.ndarray:
    freqs = _REL_FREQS * (fs / 2.0)
    assert np.all(freqs < fs / 2.0), "tone at/above Nyquist -- would alias"
    return freqs


def make_weighted_multisine(n: int, lo: float, hi: float, Ts: float, freqs_hz: np.ndarray,
                             weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same construction as riv_excitation_protocol_design_check.make_multisine
    (sum of sinusoids, random phase, rescaled to [lo,hi]) but with a
    per-tone amplitude weight, so the reconstructed signal's relative tone
    power matches the real recording's instead of being equal-amplitude."""
    t = np.arange(n) * Ts
    x = np.zeros(n)
    for f, w in zip(freqs_hz, weights):
        x += w * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x = (x - x.min()) / (x.max() - x.min())
    return lo + x * (hi - lo)


def make_scenario_real_recon(cfg: Config, rng: np.random.Generator) -> dict:
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
    freqs = real_recon_freqs_hz(cfg.fs)
    u = make_weighted_multisine(cfg.N_sim, 0.0, cfg.u_max, cfg.Ts, freqs, _TONE_WEIGHTS, rng)
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


def run_seed_real(cfg: Config, seed: int) -> dict:
    """Mirrors sweep_arm's per-seed body (riv_instrument_update_rate_
    experiment.py:499-519) for the fixed adaptive arm, with
    make_scenario_real_recon in place of make_scenario."""
    rng = np.random.default_rng(seed)
    scn = make_scenario_real_recon(cfg, rng)
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
N_SEEDS = 30  # bumped from 10 -- tighter ok-rate estimate (SE ~9% vs ~16% at p=0.5)
COARSE_RATIO = 1 / 1.4
OK_RATE_THRESHOLD = 0.5
ANCHOR_OK_RATE_MIN = 0.8
BISECT_TOL_FRAC = 0.10
MIN_FS = 0.5
MAX_FS = 100.0  # upper end of the coarse ascent above the fs=50Hz anchor


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
    """Mirrors coarse_descent_real but stepping UP from the fs=50Hz anchor
    (fs *= 1/COARSE_RATIO each step) up to MAX_FS, purely for plot coverage
    above 50Hz -- no boundary search needed here since ok-rate only rises."""
    curve = []
    fs = BASE_CFG.fs
    while True:
        fs /= COARSE_RATIO  # == fs *= 1.4
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
    """Duplicate of riv_sampling_rate_sweep_experiment.plot_ok_rate_vs_fs
    (lines 124-157), identical except the title -- that function's title is
    hardcoded to the current 3-tone signal's baseline, which would mislabel
    this real-recon-weighted-multisine sweep's plot."""
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
    print(f"reference: Entry 50 (current 3-tone signal) boundary bracket ~[5.6,6.1]Hz")

    print(f"\ncoarse ascent above the fs=50Hz anchor (up to {MAX_FS}Hz):")
    ascent = coarse_ascent_real()
    full_curve += ascent

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "ok_rate_vs_fs.png"
    plot_ok_rate_vs_fs_real(full_curve, boundary_fs, N_SEEDS, plot_path)
    print(f"\nplot saved: {plot_path}")


if __name__ == "__main__":
    main()
