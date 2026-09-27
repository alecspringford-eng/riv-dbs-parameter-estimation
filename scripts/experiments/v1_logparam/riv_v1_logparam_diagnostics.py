"""
Two diagnostics for the V1-logp estimator (riv_v1_logparam_riv_experiment.py):

1. Persistent-excitation (PE) order check on real u -- BEFORE trying a
   higher plant order (Ljung, "System Identification: Theory for the
   User", 2nd ed, sec 5.5/13.4): a linear ARX(na,nb) model needs u to be
   PE of order >= na+nb for the parameters to be (even in principle)
   identifiable from any realisation of the output. Current plant is
   n_eta=2 lags (na=nb=2, PE order 4 needed); a candidate 3-lag plant
   needs PE order 6. u is PE of order n iff the n x n Toeplitz sample-
   autocovariance matrix of u is positive definite (nonsingular) -- we
   check this directly rather than assuming richer plant = free upgrade.

2. Innovation (KF one-step residual) autocorrelation + spectrum, split
   stim-ON vs no-stim. Never done before in this project -- every prior
   whiteness check (INVESTIGATION.md Entry 13/14/15/A6) only reports the
   pass/fail frac_exceed SCALAR from residual_whiteness_test, discarding
   the actual shape of the leftover correlation. A periodic peak would
   argue for a missing oscillatory/resonant state; short-lag decay would
   argue for underspecified plant order; concentration in stim-ON windows
   would sharpen the beta-eta coupling story (Entries 20/23/25).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import CFG, DUAL_CFG, load_all_recordings, fit_ar_beta_model_per_recording
from riv_v1_logparam_riv_experiment import run_identification_logparam


# ==========================================================================
# 1. Persistent-excitation order check
# ==========================================================================

def pe_order_check(u: np.ndarray, order: int, rel_eig_floor: float = 1e-2) -> dict:
    """u is PE of `order` iff the order x order Toeplitz sample-autocovariance
    matrix is positive definite. rel_eig_floor is a practical near-singularity
    guard (real, noisy data is rarely EXACTLY singular) -- flags "PE in name
    only" the same way POLE_MAX/B_MAX flag runaway numerics elsewhere in this
    codebase, not a claim of a validated statistical threshold."""
    n = len(u)
    ru = np.array([np.dot(u[:n - lag], u[lag:]) / n for lag in range(order)])
    R = toeplitz(ru)
    eigs = np.linalg.eigvalsh(R)
    min_eig, max_eig = float(eigs.min()), float(eigs.max())
    cond = max_eig / max(min_eig, 1e-300)
    is_pe = min_eig > rel_eig_floor * ru[0]
    return {"order": order, "min_eig": min_eig, "max_eig": max_eig,
            "cond": cond, "ru0": float(ru[0]), "is_pe": bool(is_pe)}


def pe_report(cfg) -> dict:
    recordings = load_all_recordings(cfg)
    report = {}
    print(f"{'recording':<20} {'PE order 4 (n_eta=2)':<28} {'PE order 6 (n_eta=3)':<28}")
    for name in cfg.system_id_files:
        u, _ = recordings[name]
        pe4 = pe_order_check(u, 4)
        pe6 = pe_order_check(u, 6)
        report[name.replace(".csv", "")] = {"pe4": pe4, "pe6": pe6}
        print(f"{name:<20} cond={pe4['cond']:>10.3e} pe={pe4['is_pe']!s:<6}   "
              f"cond={pe6['cond']:>10.3e} pe={pe6['is_pe']!s:<6}")
    return report


# ==========================================================================
# 2. Innovation autocorrelation + spectrum, stim-ON vs no-stim
# ==========================================================================

def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(x)
    xm = x - x.mean()
    denom = np.sum(xm ** 2)
    return np.array([np.sum(xm[:n - lag] * xm[lag:]) / denom for lag in range(1, max_lag + 1)])


def periodogram(x: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unaveraged, unsmoothed) periodogram -- kept only for the visual
    ACF/PSD plots. NOT used for peak detection: a single-segment periodogram
    bin is a chi-square(2) random variable (Ljung/standard spectral-estimation
    fact) -- the max over many such bins routinely looks like a huge "peak"
    purely from bin-to-bin variance, with no averaging to suppress it. Any
    peak CLAIM must come from welch_psd below instead."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=Ts)
    X = np.fft.rfft(x - x.mean())
    psd = (np.abs(X) ** 2) / n
    return freqs, psd


def welch_psd(x: np.ndarray, Ts: float, n_segments: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Welch-averaged PSD: split into n_segments overlapping (50%) Hann-
    windowed segments, average their periodograms -- trades frequency
    resolution for a statistically consistent estimate (variance ~1/n_segments
    of the raw periodogram), the standard fix for the raw periodogram's
    inconsistency (Welch 1967; Ljung ch.6 for the system-ID framing)."""
    from scipy.signal import welch
    nperseg = max(16, len(x) // ((n_segments + 1) // 2))
    freqs, psd = welch(x - x.mean(), fs=1.0 / Ts, nperseg=min(nperseg, len(x)),
                        noverlap=min(nperseg, len(x)) // 2)
    return freqs, psd


def innovation_spectrum_diagnostic(u: np.ndarray, innov: np.ndarray, Ts: float,
                                    stim_off_tol_frac: float, path: Path,
                                    title: str) -> dict:
    thresh = u.min() * stim_off_tol_frac
    stim_on = u > thresh
    stim_off = ~stim_on
    n_on, n_off = int(stim_on.sum()), int(stim_off.sum())

    max_lag = min(60, len(innov) - 1)
    lags = np.arange(1, max_lag + 1) * Ts

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7))

    acf_all = acf(innov, max_lag)
    conf = 1.96 / np.sqrt(len(innov))
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axhline(conf, color="tab:red", lw=0.5, ls="--")
    ax1.axhline(-conf, color="tab:red", lw=0.5, ls="--")
    ax1.plot(lags, acf_all, lw=1)
    ax1.set_title("Innovation ACF (all samples)")
    ax1.set_xlabel("lag (s)"); ax1.set_ylabel("ACF")

    acf_on = acf(innov[stim_on], min(max_lag, n_on - 1)) if n_on > 20 else None
    acf_off = acf(innov[stim_off], min(max_lag, n_off - 1)) if n_off > 20 else None
    ax2.axhline(0, color="k", lw=0.5)
    if acf_on is not None:
        ax2.plot(np.arange(1, len(acf_on) + 1) * Ts, acf_on, lw=1, label=f"stim-ON (n={n_on})")
    if acf_off is not None:
        ax2.plot(np.arange(1, len(acf_off) + 1) * Ts, acf_off, lw=1, label=f"no-stim (n={n_off})")
    ax2.axhline(conf, color="gray", lw=0.5, ls="--")
    ax2.axhline(-conf, color="gray", lw=0.5, ls="--")
    ax2.set_title("Innovation ACF, stim-ON vs no-stim")
    ax2.set_xlabel("lag (s)"); ax2.legend(fontsize=8)

    freqs, psd = periodogram(innov, Ts)
    fw, pw = welch_psd(innov, Ts)
    ax3.semilogy(freqs[1:], psd[1:], lw=0.6, color="lightgray", label="raw periodogram")
    ax3.semilogy(fw, pw, lw=1.5, color="tab:blue", label="Welch-averaged")
    ax3.set_title("Innovation power spectrum (all samples)")
    ax3.set_xlabel("frequency (Hz)"); ax3.set_ylabel("PSD"); ax3.legend(fontsize=7)

    fw_on = pw_on = fw_off = pw_off = None
    if n_on > 60:
        fw_on, pw_on = welch_psd(innov[stim_on], Ts)
        ax4.semilogy(fw_on, pw_on, lw=1, label=f"stim-ON, Welch (n={n_on})")
    if n_off > 60:
        fw_off, pw_off = welch_psd(innov[stim_off], Ts)
        ax4.semilogy(fw_off, pw_off, lw=1, label=f"no-stim, Welch (n={n_off})")
    ax4.set_title("Innovation power spectrum (Welch-averaged), stim-ON vs no-stim")
    ax4.set_xlabel("frequency (Hz)"); ax4.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    # Welch-based peak finding: the averaging (n_segments overlapping windows)
    # gives each PSD estimate ~2*n_segments degrees of freedom instead of the
    # raw periodogram's 2 -- a peak surviving this averaging is much less
    # likely to be single-bin variance (Welch 1967).
    def dominant_peak(freqs, psd):
        if psd is None or len(psd) < 5:
            return None, None
        body, fbody = psd[1:], freqs[1:]
        i = int(np.argmax(body))
        ratio = float(body[i] / np.median(body))
        return float(fbody[i]), ratio

    peak_f_on, peak_ratio_on = dominant_peak(fw_on, pw_on)
    peak_f_off, peak_ratio_off = dominant_peak(fw_off, pw_off)

    return {"n_on": n_on, "n_off": n_off,
            "peak_f_on": peak_f_on, "peak_ratio_on": peak_ratio_on,
            "peak_f_off": peak_f_off, "peak_ratio_off": peak_ratio_off}


def run_innovation_diagnostics(cfg, dual_cfg, n_epochs: int = 10,
                                out_dir: Path = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_logparam") -> dict:
    recordings = load_all_recordings(cfg)
    ar_fits = fit_ar_beta_model_per_recording(recordings, cfg)
    summary = {}
    for name in cfg.system_id_files:
        key = name.replace(".csv", "")
        u, y = recordings[name]
        f = ar_fits[name]
        res = run_identification_logparam(u, y, f["phi_coeffs"], f["sigma2_ar"],
                                           0.05 * f["sigma2_ar"], 0.5 * float(np.var(y)),
                                           cfg, dual_cfg, n_epochs=n_epochs)
        rec_dir = out_dir / key
        rec_dir.mkdir(parents=True, exist_ok=True)
        diag = innovation_spectrum_diagnostic(
            u, res["innov"], cfg.Ts, cfg.stim_off_tol_frac,
            rec_dir / "innovation_spectrum.png",
            title=f"{key}: innovation ACF/spectrum diagnostic (whiteness_exceed={res['whiteness_frac_exceed']:.2f})")
        diag["whiteness_frac_exceed"] = res["whiteness_frac_exceed"]
        summary[key] = diag
        print(f"{key}: peak(ON)={diag['peak_f_on']} Hz x{diag['peak_ratio_on']}, "
              f"peak(OFF)={diag['peak_f_off']} Hz x{diag['peak_ratio_off']}, "
              f"whiteness_exceed={diag['whiteness_frac_exceed']:.2f}")
    return summary


if __name__ == "__main__":
    print("=== persistent-excitation order check ===")
    pe_report(CFG)
    print("\n=== innovation spectrum diagnostic ===")
    run_innovation_diagnostics(CFG, DUAL_CFG)
