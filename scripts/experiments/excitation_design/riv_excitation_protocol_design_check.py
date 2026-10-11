"""Which property of a future recording protocol would close the PE gap?

Real u has near-unity lag-1 autocorrelation (0.98-0.997) despite changing
almost every sample: it is a rate-limited ramp, not a PRBS-like switching
signal. Its PE(4) Toeplitz condition number is 9.4e3-1.2e5
(riv_dual_estimation.pe_order_check) vs ~25-39 for the synthetic
IID-uniform u the estimator was validated on.

Candidate excitations are matched to each real recording's length N and
amplitude range [u.min(), u.max()] and scored with the same pe_order_check.
"""
from __future__ import annotations

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
from riv_dual_estimation import CFG, load_all_recordings, pe_order_check


def acf1(x: np.ndarray) -> float:
    xm = x - x.mean()
    return float(np.dot(xm[:-1], xm[1:]) / np.dot(xm, xm))


def make_iid_uniform(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lo, hi, n)


def make_prbs2(n: int, lo: float, hi: float, dwell: int, rng: np.random.Generator) -> np.ndarray:
    """2-level pseudo-random binary sequence, held for `dwell` samples per switch."""
    n_switches = n // dwell + 1
    levels = rng.choice([lo, hi], n_switches)
    return np.repeat(levels, dwell)[:n]


def make_aprbs(n: int, lo: float, hi: float, min_dwell: int, max_dwell: int,
                rng: np.random.Generator) -> np.ndarray:
    """Amplitude-modulated PRBS: independent uniform level held for a random
    dwell in [min_dwell, max_dwell] samples (Ljung ch.13; Nelles sec 17.5)."""
    out = np.empty(n)
    i = 0
    while i < n:
        dwell = rng.integers(min_dwell, max_dwell + 1)
        out[i:i + dwell] = rng.uniform(lo, hi)
        i += dwell
    return out[:n]


def make_multisine(n: int, lo: float, hi: float, Ts: float, freqs_hz: list[float],
                    rng: np.random.Generator) -> np.ndarray:
    """Sum of sinusoids at the given frequencies, random phase, rescaled to [lo,hi]."""
    t = np.arange(n) * Ts
    x = np.zeros(n)
    for f in freqs_hz:
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    x = (x - x.min()) / (x.max() - x.min())
    return lo + x * (hi - lo)


def main() -> None:
    rng = np.random.default_rng(CFG.seed)
    recordings = load_all_recordings(CFG)

    print(f"{'recording':<20}{'design':<22}{'N':>6}{'acf1':>8}{'PE4 cond':>14}{'PE6 cond':>14}")
    print("-" * 84)

    for name in CFG.system_id_files:
        u_real, _ = recordings[name]
        n = len(u_real)
        lo, hi = float(u_real.min()), float(u_real.max())

        real_pe4 = pe_order_check(u_real, 4)
        real_pe6 = pe_order_check(u_real, 6)
        print(f"{name:<20}{'REAL (as recorded)':<22}{n:>6}{acf1(u_real):>8.3f}"
              f"{real_pe4['cond']:>14.3e}{real_pe6['cond']:>14.3e}")

        candidates = {
            "IID uniform": make_iid_uniform(n, lo, hi, rng),
            "PRBS2 dwell=1": make_prbs2(n, lo, hi, 1, rng),
            "APRBS dwell 1-5": make_aprbs(n, lo, hi, 1, 5, rng),
            "APRBS dwell 5-25": make_aprbs(n, lo, hi, 5, 25, rng),
            "Multisine (0.5-5Hz,6 tones)": make_multisine(
                n, lo, hi, CFG.Ts, list(np.linspace(0.5, 5.0, 6)), rng),
        }
        for label, u_cand in candidates.items():
            pe4 = pe_order_check(u_cand, 4)
            pe6 = pe_order_check(u_cand, 6)
            print(f"{'':<20}{label:<22}{n:>6}{acf1(u_cand):>8.3f}"
                  f"{pe4['cond']:>14.3e}{pe6['cond']:>14.3e}")
        print()


if __name__ == "__main__":
    main()
