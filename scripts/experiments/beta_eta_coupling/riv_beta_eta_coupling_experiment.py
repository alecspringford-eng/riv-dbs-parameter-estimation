"""
Experiment: SEMISYNTHETIC_VS_REAL_REPORT.md next step #2. Does stimulation
change beta's OWN dynamics (variance, AR residual, spectral content), or is
the additive split y=beta-eta+offset (assumed by V1/V2/V3 alike) actually
correct and beta is unaffected by u?

We cannot observe beta directly during stim-ON (y=beta-eta+offset, only the
sum is measured). Proxy: fit V2 (riv_v2_greybox, output-error, no explicit
beta model to bias the answer) to the WHOLE recording, then read off
beta_proxy = y - (c0 - eta_hat) at every sample -- valid during both stim-ON
and no-stim periods, since V2 does not need a beta submodel. Compare
beta_proxy's variance, AR(3) residual variance, and PSD shape between
stim-ON and no-stim samples within the same recording.

CAVEAT (stated up front, not just in the conclusion): this is not
model-free. If V2's eta model is itself wrong (plausible -- that is what
this whole investigation is about), some of what looks like "beta changes
under stim" could actually be "the eta model's error looks different under
different u regimes." Treat this as suggestive, not definitive -- see
INVESTIGATION.md Entry 20 confidence note.

main() untouched -- report only.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v2"))
import riv_dual_estimation as R
import riv_v2_greybox as V2

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"


def welch_like_psd(x: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Same simple rFFT periodogram pattern as plot_input_excitation (no
    scipy.signal dependency, consistent with the rest of this codebase)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=Ts)
    X = np.fft.rfft(x - x.mean())
    psd = (np.abs(X) ** 2) / n
    return freqs, psd


def main() -> None:
    cfg = R.CFG
    recordings = R.load_all_recordings(cfg)
    thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, cfg.system_id_files):
        u, y = recordings[name]
        with contextlib.redirect_stdout(io.StringIO()):
            f = V2.identify(u, y)
        eta_hat = V2._eta_cascade(f.k, f.taus, u)
        beta_proxy = y - (f.c0 - eta_hat)

        stim_on = u > thresh
        no_stim = ~stim_on
        n_on, n_off = int(stim_on.sum()), int(no_stim.sum())

        var_on = float(np.var(beta_proxy[stim_on])) if n_on > 20 else float("nan")
        var_off = float(np.var(beta_proxy[no_stim])) if n_off > 20 else float("nan")

        ar_var_on = ar_var_off = float("nan")
        if n_on > 20:
            _, ar_var_on, _ = R.fit_ar_ols(beta_proxy[stim_on] - beta_proxy[stim_on].mean(), 3)
        if n_off > 20:
            _, ar_var_off, _ = R.fit_ar_ols(beta_proxy[no_stim] - beta_proxy[no_stim].mean(), 3)

        rows.append({"name": name, "n_on": n_on, "n_off": n_off,
                     "var_on": var_on, "var_off": var_off, "var_ratio": var_on / var_off,
                     "ar_var_on": ar_var_on, "ar_var_off": ar_var_off,
                     "ar_var_ratio": ar_var_on / ar_var_off, "uy_corr": f.uy_corr})
        print(f"{name}: n_on={n_on} n_off={n_off}  var(beta_proxy) on/off = "
              f"{var_on:.4e}/{var_off:.4e} (ratio {var_on/var_off:.2f})  "
              f"AR(3)-resid-var on/off = {ar_var_on:.4e}/{ar_var_off:.4e} "
              f"(ratio {ar_var_on/ar_var_off:.2f})  [V2 uy_corr={f.uy_corr:.3f}]")

        freqs_on, psd_on = welch_like_psd(beta_proxy[stim_on], cfg.Ts) if n_on > 20 else (None, None)
        freqs_off, psd_off = welch_like_psd(beta_proxy[no_stim], cfg.Ts) if n_off > 20 else (None, None)
        if freqs_on is not None:
            ax.semilogy(freqs_on[1:], psd_on[1:], lw=1, label="stim-ON", color="tab:red")
        if freqs_off is not None:
            ax.semilogy(freqs_off[1:], psd_off[1:], lw=1, label="no-stim", color="tab:blue")
        ax.set_title(name.removesuffix(".csv"), fontsize=9)
        ax.set_xlabel("Hz"); ax.set_ylabel("PSD(beta_proxy)")

    axes.flat[0].legend(fontsize=7)
    fig.suptitle("beta_proxy = y - V2 eta_hat: spectral content, stim-ON vs no-stim "
                 "(caveat: proxy depends on V2's eta model being correct)")
    fig.tight_layout()
    out_path = FIG_DIR / f"beta_eta_coupling.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)

    var_ratios = [r["var_ratio"] for r in rows if np.isfinite(r["var_ratio"])]
    ar_ratios = [r["ar_var_ratio"] for r in rows if np.isfinite(r["ar_var_ratio"])]
    print(f"\nvar(beta_proxy) on/off ratio across recordings: "
          f"median={np.median(var_ratios):.2f}, range=[{min(var_ratios):.2f},{max(var_ratios):.2f}]")
    print(f"AR(3)-resid-var on/off ratio across recordings: "
          f"median={np.median(ar_ratios):.2f}, range=[{min(ar_ratios):.2f},{max(ar_ratios):.2f}]")
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
