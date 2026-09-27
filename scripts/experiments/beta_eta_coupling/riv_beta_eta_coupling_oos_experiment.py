"""
Experiment: SEMISYNTHETIC_VS_REAL_REPORT.md next step #1 (Entry 20's
de-confounding follow-up). Entry 20 found beta_proxy variance drops to
~40% (AR-resid ~24%) during stim-ON vs no-stim, using beta_proxy = y -
(c0 - eta_hat) with eta_hat from a V2 fit on the WHOLE recording. Caveat
raised there: V2's eta fit could mechanically absorb stim-ON residual
shape from the very same segments being analyzed, faking the appearance of
"beta changes under stim" without beta actually changing.

De-confounding design: split each recording in half by TIME (not by
segment, to avoid cascade-state discontinuities), fit V2's eta plant
(k, taus, c0) on one half only, then use those FIXED, held-out-blind
parameters to simulate eta_hat on the OTHER half (fresh cascade state --
negligible transient since tau ~0.05-0.25s = 1-6 samples). beta_proxy on
the test half is now computed from parameters that never saw that half's
data, so it cannot have absorbed that half's specific residual shape.
Both directions (first-half-fits-second, second-half-fits-first) are run
and pooled for more power.

Residual caveat this does NOT eliminate: if the plant is genuinely
non-stationary (Entry 19, inconclusive), a fixed-elsewhere fit predicting
a different time window will show extra prediction error for THAT reason,
not necessarily beta-eta coupling. Both this experiment and Entry 19
together bound, but don't fully separate, that possibility.

main() untouched -- report only. See INVESTIGATION.md Entry 23.
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


def half_split_ratios(u_fit, y_fit, u_test, y_test, thresh, min_n=20):
    """Fit V2 on (u_fit,y_fit), predict eta on u_test (fresh state), return
    beta_proxy on/off variance and AR(3)-residual-variance ratios computed
    ONLY on the test half."""
    with contextlib.redirect_stdout(io.StringIO()):
        f = V2.identify(u_fit, y_fit)
    eta_hat_test = V2._eta_cascade(f.k, f.taus, u_test)
    beta_proxy = y_test - (f.c0 - eta_hat_test)

    stim_on = u_test > thresh
    no_stim = ~stim_on
    n_on, n_off = int(stim_on.sum()), int(no_stim.sum())
    if n_on < min_n or n_off < min_n:
        return None

    var_on = float(np.var(beta_proxy[stim_on]))
    var_off = float(np.var(beta_proxy[no_stim]))
    _, ar_var_on, _ = R.fit_ar_ols(beta_proxy[stim_on] - beta_proxy[stim_on].mean(), 3)
    _, ar_var_off, _ = R.fit_ar_ols(beta_proxy[no_stim] - beta_proxy[no_stim].mean(), 3)
    return {"n_on": n_on, "n_off": n_off, "var_ratio": var_on / var_off,
            "ar_var_ratio": ar_var_on / ar_var_off, "uy_corr_fit": f.uy_corr,
            "k_fit": f.k, "taus_fit": f.taus}


def main() -> None:
    cfg = R.CFG
    recordings = R.load_all_recordings(cfg)
    thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac

    rows = []
    for name in cfg.system_id_files:
        u, y = recordings[name]
        N = len(u)
        mid = N // 2
        u1, y1, u2, y2 = u[:mid], y[:mid], u[mid:], y[mid:]

        r_1to2 = half_split_ratios(u1, y1, u2, y2, thresh)
        r_2to1 = half_split_ratios(u2, y2, u1, y1, thresh)

        print(f"{name}:")
        if r_1to2 is not None:
            print(f"  fit on 1st half -> predict 2nd half: n_on={r_1to2['n_on']} n_off={r_1to2['n_off']}  "
                  f"var_ratio={r_1to2['var_ratio']:.3f}  ar_var_ratio={r_1to2['ar_var_ratio']:.3f}  "
                  f"(fit uy_corr={r_1to2['uy_corr_fit']:.3f}, k={r_1to2['k_fit']:.1f})")
        else:
            print("  fit on 1st half -> predict 2nd half: insufficient no-stim/stim-on samples, skipped")
        if r_2to1 is not None:
            print(f"  fit on 2nd half -> predict 1st half: n_on={r_2to1['n_on']} n_off={r_2to1['n_off']}  "
                  f"var_ratio={r_2to1['var_ratio']:.3f}  ar_var_ratio={r_2to1['ar_var_ratio']:.3f}  "
                  f"(fit uy_corr={r_2to1['uy_corr_fit']:.3f}, k={r_2to1['k_fit']:.1f})")
        else:
            print("  fit on 2nd half -> predict 1st half: insufficient no-stim/stim-on samples, skipped")

        for direction, r in (("1to2", r_1to2), ("2to1", r_2to1)):
            if r is not None:
                rows.append({"name": name, "direction": direction, **r})

    if not rows:
        print("\nNo recording had enough held-out no-stim samples in both halves -- "
              "out-of-sample test inconclusive by data scarcity, not by the coupling question.")
        return

    var_ratios = [r["var_ratio"] for r in rows]
    ar_ratios = [r["ar_var_ratio"] for r in rows]
    print(f"\n=== Pooled out-of-sample results ({len(rows)} fit/predict directions "
          f"across {len(set(r['name'] for r in rows))} recordings) ===")
    print(f"var(beta_proxy) on/off ratio: median={np.median(var_ratios):.3f}, "
          f"range=[{min(var_ratios):.3f},{max(var_ratios):.3f}]")
    print(f"AR(3)-resid-var on/off ratio: median={np.median(ar_ratios):.3f}, "
          f"range=[{min(ar_ratios):.3f},{max(ar_ratios):.3f}]")
    print("(Entry 20 in-sample reference: var_ratio median=0.40 [0.37,0.51], "
          "ar_var_ratio median=0.24 [0.20,0.31])")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    names_seen = sorted(set(r["name"] for r in rows))
    x = np.arange(len(names_seen))
    for i, metric, title in ((0, "var_ratio", "var(beta_proxy) on/off, out-of-sample"),
                              (1, "ar_var_ratio", "AR(3)-resid-var on/off, out-of-sample")):
        for direction, marker, color in (("1to2", "o", "tab:blue"), ("2to1", "s", "tab:orange")):
            xs, ys = [], []
            for r in rows:
                if r["direction"] == direction:
                    xs.append(names_seen.index(r["name"]))
                    ys.append(r[metric])
            ax[i].scatter(xs, ys, marker=marker, color=color, label=f"fit {direction}", s=60)
        ax[i].axhline(1.0, color="k", lw=0.7, ls=":")
        ax[i].set_xticks(x); ax[i].set_xticklabels([n.removesuffix(".csv") for n in names_seen],
                                                     rotation=45, ha="right", fontsize=8)
        ax[i].set_title(title, fontsize=9); ax[i].legend(fontsize=7)
    fig.suptitle("Out-of-sample beta-eta coupling test: eta plant fit on one half, "
                 "beta_proxy measured on the OTHER (held-out) half")
    fig.tight_layout()
    out_path = FIG_DIR / f"beta_eta_coupling_oos.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
