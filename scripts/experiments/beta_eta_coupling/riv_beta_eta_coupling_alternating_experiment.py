"""
Experiment: Entry 23's own flagged next step. The half-split OOS test
(riv_beta_eta_coupling_oos_experiment.py) de-confounds "model absorbed the
test data's residual shape" but NOT "the plant genuinely drifted between
the two halves" (Entry 19's non-stationarity question, itself
inconclusive) -- a fixed-elsewhere fit predicting a temporally distant,
contiguous half would show extra error from drift too, indistinguishable
from coupling in that design.

This design controls for both at once: split each recording into many
short, fixed-length blocks spanning its FULL duration, alternate blocks
into FIT and TEST sets (odd/even), fit V2's eta plant (order 1, single
lag -- keeps the block-aware refit simple) on the FIT blocks only (each
block simulated with a FRESH cascade state, consistent with how it's
later applied), then measure beta_proxy on the TEST blocks (also fresh
state per block). Because FIT and TEST blocks are interleaved throughout
the whole recording rather than occupying separate contiguous halves, a
slow drift in the true plant affects both sets roughly equally -- it can
no longer concentrate in the held-out portion the way it could in the
half-split design. What survives this design is not explainable by either
confound raised so far.

Trade-off: order=1 only (not AIC order-selected per fit, to keep the
block-aware refit simple) and short per-block simulation windows (edge
transients at each block start are a larger fraction of a short block than
a long contiguous one) -- both make each individual block's eta fit
noisier than the whole-recording or half-split fits. The pooled statistic
across many blocks is what carries the signal, not any single block.

main() untouched -- report only. See INVESTIGATION.md Entry 25.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v2"))
import riv_dual_estimation as R
import riv_v2_greybox as V2

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
BLOCK_LEN = 150  # ~6s at 25Hz; gives 18-40 blocks per recording (2745-6120 samples)


def make_blocks(N: int, block_len: int) -> list[tuple[int, int]]:
    return [(s, min(s + block_len, N)) for s in range(0, N, block_len) if min(s + block_len, N) - s >= 20]


def _check_make_blocks() -> None:
    b = make_blocks(2745, 150)
    assert len(b) >= 15
    assert b[-1][1] == 2745


def fit_blockwise_eta(u_blocks: list[np.ndarray], y_blocks: list[np.ndarray]) -> tuple[float, float, float, float]:
    """Order-1 grey-box OE fit, but eta is simulated PER BLOCK with fresh
    cascade state (V2._eta_cascade already resets prev=0 per call) instead
    of one continuous simulation -- matches how the fit is used at test
    time, and how held-out blocks will be simulated too."""
    y_all = np.concatenate(y_blocks)
    u_all = np.concatenate(u_blocks)
    p0 = np.array([np.log(60.0), np.log(0.1), float(y_all.mean())])

    def resid(p: np.ndarray) -> np.ndarray:
        k = np.exp(p[0]); tau = np.exp(p[1]); c0 = p[2]
        eta_hat = np.concatenate([V2._eta_cascade(k, (tau,), ub) for ub in u_blocks])
        return (c0 - eta_hat) - y_all

    sol = least_squares(resid, p0, method="lm")
    k, tau, c0 = float(np.exp(sol.x[0])), float(np.exp(sol.x[1])), float(sol.x[2])
    uy = float(np.corrcoef(u_all, y_all)[0, 1]) if np.std(u_all) > 0 else 0.0
    return k, tau, c0, uy


def alternating_ratios(u_fit_blocks, y_fit_blocks, u_test_blocks, y_test_blocks, thresh, min_n=20):
    k, tau, c0, uy = fit_blockwise_eta(u_fit_blocks, y_fit_blocks)
    eta_test = np.concatenate([V2._eta_cascade(k, (tau,), ub) for ub in u_test_blocks])
    u_test = np.concatenate(u_test_blocks)
    y_test = np.concatenate(y_test_blocks)
    beta_proxy = y_test - (c0 - eta_test)

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
            "ar_var_ratio": ar_var_on / ar_var_off, "uy_corr_fit": uy, "k_fit": k}


def main() -> None:
    _check_make_blocks()
    cfg = R.CFG
    recordings = R.load_all_recordings(cfg)
    thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac

    rows = []
    for name in cfg.system_id_files:
        u, y = recordings[name]
        blocks = make_blocks(len(u), BLOCK_LEN)
        u_blocks = [u[s:e] for s, e in blocks]
        y_blocks = [y[s:e] for s, e in blocks]
        fit_idx = list(range(0, len(blocks), 2))
        test_idx = list(range(1, len(blocks), 2))
        u_fit = [u_blocks[i] for i in fit_idx]; y_fit = [y_blocks[i] for i in fit_idx]
        u_test = [u_blocks[i] for i in test_idx]; y_test = [y_blocks[i] for i in test_idx]

        with contextlib.redirect_stdout(io.StringIO()):
            r_even_odd = alternating_ratios(u_fit, y_fit, u_test, y_test, thresh)
            r_odd_even = alternating_ratios(u_test, y_test, u_fit, y_fit, thresh)

        print(f"{name}: {len(blocks)} blocks ({len(fit_idx)} fit / {len(test_idx)} test, alternating)")
        for direction, r in (("even->odd", r_even_odd), ("odd->even", r_odd_even)):
            if r is not None:
                print(f"  fit {direction}: n_on={r['n_on']} n_off={r['n_off']}  "
                      f"var_ratio={r['var_ratio']:.3f}  ar_var_ratio={r['ar_var_ratio']:.3f}  "
                      f"(fit uy_corr={r['uy_corr_fit']:.3f}, k={r['k_fit']:.1f})")
                rows.append({"name": name, "direction": direction, **r})
            else:
                print(f"  fit {direction}: insufficient no-stim/stim-on samples in test blocks, skipped")

    if not rows:
        print("\nNo recording had enough held-out no-stim samples -- inconclusive by data scarcity.")
        return

    var_ratios = [r["var_ratio"] for r in rows]
    ar_ratios = [r["ar_var_ratio"] for r in rows]
    print(f"\n=== Pooled alternating-block results ({len(rows)} fit/test directions "
          f"across {len(set(r['name'] for r in rows))} recordings) ===")
    print(f"var(beta_proxy) on/off ratio: median={np.median(var_ratios):.3f}, "
          f"range=[{min(var_ratios):.3f},{max(var_ratios):.3f}]")
    print(f"AR(3)-resid-var on/off ratio: median={np.median(ar_ratios):.3f}, "
          f"range=[{min(ar_ratios):.3f},{max(ar_ratios):.3f}]")
    print("(Entry 20 in-sample reference: 0.40, 0.24; Entry 23 half-split OOS: 0.411, 0.268)")

    names_seen = sorted(set(r["name"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(names_seen))
    for i, metric, title in ((0, "var_ratio", "var(beta_proxy) on/off, alternating-block"),
                              (1, "ar_var_ratio", "AR(3)-resid-var on/off, alternating-block")):
        for direction, marker, color in (("even->odd", "o", "tab:blue"), ("odd->even", "s", "tab:orange")):
            xs, ys = [], []
            for r in rows:
                if r["direction"] == direction:
                    xs.append(names_seen.index(r["name"])); ys.append(r[metric])
            ax[i].scatter(xs, ys, marker=marker, color=color, label=f"fit {direction}", s=60)
        ax[i].axhline(1.0, color="k", lw=0.7, ls=":")
        ax[i].set_xticks(x); ax[i].set_xticklabels([n.removesuffix(".csv") for n in names_seen],
                                                     rotation=45, ha="right", fontsize=8)
        ax[i].set_title(title, fontsize=9); ax[i].legend(fontsize=7)
    fig.suptitle("Alternating-block beta-eta coupling test (controls for both the model-artifact\n"
                 "and non-stationarity confounds at once)")
    fig.tight_layout()
    out_path = FIG_DIR / f"beta_eta_coupling_alternating.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
