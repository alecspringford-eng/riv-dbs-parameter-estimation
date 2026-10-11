"""Does V1 recover k on a semi-synthetic construction?

eta comes from the known 2-lag ARX plant (as in the S4.8 fully-synthetic
self-check in riv_dual_estimation.py), but beta is a moving-block bootstrap of
the recording's own no-stim segments (the pool fit_ar_beta_model_per_recording
uses) instead of an AR(p) draw. That isolates beta's generative process
against the fully-synthetic control: the bootstrap keeps whatever non-AR
structure (periodicity, non-Gaussianity) real beta has.

If semi-synthetic recovers k about as well as fully-synthetic, real beta's
non-AR statistics do not explain the real-data gap, and the cause is on the
eta/plant side. If it fails about as badly as real data, beta's own
statistics break the KF/RIV separation regardless of the plant, pointing at
the disturbance model or architecture.

Report only; main() is untouched.
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
K_TRUE, TAU1_TRUE, TAU2_TRUE = 62.11, 0.05, 0.25
N_SEEDS = 6
BLOCK_LEN = 15
MEAS_NOISE_STD = 0.02


def moving_block_bootstrap(segments: list[np.ndarray], n_samples: int, block_len: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Concatenate randomly placed, possibly overlapping blocks of length
    block_len drawn with replacement from the pooled real segments until
    n_samples is reached. Keeps real beta's short-range statistics without
    imposing an AR(p) form."""
    pool = np.concatenate(segments)
    block_len = min(block_len, len(pool))
    chunks, total = [], 0
    while total < n_samples:
        start = rng.integers(0, len(pool) - block_len + 1)
        chunk = pool[start:start + block_len]
        chunks.append(chunk)
        total += len(chunk)
    return np.concatenate(chunks)[:n_samples]


def _check_bootstrap() -> None:
    rng = np.random.default_rng(0)
    pool_segs = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([10.0, 20.0, 30.0])]
    out = moving_block_bootstrap(pool_segs, n_samples=17, block_len=3, rng=rng)
    assert len(out) == 17
    assert set(np.unique(out)) <= set(np.concatenate(pool_segs))


def main() -> None:
    _check_bootstrap()
    cfg, dual_cfg = R.CFG, R.DUAL_CFG
    recordings = R.load_all_recordings(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        ar_fits = R.fit_ar_beta_model_per_recording(recordings, cfg)

        all_outputs = np.concatenate([y for _, y in recordings.values()])
        epsilon = R.compute_epsilon_floor(all_outputs)
        thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac
        trim_samples = round(cfg.settle_time_s / cfg.Ts)
        min_len = cfg.seg_ar_order_max + 20
        seg_pool = {name: R.no_stim_segments_for_recording(u, y, thresh, trim_samples, min_len,
                                                             cfg.ar_log_domain, epsilon)
                    for name, (u, y) in recordings.items()}

    theta_true = R.params_to_theta(K_TRUE, TAU1_TRUE, TAU2_TRUE, cfg.Ts)

    rows = []
    for name in cfg.system_id_files:
        u, y_real = recordings[name]
        N = len(u)
        fit = ar_fits[name]
        own = seg_pool[name]
        segments = own if own else R._find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)[0]
        n_pool = sum(len(s) for s in segments)
        eta_true = R.simulate_arx_recursion(theta_true, u)
        c0 = float(np.mean(y_real))
        R_heur = 0.05 * fit["sigma2_ar"]

        with contextlib.redirect_stdout(io.StringIO()):
            q_eta_real = 0.5 * float(np.var(y_real))
            res_real = R.run_identification(u, y_real, fit["phi_coeffs"], fit["sigma2_ar"],
                                              R_heur, q_eta_real, cfg, dual_cfg,
                                              FIG_DIR / "_t", make_plots=False)

        k_fs, k_ss, white_fs, white_ss, nis_fs, nis_ss = [], [], [], [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                # fully-synthetic control: AR(fit) draw
                A_beta, C_beta = R.ar_to_state_space(fit["phi_coeffs"])
                p = len(fit["phi_coeffs"])
                xb = np.zeros(p)
                beta_ar = np.zeros(N)
                for kk in range(N):
                    beta_ar[kk] = (C_beta @ xb)[0]
                    xb = A_beta @ xb
                    xb[0] += rng.normal(0, np.sqrt(fit["sigma2_ar"]))
                y_fs = c0 + beta_ar - eta_true + rng.normal(0, MEAS_NOISE_STD, N)
                res_fs = R.run_identification(u, y_fs, fit["phi_coeffs"], fit["sigma2_ar"],
                                                R_heur, 0.5 * float(np.var(y_fs)), cfg, dual_cfg,
                                                FIG_DIR / "_t", make_plots=False)
                k_fs.append(res_fs["k"]); white_fs.append(res_fs["whiteness_frac_exceed"])
                nis_fs.append(res_fs["nis_mean"])

                # semi-synthetic: moving-block bootstrap of real no-stim data
                beta_bb = moving_block_bootstrap(segments, N, BLOCK_LEN, rng)
                y_ss = c0 + beta_bb - eta_true + rng.normal(0, MEAS_NOISE_STD, N)
                res_ss = R.run_identification(u, y_ss, fit["phi_coeffs"], fit["sigma2_ar"],
                                                R_heur, 0.5 * float(np.var(y_ss)), cfg, dual_cfg,
                                                FIG_DIR / "_t", make_plots=False)
                k_ss.append(res_ss["k"]); white_ss.append(res_ss["whiteness_frac_exceed"])
                nis_ss.append(res_ss["nis_mean"])

        k_fs, k_ss = np.array(k_fs), np.array(k_ss)
        err_fs, err_ss = np.abs(k_fs - K_TRUE) / K_TRUE, np.abs(k_ss - K_TRUE) / K_TRUE
        rows.append({
            "name": name, "n_pool": n_pool, "k_real": res_real["k"],
            "white_real": res_real["whiteness_frac_exceed"], "nis_real": res_real["nis_mean"],
            "k_fs_median": float(np.median(k_fs)), "ok_fs": int(np.sum((k_fs > 0) & (err_fs < 0.3))),
            "white_fs_median": float(np.median(white_fs)), "nis_fs_median": float(np.median(nis_fs)),
            "k_ss_median": float(np.median(k_ss)), "ok_ss": int(np.sum((k_ss > 0) & (err_ss < 0.3))),
            "white_ss_median": float(np.median(white_ss)), "nis_ss_median": float(np.median(nis_ss)),
        })
        r = rows[-1]
        print(f"{name}: pool={n_pool:4d} samples")
        print(f"  fully-real:              k={r['k_real']:9.2f}  white_frac={r['white_real']:.3f}  "
              f"NIS={r['nis_real']:.3f}")
        print(f"  fully-synthetic (AR beta): k_med={r['k_fs_median']:8.2f} (ok {r['ok_fs']}/{N_SEEDS})  "
              f"white_frac_med={r['white_fs_median']:.3f}  NIS_med={r['nis_fs_median']:.3f}")
        print(f"  semi-synthetic (real beta): k_med={r['k_ss_median']:8.2f} (ok {r['ok_ss']}/{N_SEEDS})  "
              f"white_frac_med={r['white_ss_median']:.3f}  NIS_med={r['nis_ss_median']:.3f}")

    n_ok_fs = sum(r["ok_fs"] for r in rows)
    n_ok_ss = sum(r["ok_ss"] for r in rows)
    print(f"\ntotal ok (k>0, err<30% of truth={K_TRUE:.0f}): "
          f"fully-synthetic {n_ok_fs}/{len(rows) * N_SEEDS}, "
          f"semi-synthetic (real beta) {n_ok_ss}/{len(rows) * N_SEEDS}")

    names = [r["name"].removesuffix(".csv") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(names)); width = 0.28
    ax = axes[0]
    ax.bar(x - width, [r["k_fs_median"] for r in rows], width, label="fully-synthetic (AR beta)")
    ax.bar(x, [r["k_ss_median"] for r in rows], width, label="semi-synthetic (real-beta bootstrap)")
    ax.bar(x + width, [r["k_real"] for r in rows], width, label="fully real y")
    ax.axhline(K_TRUE, color="k", lw=0.8, ls=":", label=f"truth k={K_TRUE:.0f} (synthetic conditions)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("k"); ax.legend(fontsize=7); ax.set_title("recovered k")

    ax = axes[1]
    ax.bar(x - width, [r["white_fs_median"] for r in rows], width, label="fully-synthetic")
    ax.bar(x, [r["white_ss_median"] for r in rows], width, label="semi-synthetic")
    ax.bar(x + width, [r["white_real"] for r in rows], width, label="fully real")
    ax.axhline(R.WHITENESS_FAIL_FRAC, color="k", lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("innovation whiteness frac_exceed"); ax.legend(fontsize=7)
    ax.set_title("innovation whiteness (lower=better)")

    fig.suptitle("V1: fully-synthetic vs semi-synthetic (real-beta bootstrap, known eta) vs fully-real")
    fig.tight_layout()
    out_path = FIG_DIR / f"semisynth_comparison.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
