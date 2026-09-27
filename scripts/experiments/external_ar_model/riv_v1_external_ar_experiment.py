"""
Experiment: does swapping in the higher-order AR shape derived from the
richer external dataset (derive_external_ar_model.py -> external_ar_model.npz,
order 12, downsampled exactly to our 25Hz) fix V1's innovation-whiteness
problem, compared to our own per-recording order-3 fit (forced low by our
own no-stim segments being too short to support anything higher -- see
INVESTIGATION.md Entry 10)?

The external model's absolute noise scale is NOT reused (different
recording, different beta amplitude calibration/units -- see
INVESTIGATION.md). Only its AR order/shape (phi_25hz) is borrowed; sigma2 is
recomputed by evaluating that shape's residual variance on OUR OWN
per-recording no-stim segment, so the noise level stays in our data's units.

This is a cross-dataset structural probe (different subject/session), not a
silent production swap -- main() is untouched. Report only.
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

CACHE_PATH = Path(__file__).parent / "external_ar_model.npz"
FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"


def rescaled_sigma2(phi: np.ndarray, segments: list[np.ndarray]) -> float:
    """Residual variance of a FIXED phi evaluated on OUR OWN segments --
    borrows the external shape, keeps our own noise scale/units."""
    order = len(phi)
    rows, targets = [], []
    for s in segments:
        if len(s) <= order:
            continue
        rows.extend(s[kk - order:kk][::-1] for kk in range(order, len(s)))
        targets.extend(s[order:])
    Phi, yv = np.array(rows), np.array(targets)
    resid = yv - Phi @ phi
    return float(np.sum(resid ** 2) / len(yv))


def main() -> None:
    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found -- run derive_external_ar_model.py first "
                          f"(needs the OneDrive external dataset mounted).")
    cache = np.load(CACHE_PATH)
    phi_ext = cache["phi_25hz"]
    print(f"external AR shape: order={len(phi_ext)} (fit on {cache['n_samples']} samples, "
          f"{cache['n_segments']} segments of a different 50Hz recording, downsampled exactly)")

    cfg, dual_cfg = R.CFG, R.DUAL_CFG
    recordings = R.load_all_recordings(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        ar_fits = R.fit_ar_beta_model_per_recording(recordings, cfg)

        # Rebuild the same own/borrowed segment pool fit_ar_beta_model_per_recording
        # used, so the external shape is rescaled against the SAME data our own
        # order-3 baseline was fit on (apples to apples).
        all_outputs = np.concatenate([y for _, y in recordings.values()])
        epsilon = R.compute_epsilon_floor(all_outputs)
        thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac
        trim_samples = round(cfg.settle_time_s / cfg.Ts)
        min_len = cfg.seg_ar_order_max + 20
        seg_pool = {name: R.no_stim_segments_for_recording(u, y, thresh, trim_samples, min_len,
                                                             cfg.ar_log_domain, epsilon)
                    for name, (u, y) in recordings.items()}

    rows = []
    for name in cfg.system_id_files:
        u, y = recordings[name]
        base = ar_fits[name]
        own = seg_pool[name]
        segments = own if own else R._find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)[0]
        sigma2_ext = rescaled_sigma2(phi_ext, segments)
        stable, eig = R.ar_stability_check(phi_ext)

        with contextlib.redirect_stdout(io.StringIO()):
            R_base = 0.05 * base["sigma2_ar"]
            q_eta = 0.5 * float(np.var(y))
            res_base = R.run_identification(u, y, base["phi_coeffs"], base["sigma2_ar"],
                                             R_base, q_eta, cfg, dual_cfg,
                                             FIG_DIR / "_t", make_plots=False)
            R_ext = 0.05 * sigma2_ext
            res_ext = R.run_identification(u, y, phi_ext, sigma2_ext,
                                            R_ext, q_eta, cfg, dual_cfg,
                                            FIG_DIR / "_t", make_plots=False)

        rows.append({"name": name, "order_base": base["order"], "order_ext": len(phi_ext),
                     "white_base": res_base["whiteness_frac_exceed"],
                     "white_ext": res_ext["whiteness_frac_exceed"],
                     "nis_base": res_base["nis_mean"], "nis_ext": res_ext["nis_mean"],
                     "k_base": res_base["k"], "k_ext": res_ext["k"]})
        print(f"{name}: order {base['order']:2d}->{ len(phi_ext):2d}  "
              f"whiteness_frac {res_base['whiteness_frac_exceed']:.3f}->{res_ext['whiteness_frac_exceed']:.3f}  "
              f"NIS {res_base['nis_mean']:.3f}->{res_ext['nis_mean']:.3f}  "
              f"k {res_base['k']:8.2f}->{res_ext['k']:8.2f}")

    n_improved = sum(1 for r in rows if r["white_ext"] < r["white_base"])
    print(f"\nwhiteness improved on {n_improved}/{len(rows)} recordings")

    names = [r["name"].removesuffix(".csv") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(names)); width = 0.35
    for ax, key, title in ((axes[0], "white", "innovation whiteness frac_exceed (lower=better)"),
                            (axes[1], "nis", "NIS mean (target 1.0)"),
                            (axes[2], "k", "k")):
        base_vals = [r[f"{key}_base"] for r in rows]
        ext_vals = [r[f"{key}_ext"] for r in rows]
        ax.bar(x - width / 2, base_vals, width, label="own AR(3)")
        ax.bar(x + width / 2, ext_vals, width, label=f"external shape AR({len(phi_ext)})")
        if key == "white":
            ax.axhline(R.WHITENESS_FAIL_FRAC, color="k", lw=0.7, ls=":")
        if key == "nis":
            ax.axhline(1.0, color="k", lw=0.7, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("Own order-3 AR (data-starved) vs external-shape AR (richer, different-subject "
                 "no-stim data, rescaled to our units) -- cross-dataset probe, not a production swap")
    fig.tight_layout()
    out_path = FIG_DIR / f"external_ar_comparison.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
