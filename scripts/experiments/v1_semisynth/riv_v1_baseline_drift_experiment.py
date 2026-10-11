
import contextlib
import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R
from riv_v1_semisynth_experiment import moving_block_bootstrap

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
K_TRUE, TAU1_TRUE, TAU2_TRUE = 80.0, 0.05, 0.25
N_SEEDS = 4
BLOCK_LEN = 15
MEAS_NOISE_STD = 0.02
N_ANCHORS = 5  # the 5 real no-stim segments on simple_controllers
DRIFT_FRACS = (0.0, 0.1, 0.2, 0.3, 0.5)  # fraction of std(real y); real ~0.18-0.28


def drift_baseline(N: int, target_std: float, n_anchors: int, rng: np.random.Generator) -> np.ndarray:
    """Piecewise-linear baseline through n_anchors random levels (anchor std =
    target_std), mirroring the handful of segment-mean levels seen in real data."""
    if target_std <= 0:
        return np.zeros(N)
    anchor_idx = np.linspace(0, N - 1, n_anchors)
    anchor_val = rng.normal(0.0, target_std, n_anchors)
    return np.interp(np.arange(N), anchor_idx, anchor_val)


def _check_drift_baseline() -> None:
    rng = np.random.default_rng(0)
    flat = drift_baseline(100, 0.0, 5, rng)
    assert np.all(flat == 0.0)
    drifted = drift_baseline(1000, 1.0, 5, rng)
    assert len(drifted) == 1000
    assert 0.1 < drifted.std() < 1.0  # interpolation smooths, so anchor std != trajectory std exactly


def main() -> None:
    _check_drift_baseline()
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
        eta_true = R.simulate_arx_recursion(theta_true, u)
        c0 = float(np.mean(y_real))
        std_y = float(np.std(y_real))
        R_heur = 0.05 * fit["sigma2_ar"]

        print(f"{name}: std(real y)={std_y:.4f}")
        for frac in DRIFT_FRACS:
            anchor_std = frac * std_y
            k_vals, white_vals, nis_vals, realized_fracs = [], [], [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    beta_bb = moving_block_bootstrap(segments, N, BLOCK_LEN, rng)
                    drift = drift_baseline(N, anchor_std, N_ANCHORS, rng)
                    realized_fracs.append(float(drift.std()) / std_y)
                    y_ss = c0 + drift + beta_bb - eta_true + rng.normal(0, MEAS_NOISE_STD, N)
                    res = R.run_identification(u, y_ss, fit["phi_coeffs"], fit["sigma2_ar"],
                                                 R_heur, 0.5 * float(np.var(y_ss)), cfg, dual_cfg,
                                                 FIG_DIR / "_t", make_plots=False)
                    k_vals.append(res["k"]); white_vals.append(res["whiteness_frac_exceed"])
                    nis_vals.append(res["nis_mean"])
            k_vals = np.array(k_vals)
            err = np.abs(k_vals - K_TRUE) / K_TRUE
            ok = int(np.sum((k_vals > 0) & (err < 0.3)))
            realized_frac = float(np.median(realized_fracs))
            rows.append({"name": name, "frac": realized_frac, "input_frac": frac,
                         "k_median": float(np.median(k_vals)),
                         "k_min": float(k_vals.min()), "k_max": float(k_vals.max()),
                         "ok": ok, "white_median": float(np.median(white_vals)),
                         "nis_median": float(np.median(nis_vals))})
            print(f"  input_frac={frac:.1f} (realized drift_std/std(y)={realized_frac:.3f}): "
                  f"k_med={np.median(k_vals):8.2f} range=[{k_vals.min():.1f},{k_vals.max():.1f}]  "
                  f"ok={ok}/{N_SEEDS}  white_med={np.median(white_vals):.3f}  nis_med={np.median(nis_vals):.3f}")

    print("\n=== Aggregate across all 6 recordings, by drift fraction ===")
    for input_frac in DRIFT_FRACS:
        subset = [r for r in rows if r["input_frac"] == input_frac]
        n_ok = sum(r["ok"] for r in subset)
        med_realized = np.median([r["frac"] for r in subset])
        med_k = np.median([r["k_median"] for r in subset])
        med_white = np.median([r["white_median"] for r in subset])
        med_nis = np.median([r["nis_median"] for r in subset])
        print(f"  input_frac={input_frac:.1f} (realized~{med_realized:.3f}): "
              f"ok={n_ok}/{len(subset) * N_SEEDS}  median(k_median)={med_k:7.2f}  "
              f"median(whiteness)={med_white:.3f}  median(NIS)={med_nis:.3f}")

    names = sorted(set(r["name"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in names:
        sub = [r for r in rows if r["name"] == name]
        fracs = [r["frac"] for r in sub]
        axes[0].plot(fracs, [r["k_median"] for r in sub], marker="o", label=name.removesuffix(".csv"))
        axes[1].plot(fracs, [r["white_median"] for r in sub], marker="o", label=name.removesuffix(".csv"))
        axes[2].plot(fracs, [r["nis_median"] for r in sub], marker="o", label=name.removesuffix(".csv"))
    axes[0].axhline(K_TRUE, color="k", lw=0.8, ls=":", label="truth")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("baseline drift std / std(real y)"); axes[0].set_ylabel("recovered k")
    axes[0].set_title("k vs drift magnitude")
    axes[1].axhline(R.WHITENESS_FAIL_FRAC, color="k", lw=0.7, ls=":")
    axes[1].axvspan(0.18, 0.28, color="gray", alpha=0.15)
    axes[1].set_xlabel("baseline drift std / std(real y)"); axes[1].set_ylabel("whiteness frac_exceed")
    axes[1].set_title("whiteness vs drift magnitude\n(gray band = measured real drift range)")
    axes[2].axhline(1.0, color="k", lw=0.7, ls=":")
    axes[2].axvspan(0.18, 0.28, color="gray", alpha=0.15)
    axes[2].set_xlabel("baseline drift std / std(real y)"); axes[2].set_ylabel("NIS mean")
    axes[2].set_title("NIS vs drift magnitude")
    axes[0].legend(fontsize=6)
    fig.suptitle("V1 on semi-synthetic (real beta, known eta) as baseline drift magnitude increases")
    fig.tight_layout()
    out_path = FIG_DIR / f"baseline_drift_sweep.{cfg.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
