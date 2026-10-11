"""V1 (dual KF+RIV) pipeline health diagnostics on the real system_id recordings.

  1. AR disturbance-model order: whether [seg_ar_order_min, seg_ar_order_max]
     = [3, 10] is the right search range, or AIC/whiteness keep improving
     outside it.
  2. KF/RIV tuning knobs (lambda_ff, p0_kf_state_var, p0_rls, q_off_frac,
     gamma_pole, n_epochs): one-at-a-time sensitivity around the DualConfig
     defaults, tracking k/tau1/tau2 plus NIS and innovation whiteness.
  3. RIV instrument: auxiliary-model vs lagged-u, compared on the same
     metrics through the dense reference estimator.

Writes plots to riv_v1_diagnostics_output/ and prints a text summary.
"""
from __future__ import annotations
import contextlib
import dataclasses
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"
FMT = R.CFG.fig_format


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# AR order sweep

def sweep_ar_order(recordings: dict, cfg: R.Config, order_range=range(1, 16)) -> dict:
    all_outputs = np.concatenate([y for _, y in recordings.values()])
    epsilon = R.compute_epsilon_floor(all_outputs)
    thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac
    trim_samples = round(cfg.settle_time_s / cfg.Ts)
    min_len = cfg.seg_ar_order_max + 20
    seg_pool = {name: R.no_stim_segments_for_recording(u, y, thresh, trim_samples, min_len,
                                                         cfg.ar_log_domain, epsilon)
                for name, (u, y) in recordings.items()}

    results = {}
    for name in cfg.system_id_files:
        own = seg_pool[name]
        segments, donor = (own, None) if own else R._find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)
        rows = []
        for order in order_range:
            usable = [s for s in segments if len(s) - order >= order + 5]
            if not usable:
                continue
            phi, sigma2 = R.fit_ar_pooled(usable, order)
            resid_all = []
            for s in usable:
                Phi = np.array([s[kk - order:kk][::-1] for kk in range(order, len(s))])
                resid_all.append(s[order:] - Phi @ phi)
            resid_all = np.concatenate(resid_all)
            passed, frac = R.residual_whiteness_test(resid_all)
            aic = R.aic_score(sigma2, order, len(resid_all))
            stable, eigvals = R.ar_stability_check(phi)
            rows.append({"order": order, "aic": aic, "sigma2": sigma2,
                         "whiteness_pass": passed, "whiteness_frac": frac, "stable": stable})
        results[name] = {"rows": rows, "donor": donor,
                          "n_samples": sum(len(s) for s in segments)}
    return results


def plot_ar_order_sweep(results: dict, cfg: R.Config, path: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, name in zip(axes.flat, names):
        rows = results[name]["rows"]
        orders = [r["order"] for r in rows]
        aics = [r["aic"] for r in rows]
        fracs = [r["whiteness_frac"] for r in rows]
        ax2 = ax.twinx()
        ax.plot(orders, aics, "o-", color="tab:blue", label="AIC")
        ax2.plot(orders, fracs, "s--", color="tab:red", label="whiteness frac_exceed")
        ax2.axhline(R.WHITENESS_FAIL_FRAC, color="tab:red", lw=0.5, ls=":")
        ax.axvspan(cfg.seg_ar_order_min, cfg.seg_ar_order_max, color="tab:green", alpha=0.1)
        best_aic_order = orders[int(np.argmin(aics))]
        ax.axvline(best_aic_order, color="tab:blue", lw=0.7, ls=":")
        donor = results[name]["donor"]
        title = f"{name.removesuffix('.csv')}" + (f"\n(borrowed: {donor.removesuffix('.csv')})" if donor else "")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("AR order"); ax.set_ylabel("AIC", color="tab:blue")
        ax2.set_ylabel("whiteness frac_exceed", color="tab:red")
    fig.suptitle("AR order sweep: AIC + whiteness vs order (shaded = current search range "
                 f"[{cfg.seg_ar_order_min},{cfg.seg_ar_order_max}], dotted blue = AIC-best)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# KF/RIV parameter sweeps

SWEEP_GRIDS = {
    "lambda_ff": [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999],
    "p0_kf_state_var": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
    "p0_rls": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "q_off_frac": [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    "gamma_pole": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    "n_epochs": [1, 3, 5, 10, 20, 30, 50, 100],
}
LOG_SCALE_PARAMS = {"lambda_ff", "p0_kf_state_var", "p0_rls", "q_off_frac", "n_epochs"}


def sweep_kf_param(param: str, values: list, recordings: dict, ar_fits: dict,
                    cfg: R.Config, base_dual_cfg: R.DualConfig, q_eta: float) -> dict:
    metrics = {name: {"k": [], "tau1": [], "tau2": [], "nis_mean": [],
                       "whiteness_frac": [], "prls_a1a2_cond": []}
               for name in cfg.system_id_files}
    for v in values:
        dual_cfg = dataclasses.replace(base_dual_cfg, **{param: v})
        for name in cfg.system_id_files:
            u, y = recordings[name]
            f = ar_fits[name]
            R_this = 0.05 * f["sigma2_ar"]
            out = _quiet(R.run_identification, u, y, f["phi_coeffs"], f["sigma2_ar"],
                         R_this, q_eta, cfg, dual_cfg, FIG_DIR / "_sweep_tmp", make_plots=False)
            metrics[name]["k"].append(out["k"])
            metrics[name]["tau1"].append(out["tau1"])
            metrics[name]["tau2"].append(out["tau2"])
            metrics[name]["nis_mean"].append(out["nis_mean"])
            metrics[name]["whiteness_frac"].append(out["whiteness_frac_exceed"])
            metrics[name]["prls_a1a2_cond"].append(out["prls_a1a2_cond"])
    return metrics


def plot_kf_param_sweep(param: str, values: list, metrics: dict, default_value, path: Path) -> None:
    plotted = ["k", "tau1", "tau2", "nis_mean", "whiteness_frac", "prls_a1a2_cond"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    xscale = "log" if param in LOG_SCALE_PARAMS and min(values) > 0 else "linear"
    # lambda_ff clusters near 1, so plot 1-lambda on a log axis
    if param == "lambda_ff":
        x = [1.0 - v for v in values]
        default_x = 1.0 - default_value
        xlabel = "1 - lambda_ff (log)"
    else:
        x = values
        default_x = default_value
        xlabel = param

    for ax, metric in zip(axes.flat, plotted):
        for name, series in metrics.items():
            ax.plot(x, series[metric], marker="o", ms=3, lw=1, label=name.removesuffix(".csv"))
        if metric == "nis_mean":
            ax.axhline(1.0, color="k", lw=0.7, ls=":")
        if metric == "whiteness_frac":
            ax.axhline(R.WHITENESS_FAIL_FRAC, color="k", lw=0.7, ls=":")
        if metric == "prls_a1a2_cond":
            ax.set_yscale("log")
        ax.axvline(default_x, color="gray", lw=1, ls="--")
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel); ax.set_ylabel(metric)
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Sensitivity sweep: {param} (dashed gray = current default {default_value})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# RIV instrument comparison (dense reference)

def run_dense_epochs(u_norm, y, phi_c, Q_c, R_this, x0, P0_kf, theta0, P0_rls,
                      dual_cfg: R.DualConfig, n_eta, p, n_off, a1_prior, a2_prior,
                      instrument: str, n_epochs: int):
    theta_cur = theta0.copy()
    out = None
    for _ in range(n_epochs):
        out = R.reference_dual_estimator_dense(
            u_norm, y, phi_c, Q_c, R_this, x0.copy(), P0_kf.copy(), theta_cur, P0_rls.copy(),
            dual_cfg.lambda_ff, n_eta, p, n_off=n_off, gamma_pole=dual_cfg.gamma_pole,
            a1_prior=a1_prior, a2_prior=a2_prior, instrument=instrument)
        theta_cur = out[1][-1].copy()
    return out


def compare_instruments(recordings: dict, ar_fits: dict, cfg: R.Config,
                         dual_cfg: R.DualConfig, q_eta: float, n_epochs: int = 5) -> dict:
    results = {}
    for name in cfg.system_id_files:
        u, y = recordings[name]
        f = ar_fits[name]
        R_this = 0.05 * f["sigma2_ar"]
        setup = R.build_kf_setup(u, y, f["phi_coeffs"], f["sigma2_ar"], q_eta, cfg, dual_cfg)
        results[name] = {}
        for instrument in ("aux_model", "lagged_u"):
            x_hat, theta_hat, y_hat, Pkf_f, Prls_f, innov, S = run_dense_epochs(
                setup["u_norm"], y, setup["phi_c"], setup["Q_diag"], R_this,
                setup["x0"], setup["P0_kf"], setup["theta0"], setup["P0_rls"],
                dual_cfg, R.N_ETA, setup["p"], setup["n_off"],
                setup["a1_prior"], setup["a2_prior"], instrument, n_epochs)
            k, tau1, tau2 = R.theta_to_params_physical(theta_hat[-1], setup["u_scale"], cfg.Ts)
            _, nis_mean, nis_ok = R.nis_consistency_check(innov, S)
            white_pass, white_frac = R.residual_whiteness_test(innov)
            results[name][instrument] = {"k": k, "tau1": tau1, "tau2": tau2,
                                          "nis_mean": nis_mean, "whiteness_frac": white_frac}
    return results


def plot_instrument_comparison(results: dict, path: Path) -> None:
    names = list(results)
    metrics = ["k", "tau1", "tau2", "nis_mean", "whiteness_frac"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    x = np.arange(len(names))
    width = 0.35
    for ax, metric in zip(axes, metrics):
        aux = [results[n]["aux_model"][metric] for n in names]
        lag = [results[n]["lagged_u"][metric] for n in names]
        ax.bar(x - width / 2, aux, width, label="aux_model")
        ax.bar(x + width / 2, lag, width, label="lagged_u")
        ax.set_xticks(x); ax.set_xticklabels([n.removesuffix(".csv") for n in names],
                                              rotation=45, ha="right", fontsize=7)
        ax.set_title(metric)
    axes[0].legend(fontsize=8)
    fig.suptitle("RIV instrument comparison: auxiliary-model vs lagged-u (dense reference, "
                 f"{5} epochs)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# main

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg, dual_cfg = R.CFG, R.DUAL_CFG
    recordings = R.load_all_recordings(cfg)

    print("=== 1. AR order sweep ===")
    ar_sweep = sweep_ar_order(recordings, cfg)
    for name, res in ar_sweep.items():
        best = min(res["rows"], key=lambda r: r["aic"])
        in_range = cfg.seg_ar_order_min <= best["order"] <= cfg.seg_ar_order_max
        print(f"  {name}: AIC-best order={best['order']} "
              f"(in current search range [{cfg.seg_ar_order_min},{cfg.seg_ar_order_max}]: {in_range}), "
              f"whiteness_frac={best['whiteness_frac']:.3f}, stable={best['stable']}, "
              f"n_samples={res['n_samples']}")
    plot_ar_order_sweep(ar_sweep, cfg, FIG_DIR / f"ar_order_sweep.{FMT}")

    print("\nFitting per-recording AR models once (shared by all sweeps below)...")
    ar_fits = _quiet(R.fit_ar_beta_model_per_recording, recordings, cfg)
    y_all_system_id = np.concatenate([recordings[f][1] for f in cfg.system_id_files])
    q_eta = 0.5 * float(np.var(y_all_system_id))

    print("\n=== 2. KF/RIV parameter sweeps ===")
    default_values = {"lambda_ff": dual_cfg.lambda_ff, "p0_kf_state_var": dual_cfg.p0_kf_state_var,
                       "p0_rls": dual_cfg.p0_rls, "q_off_frac": dual_cfg.q_off_frac,
                       "gamma_pole": dual_cfg.gamma_pole, "n_epochs": dual_cfg.n_epochs}
    for param, values in SWEEP_GRIDS.items():
        print(f"  sweeping {param} over {values} ...")
        metrics = sweep_kf_param(param, values, recordings, ar_fits, cfg, dual_cfg, q_eta)
        plot_kf_param_sweep(param, values, metrics, default_values[param],
                            FIG_DIR / f"sweep_{param}.{FMT}")
        k_range = [max(m["k"]) - min(m["k"]) for m in metrics.values()]
        print(f"    max k-range across recordings: {max(k_range):.2f} "
              f"(recording-median default k spread: {np.median([abs(m['k'][values.index(default_values[param])]) for m in metrics.values()]):.2f})")

    print("\n=== 3. RIV instrument comparison ===")
    instr_results = compare_instruments(recordings, ar_fits, cfg, dual_cfg, q_eta, n_epochs=5)
    for name, res in instr_results.items():
        aux, lag = res["aux_model"], res["lagged_u"]
        print(f"  {name}: aux_model k={aux['k']:.2f} nis={aux['nis_mean']:.3f} | "
              f"lagged_u k={lag['k']:.2f} nis={lag['nis_mean']:.3f}")
    plot_instrument_comparison(instr_results, FIG_DIR / f"instrument_comparison.{FMT}")

    print(f"\nDiagnostics figures written under {FIG_DIR}")


if __name__ == "__main__":
    main()
