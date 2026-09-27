"""riv_sampling_rate_sweep_experiment.py -- what is the minimum sampling
rate fs at which the confirmed-working 50Hz semi-synthetic 2-pole
multisine setup (INVESTIGATION.md Entry 47/49, "adaptive" arm, 10/10 ok)
still recovers (k, tau1, tau2) reliably?

Base scenario: bit-for-bit Entry 49's "adaptive" arm (instrument_mode=
AUX_DECIMATED, M=1) -- k_true/tau1_true/tau2_true=(62.11,0.05,0.25),
perturb_frac=0.5, rate-limited multisine (u_max=2.5, du_max=1.5),
N_sim=80000 HELD FIXED across the whole sweep (per user direction: isolate
the pure sample-resolution effect rather than also cutting the amount of
data the estimator sees -- so simulated duration T=N_sim/fs grows as fs
drops), fixed AR(12) beta fallback, fixed noise constants (R=0.1,
q_eta=1e4, q_beta_nominal=5.0, q_beta_other=0.1), n_epochs=30. ONLY fs/Ts
changes between runs.

The multisine's own tone frequencies already auto-adjust with Ts --
make_multisine_input's f_max = du_max/(2*pi*(u_max/2)*Ts) formula (reused
unmodified, transitively via riv_instrument_update_rate_experiment.py)
means every fs gets its own appropriately rate-limited excitation for
free; no separate per-fs retuning of the input is needed.

Reused, unmodified (no source file below is edited in place): Config,
make_scenario, sweep_arm, run_arm_detailed, AUX_DECIMATED, all from
riv_instrument_update_rate_experiment.py (itself already a validated
Entry-47/49 replication built on riv_single_pole_50hz_replication_check.py
and riv_dual_estimation.py).

Search strategy: coarse geometric descent from the fs=50Hz anchor (ratio
1/1.4 per step, ~30% per step) until ok-rate (N_SEEDS=10 seeds, same
<30% relative-error bar as Entry 47/49) drops below OK_RATE_THRESHOLD=0.5,
then bisect in log(fs) space between the last passing and first failing fs
until the bracket is within BISECT_TOL_FRAC=10% relative width. The fs=50Hz
anchor must itself reproduce ok-rate>=ANCHOR_OK_RATE_MIN=0.8 (close to
Entry 49's 10/10) before the rest of the sweep is trusted.

Run: python3 scripts/riv_sampling_rate_sweep_experiment.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "experiments/instrument_update_rate"))
import riv_instrument_update_rate_experiment as rate_mod
from riv_instrument_update_rate_experiment import Config, AUX_DECIMATED, sweep_arm

OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "sampling_rate_sweep"

BASE_CFG = Config()  # fs=50Hz, Ts=0.02 -- Entry 49's confirmed "adaptive" baseline
ADAPTIVE_ARM = dict(name="adaptive", instrument_mode=AUX_DECIMATED, M=1, theta_source=None)

N_SEEDS = 10
COARSE_RATIO = 1 / 1.4       # ~30% step down per coarse iteration
OK_RATE_THRESHOLD = 0.5      # majority-of-seeds bar defining the fail boundary
ANCHOR_OK_RATE_MIN = 0.8     # fs=50Hz must reproduce close to Entry 49's 10/10
BISECT_TOL_FRAC = 0.10       # stop bisecting once bracket width < 10% of fs
MIN_FS = 0.5                 # safety floor so the coarse pass can't run forever


def cfg_at_fs(fs: float) -> Config:
    return dataclasses.replace(BASE_CFG, fs=fs, Ts=1.0 / fs)


def eval_fs(fs: float, n_seeds: int = N_SEEDS) -> dict:
    cfg = cfg_at_fs(fs)
    rows = sweep_arm(ADAPTIVE_ARM, cfg, n_seeds=n_seeds)
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
          f"median theta_err={result['median_theta_err']:.4f}  "
          f"samples/tau1={result['samples_per_tau1']:.2f} samples/tau2={result['samples_per_tau2']:.2f}")
    return result


def coarse_descent(n_seeds: int = N_SEEDS) -> list[dict]:
    """Step fs down geometrically from the confirmed 50Hz anchor until
    ok-rate crosses OK_RATE_THRESHOLD, recording the full curve on the way."""
    curve = []
    fs = BASE_CFG.fs
    anchor = eval_fs(fs, n_seeds=n_seeds)
    curve.append(anchor)
    if anchor["ok_rate"] < ANCHOR_OK_RATE_MIN:
        raise RuntimeError(
            f"fs={fs}Hz anchor ok-rate={anchor['ok_rate']:.2f} does not reproduce "
            f"Entry 49's confirmed baseline (expected >={ANCHOR_OK_RATE_MIN}) -- "
            "sweep would not be trustworthy, stopping.")

    while curve[-1]["ok_rate"] >= OK_RATE_THRESHOLD and fs > MIN_FS:
        fs *= COARSE_RATIO
        curve.append(eval_fs(fs, n_seeds=n_seeds))

    return curve


def bisect_boundary(fs_pass: float, fs_fail: float,
                     n_seeds: int = N_SEEDS) -> tuple[float, float, list[dict]]:
    """Bisect in log(fs) space between a passing (higher) and a failing
    (lower) fs until the bracket is within BISECT_TOL_FRAC of its width."""
    trace = []
    while fs_pass / fs_fail - 1.0 > BISECT_TOL_FRAC:
        fs_mid = float(np.sqrt(fs_pass * fs_fail))  # geometric midpoint
        r = eval_fs(fs_mid, n_seeds=n_seeds)
        trace.append(r)
        if r["ok_rate"] >= OK_RATE_THRESHOLD:
            fs_pass = fs_mid
        else:
            fs_fail = fs_mid
    return fs_pass, fs_fail, trace


def plot_ok_rate_vs_fs(curve: list[dict], boundary_fs: float, n_seeds: int, path: Path,
                        logx: bool = True, err_logy: bool = True,
                        fs_max: float | None = None) -> None:
    pts = sorted(curve, key=lambda r: r["fs"])
    if fs_max is not None:
        pts = [r for r in pts if r["fs"] <= fs_max]
    fss = [r["fs"] for r in pts]
    ok_rates = [r["ok_rate"] for r in pts]
    med_err = [r["median_theta_err"] for r in pts]
    spt1 = [r["samples_per_tau1"] for r in pts]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    axes[0].plot(fss, ok_rates, marker="o")
    axes[0].axvline(boundary_fs, color="r", ls="--", lw=1, label=f"boundary~{boundary_fs:.2f}Hz")
    axes[0].axhline(OK_RATE_THRESHOLD, color="gray", ls=":", lw=1)
    axes[0].set_ylabel(f"ok-rate ({n_seeds} seeds)"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Sampling-rate sweep: 2-pole 50Hz baseline (Entry 49 'adaptive' arm)")
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


def plot_final_params_vs_fs(curve: list[dict], path: Path,
                             logx: bool = False, fs_max: float | None = None) -> None:
    """Final k_hat/tau1_hat/tau2_hat (median across seeds, IQR shaded) vs
    fs, with a dashed line at each true value. Requires curve entries to
    carry a 'rows' list (i.e. built via eval_fs, not a hand-reconstructed
    summary dict)."""
    pts = sorted(curve, key=lambda r: r["fs"])
    if fs_max is not None:
        pts = [r for r in pts if r["fs"] <= fs_max]
    fss = [r["fs"] for r in pts]

    specs = [("k_hat", "k", BASE_CFG.k_true, "tab:blue"),
             ("tau1_hat", "tau1", BASE_CFG.tau1_true, "tab:orange"),
             ("tau2_hat", "tau2", BASE_CFG.tau2_true, "tab:green")]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for ax, (key, label, true_val, color) in zip(axes, specs):
        med, lo, hi = [], [], []
        for r in pts:
            vals = np.array([row[key] for row in r["rows"] if np.isfinite(row[key])])
            med.append(np.median(vals) if len(vals) else np.nan)
            lo.append(np.percentile(vals, 25) if len(vals) else np.nan)
            hi.append(np.percentile(vals, 75) if len(vals) else np.nan)
        ax.plot(fss, med, marker="o", color=color, label=f"{label}_hat (median)")
        ax.fill_between(fss, lo, hi, color=color, alpha=0.2, label="IQR (25-75%)")
        ax.axhline(true_val, color="k", ls="--", lw=1, label=f"{label}_true={true_val}")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        if logx:
            ax.set_xscale("log")
        # Axis range follows the 5th-95th percentile of the median
        # trajectory + true value -- min/max alone still gets dominated by
        # a single wild point right at the low-fs collapse (e.g. tau2_hat
        # can itself, as a median, reach 10-20x true near fs=3Hz), so use
        # percentiles to stay robust to that one point rather than min/max.
        # Such points simply plot off-scale/clipped instead of setting the
        # scale.
        med_arr = np.array(med, dtype=float)
        finite = med_arr[np.isfinite(med_arr)]
        if finite.size:
            p5, p95 = np.percentile(finite, [5, 95])
            lo_y, hi_y = min(p5, true_val), max(p95, true_val)
            pad = 0.15 * (hi_y - lo_y) if hi_y > lo_y else 0.1 * abs(true_val) + 0.1
            ax.set_ylim(lo_y - pad, hi_y + pad)
    axes[-1].set_xlabel("fs (Hz)")
    axes[0].set_title("Final identified parameters vs sampling rate")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_fail_count_vs_fs(curve: list[dict], n_seeds: int, path: Path,
                           logx: bool = True, fs_max: float | None = None) -> None:
    """Number of seeds (out of n_seeds) that fail the ok bar, vs fs."""
    pts = sorted(curve, key=lambda r: r["fs"])
    if fs_max is not None:
        pts = [r for r in pts if r["fs"] <= fs_max]
    fss = [r["fs"] for r in pts]
    n_fail = [r["n_seeds"] - r["n_ok"] for r in pts]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (ax.semilogx if logx else ax.plot)(fss, n_fail, marker="o", color="tab:red")
    ax.set_ylim(-0.5, n_seeds + 0.5)
    ax.set_xlabel("fs (Hz)"); ax.set_ylabel(f"# seeds failing (of {n_seeds})")
    ax.set_title(f"Seeds failing to converge as sampling rate drops ({n_seeds} seeds/point)")
    ax.grid(True, which="both" if logx else "major", axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def interp_boundary_fs(curve: list[dict], threshold: float = OK_RATE_THRESHOLD) -> float:
    """Linear-in-log(fs) interpolation of the ok-rate=threshold crossing,
    for annotating a dense fixed-grid curve that wasn't itself bisected."""
    pts = sorted(curve, key=lambda r: -r["fs"])  # descending fs
    for a, b in zip(pts, pts[1:]):
        if a["ok_rate"] >= threshold > b["ok_rate"]:
            t = (a["ok_rate"] - threshold) / (a["ok_rate"] - b["ok_rate"])
            log_fs = np.log(a["fs"]) + t * (np.log(b["fs"]) - np.log(a["fs"]))
            return float(np.exp(log_fs))
    return float("nan")


def dense_sweep(fs_values: list[float], n_seeds: int, out_dir: Path,
                 reused: list[dict] | None = None) -> tuple[list[dict], float]:
    """Evaluate a fixed, densely-spaced fs grid (no coarse/bisect search) for
    a clean, well-resolved ok-rate-vs-fs curve. `reused` lets already-computed
    points (e.g. from a prior run_sweep at the same n_seeds) be merged in
    instead of recomputed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = list(reused or [])
    print(f"dense grid: {len(reused or [])} reused points + {len(fs_values)} new points, "
          f"n_seeds={n_seeds}")
    for fs in fs_values:
        curve.append(eval_fs(fs, n_seeds=n_seeds))

    boundary_fs = interp_boundary_fs(curve)
    plot_ok_rate_vs_fs(curve, boundary_fs, n_seeds, out_dir / "ok_rate_vs_fs.png")
    plot_fail_count_vs_fs(curve, n_seeds, out_dir / "fail_count_vs_fs.png")

    print(f"\ninterpolated ok-rate=0.5 crossing: fs~{boundary_fs:.3f}Hz")
    print(f"\n{'fs (Hz)':>10s} {'ok-rate':>10s} {'n_fail':>8s} {'median theta_err':>18s} "
          f"{'samples/tau1':>13s} {'samples/tau2':>13s}")
    for r in sorted(curve, key=lambda r: -r["fs"]):
        print(f"{r['fs']:10.3f} {r['ok_rate']:10.2f} {r['n_seeds'] - r['n_ok']:8d} "
              f"{r['median_theta_err']:18.5f} {r['samples_per_tau1']:13.2f} {r['samples_per_tau2']:13.2f}")

    return curve, boundary_fs


def run_sweep(n_seeds: int, out_dir: Path, run_detailed: bool = True) -> tuple[list[dict], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"true (k,tau1,tau2)=({BASE_CFG.k_true},{BASE_CFG.tau1_true},{BASE_CFG.tau2_true}); "
          f"N_sim={BASE_CFG.N_sim} fixed across sweep, anchor fs={BASE_CFG.fs}Hz "
          f"(Entry 49 'adaptive' baseline), N_SEEDS={n_seeds}, "
          f"ok-rate threshold={OK_RATE_THRESHOLD}")

    print("\n=== coarse descent ===")
    curve = coarse_descent(n_seeds=n_seeds)

    fs_pass, fs_fail = curve[-2]["fs"], curve[-1]["fs"]
    if curve[-1]["ok_rate"] >= OK_RATE_THRESHOLD:
        print(f"\ncoarse descent hit MIN_FS={MIN_FS}Hz floor without failing -- "
              f"minimum converging fs is <= {MIN_FS}Hz over this range.")
        boundary_fs = MIN_FS
    else:
        print(f"\n=== bisecting between fs_pass={fs_pass:.3f}Hz and fs_fail={fs_fail:.3f}Hz ===")
        fs_pass, fs_fail, trace = bisect_boundary(fs_pass, fs_fail, n_seeds=n_seeds)
        curve += trace
        boundary_fs = fs_pass
        print(f"\nminimum converging fs ~= {fs_pass:.3f}Hz (bracket: fails by {fs_fail:.3f}Hz, "
              f"width {(fs_pass / fs_fail - 1.0) * 100:.1f}%)")

    plot_ok_rate_vs_fs(curve, boundary_fs, n_seeds, out_dir / "ok_rate_vs_fs.png")
    plot_fail_count_vs_fs(curve, n_seeds, out_dir / "fail_count_vs_fs.png")

    print(f"\n{'fs (Hz)':>10s} {'ok-rate':>10s} {'n_fail':>8s} {'median theta_err':>18s} "
          f"{'samples/tau1':>13s} {'samples/tau2':>13s}")
    for r in sorted(curve, key=lambda r: -r["fs"]):
        print(f"{r['fs']:10.3f} {r['ok_rate']:10.2f} {r['n_seeds'] - r['n_ok']:8d} "
              f"{r['median_theta_err']:18.5f} {r['samples_per_tau1']:13.2f} {r['samples_per_tau2']:13.2f}")

    if run_detailed:
        # Detailed diagnostic plots at the anchor and the identified boundary,
        # reusing riv_instrument_update_rate_experiment's own per-arm plotting
        # orchestration -- redirect its OUT_DIR so plots land here, not in
        # that script's own output directory.
        rate_mod.OUT_DIR = out_dir
        for label, fs in (("anchor_fs50", BASE_CFG.fs), ("boundary", boundary_fs)):
            arm = dict(ADAPTIVE_ARM, name=label)
            rate_mod.run_arm_detailed(arm, cfg_at_fs(fs), seed=0)

    return curve, boundary_fs


def main() -> None:
    run_sweep(N_SEEDS, OUT_DIR)


if __name__ == "__main__":
    main()
