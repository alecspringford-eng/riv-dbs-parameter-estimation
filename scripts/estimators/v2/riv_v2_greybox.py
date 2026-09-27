"""
Version 2 estimator: grey-box output-error identification of the DBS
beta-suppression plant, designed from the Phase 1-4 investigation.

Why this replaces the V1 dual KF+RIV estimator (see PHASE4_ROOT_CAUSE_ANALYSIS.md):
  * physical log-parameterisation p=(log k, log tau...) makes k>0 and every
    pole exp(-Ts/tau) in (0,1) BY CONSTRUCTION -> the V1 negative-k / clamp-
    corner blowup (k=(b1+b2)/(1-a1-a2) dividing by ~0) is structurally impossible;
  * an explicit baseline offset c0 -> the beta DC level no longer hijacks the
    stim-response state (V1's root failure, confirmed by the de-mean experiment);
  * a single joint output-error objective solved by Levenberg-Marquardt -> no
    two-stage KF<->RIV fixed point, hence no global attractor (V1 converged to
    the same wrong answer from every initial guess);
  * output-error is consistent for the plant params whenever u is uncorrelated
    with the (coloured beta) disturbance, so the disturbance model no longer has
    to be correct -- dissolving V1's log/linear domain mismatch.

Model:  y_k = c0 - eta_k(theta) + disturbance_k + noise_k
        eta = ARX response to u of a cascade of `order` first-order lags, gain k.
Order is chosen by AIC (Phase 5 finding: the data support ONE lag on 5/6
recordings -- the second lag is not warranted). Recordings whose input is
essentially uncorrelated with the output (|corr(u,y)| below a floor) are flagged:
u does not drive y there, so k is weakly determined regardless of estimator.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R  # reuse params_to_theta / ARX simulation / plotting

Ts = R.CFG.Ts
UY_COHERENCE_FLOOR = 0.15  # |corr(u,y)| below this -> u barely drives y (flag)
FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "real_data_v2"


def _eta_cascade(k: float, taus: tuple[float, ...], u: np.ndarray) -> np.ndarray:
    """Stim-response of a cascade of first-order lags (gain k), via successive
    exact first-order ZOH filters. One lag for order 1, two for order 2."""
    poles = [np.exp(-Ts / tau) for tau in taus]
    stage_input = u.astype(float)
    for stage, pole in enumerate(poles):
        stage_gain = k if stage == 0 else 1.0  # put the DC gain on the first stage
        input_coeff = stage_gain * (1.0 - pole)

        stage_output = np.zeros_like(stage_input)
        prev = 0.0
        for i in range(len(stage_input)):
            delayed_input = stage_input[i - 1] if i > 0 else 0.0
            prev = pole * prev + input_coeff * delayed_input
            stage_output[i] = prev
        stage_input = stage_output
    return stage_input


@dataclass
class Fit:
    order: int
    k: float
    taus: tuple[float, ...]
    c0: float
    k_sd: float
    tau_sd: tuple[float, ...]
    nrmse: float
    aic: float
    resid: np.ndarray
    uy_corr: float
    flagged: bool
    success: bool


def greybox_oe_fit(u: np.ndarray, y: np.ndarray, order: int = 1,
                   p0: np.ndarray | None = None) -> Fit:
    """Output-error grey-box fit. p = [log k, log tau_1, ..., c0]."""
    ybar = float(y.mean())
    if p0 is None:
        tau_seed = [0.1] if order == 1 else [0.05, 0.25]
        p0 = np.array([np.log(60.0)] + [np.log(t) for t in tau_seed] + [ybar])

    def resid(p: np.ndarray) -> np.ndarray:
        k = np.exp(p[0])
        taus = tuple(np.exp(p[1:1 + order]))
        c0 = p[-1]
        return (c0 - _eta_cascade(k, taus, u)) - y

    sol = least_squares(resid, p0, method="lm")

    n_samples = len(y)
    n_params = order + 2
    rss = float(2.0 * sol.cost)
    aic = n_samples * np.log(rss / n_samples) + 2 * n_params

    # parameter covariance from the Gauss-Newton Jacobian (delta method for
    # log->linear); underestimates under coloured residuals -- bootstrap in the
    # validation script gives the honest interval.
    sigma2 = rss / max(n_samples - n_params, 1)
    try:
        cov = sigma2 * np.linalg.pinv(sol.jac.T @ sol.jac)
        param_sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        param_sd = np.full(n_params, np.nan)

    k = float(np.exp(sol.x[0]))
    taus_unsorted = tuple(float(np.exp(v)) for v in sol.x[1:1 + order])
    sort_idx = np.argsort(taus_unsorted)
    taus = tuple(np.array(taus_unsorted)[sort_idx])
    tau_sd = tuple((np.array(taus) * param_sd[1:1 + order][sort_idx]).tolist())
    uy_corr = float(np.corrcoef(u, y)[0, 1])

    return Fit(
        order=order,
        k=k,
        taus=taus,
        c0=float(sol.x[-1]),
        k_sd=float(k * param_sd[0]),
        tau_sd=tau_sd,
        nrmse=float(np.sqrt(rss / n_samples) / np.std(y)),
        aic=aic,
        resid=resid(sol.x),
        uy_corr=uy_corr,
        flagged=abs(uy_corr) < UY_COHERENCE_FLOOR,
        success=bool(sol.success),
    )


def identify(u: np.ndarray, y: np.ndarray) -> Fit:
    """Fit order 1 and 2, return the AIC-preferred one."""
    f1 = greybox_oe_fit(u, y, order=1)
    f2 = greybox_oe_fit(u, y, order=2)
    return f1 if f1.aic <= f2.aic else f2


def _plot_fit(name: str, u: np.ndarray, y: np.ndarray, f: Fit, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(u)) * Ts
    y_hat = y + f.resid
    fmt = R.CFG.fig_format
    R.plot_input(t, u, out_dir / f"input.{fmt}",
                 title=f"Real stimulation input ({name})")
    R.plot_output_fit(t, y, y_hat, out_dir / f"output_fit.{fmt}",
                       y_hat_label="y_hat (grey-box OE)")
    R.plot_residual(t, f.resid, out_dir / f"residual.{fmt}",
                     ylabel="y_hat - y", title="Output-error residual vs time")


def _selfcheck() -> None:
    """Recover a known synthetic plant (assumptions hold) -> k within a few %."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0.004, 0.026, size=3000)
    k_true, tau1_true, tau2_true, c0_true = 80.0, 0.04, 0.30, 1.10

    eta = _eta_cascade(k_true, (tau1_true, tau2_true), u)
    y = c0_true - eta + rng.normal(0, 0.05, size=len(u))

    f = greybox_oe_fit(u, y, order=2)
    assert f.k > 0, "k must be positive by construction"
    assert abs(f.k - k_true) / k_true < 0.1, f"k recovery off: {f.k} vs {k_true}"
    print(f"[selfcheck] recovered k={f.k:.2f} (truth {k_true}), taus={f.taus} -- OK")


if __name__ == "__main__":
    _selfcheck()
    recs = R.load_all_recordings(R.CFG)
    print(f"\n{'recording':22s} {'order':>5s} {'k':>16s} {'tau(s)':>22s} "
          f"{'NRMSE':>6s} {'corr(u,y)':>9s}")
    results = {}
    for name in R.CFG.system_id_files:
        u, y = recs[name]
        f = identify(u, y)
        taus = ", ".join(f"{t:.4f}" for t in f.taus)
        flag = "  <-- u barely drives y" if f.flagged else ""
        print(f"{name:22s} {f.order:5d} {f.k:7.2f}+-{f.k_sd:6.2f} "
              f"[{taus:>20s}] {f.nrmse:6.3f} {f.uy_corr:+9.3f}{flag}")
        _plot_fit(name, u, y, f, FIG_DIR / name.removesuffix(".csv"))
        results[name] = {
            "k": f.k,
            "tau1": f.taus[0],
            "tau2": f.taus[1] if f.order == 2 else 0.0,
        }

    R.plot_final_params_bar(
        results,
        FIG_DIR / f"final_params_bar.{R.CFG.fig_format}",
    )
    R.plot_final_params_bar(
        results,
        FIG_DIR / f"final_params_bar_vs_avg_patient.{R.CFG.fig_format}",
        include_reference=True,
    )
    print(f"\nFigures written under {FIG_DIR}")
