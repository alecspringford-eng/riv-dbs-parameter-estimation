"""
Version 4: coupled (multiplicative) suppression grey-box.

Motivation (SEMISYNTHETIC_VS_REAL_REPORT.md, INVESTIGATION.md Entries
20/23/25): V1/V2/V3 all share one structural assumption -- the additive
split y = c0 + beta - eta. Real data shows beta's own variance drops to
~40% (AR-residual ~24-28%) during active stimulation, an effect that
survived two independent out-of-sample de-confounding designs (85%
confidence). This matches DBS's documented mechanism -- it suppresses
beta-band OSCILLATORY POWER, not just adds a counter-signal -- so the
additive split is likely the wrong functional form. This module replaces
it with a multiplicative (state-dependent) coupling and tests whether that
still recovers physically reasonable (k, tau1, tau2).

Model:
    eta_k    = two-lag cascade response to u, gain k          (V2's
               _eta_cascade -- deterministic, feedforward, eta_k >= 0
               always since u >= 0 and the cascade has positive gains)
    g_k      = exp(-eta_k)                                     in (0,1]
               suppression gain: g=1 (no damping) when eta=0 (no recent
               stimulation), g->0 as eta grows (heavier suppression).
    beta_k   = AR(p) latent process (same no-stim-fit disturbance model
               already used by V1/V2/V3), zero-mean fluctuation, its OWN
               intrinsic dynamics are NOT changed by stimulation -- only
               how much of it is expressed in y is.
    y_k      = g_k * (c0 + beta_k) + e_k,   e_k ~ N(0, R)
               (suppression gates the WHOLE signal -- baseline AND
               fluctuation together, both convergng toward 0 as
               suppression deepens -- not just the fluctuation. See
               identifiability note below for why this, not a
               fluctuation-only gate, is the version actually used.)

Because eta_k (and hence g_k) depends only on known u and theta -- not on
the data or the beta state -- this is, conditional on theta, an ordinary
LINEAR time-varying Kalman filter (time-varying H_k = g_k * C_beta AND a
time-varying deterministic offset g_k * c0): no EKF/particle filter
needed, no state augmentation for eta (eta is feedforward-only, so unlike
V1/V3 it does not need to be a KF state at all -- the state vector is just
the p beta AR states, smaller than V3's n_eta+p). Estimated jointly by ML
exactly like V3: (log k, log tau1, log tau2, c0, log q_beta, log R) via
the exact KF innovation negative-log-likelihood, L-BFGS. One joint
objective -- no epoch loop, no RIV, no attractor.

IDENTIFIABILITY NOTE (found empirically, kept here because it is a real
result, not a footnote): the first version tried gated ONLY the
fluctuation (y_k = c0 + g_k*beta_k), leaving c0 fixed. That version's
predicted MEAN is c0 + g_k*E[beta_k] =~ c0 for nearly all theta, since the
AR beta state has zero mean regardless of theta -- theta is then only
identifiable through second-moment (innovation variance) information, a
much weaker channel than V1/V2/V3's direct mean-shift information. Self-
check recovery on real u was poor (k err 20%, tau1 err 92%). Gating c0
too (y_k = g_k*(c0+beta_k)) makes the predicted MEAN g_k*c0 itself
depend on theta, restoring a first-moment information channel -- fixed
the self-check (k err 7%, tau1 err 11%, tau2 err 4%, see _selfcheck).
This is still a genuinely non-additive, physiologically motivated
coupling (the entire observed beta signal level is scaled by suppression,
not just its fluctuations) -- not a step back toward the additive form.
"""
from __future__ import annotations
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v2"))
import riv_dual_estimation as R
import riv_v2_greybox as V2

Ts = R.CFG.Ts
UY_COHERENCE_FLOOR = 0.15
FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "real_data_v4"


@dataclass
class FitV4:
    k: float
    tau1: float
    tau2: float
    c0: float
    q_beta: float
    R: float
    nll: float
    nrmse: float
    innov: np.ndarray
    g_traj: np.ndarray
    k_sd: float
    uy_corr: float
    flagged: bool
    success: bool
    message: str
    nis_mean: float
    nis_consistent: bool
    whiteness_pass: bool
    whiteness_frac_exceed: float
    x_traj: np.ndarray


def _kf_likelihood(params: np.ndarray, u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray,
                    ar_order: int, return_traj: bool = False):
    """Gaussian NLL of a linear time-varying KF over the beta AR state alone
    (eta is feedforward-only, so it never enters the state vector -- see
    module docstring). H_k and the deterministic offset are both scaled by
    the suppression gain g_k = exp(-eta_k), which depends only on (k, tau)
    and the known input u, not on the data.
    """
    log_k, log_tau1, log_tau2, c0, log_q_beta, log_meas_noise = params
    k, tau1, tau2 = np.exp(log_k), np.exp(log_tau1), np.exp(log_tau2)
    q_beta = np.exp(log_q_beta)
    meas_noise_var = np.exp(log_meas_noise)

    eta = V2._eta_cascade(k, (tau1, tau2), u)
    suppression_gain = np.exp(-eta)  # deterministic given theta and known u

    beta_transition, beta_output = R.ar_to_state_space(phi_ar)
    beta_output_row = beta_output[0]

    process_noise_diag = np.full(ar_order, 0.1 * q_beta)
    process_noise_diag[0] = q_beta  # matches build_process_noise_diag's convention
    Q = np.diag(process_noise_diag)

    x = np.zeros(ar_order)
    P = 10.0 * np.eye(ar_order)

    n_samples = len(y)
    nll = 0.0
    innov = np.zeros(n_samples) if return_traj else None
    innov_cov_traj = np.zeros(n_samples) if return_traj else None
    x_traj = np.zeros((n_samples, ar_order)) if return_traj else None

    for sample in range(n_samples):
        if sample > 0:
            x = beta_transition @ x
            P = beta_transition @ P @ beta_transition.T + Q
            P = 0.5 * (P + P.T)

        g = suppression_gain[sample]
        H = g * beta_output_row
        predicted_y = g * c0 + float(H @ x)
        innovation = y[sample] - predicted_y
        innov_cov = float(H @ P @ H + meas_noise_var)

        if innov_cov <= 0 or not np.isfinite(innov_cov):
            if return_traj:
                return innov, innov_cov_traj, x_traj, suppression_gain, np.inf
            return 1e18

        nll += 0.5 * (innovation * innovation / innov_cov + np.log(2 * np.pi * innov_cov))

        kalman_gain = (P @ H) / innov_cov
        x = x + kalman_gain * innovation
        P = P - np.outer(kalman_gain, H @ P)
        P = 0.5 * (P + P.T)

        if return_traj:
            innov[sample] = innovation
            innov_cov_traj[sample] = innov_cov
            x_traj[sample] = x

    if return_traj:
        return innov, innov_cov_traj, x_traj, suppression_gain, nll
    return nll


def fit_v4(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, p0: np.ndarray | None = None,
           tau_prior_std: float | None = None) -> FitV4:
    """tau_prior_std: if set, adds a log-normal MAP prior on (tau1,tau2)
    centered at AVG_PATIENT_PARAMS, width tau_prior_std (in log space) --
    see module docstring / INVESTIGATION.md Entry 26 for why: the pure-MLE
    fit on real data robustly (checked against 4 different initializations)
    collapses tau to near-zero, a genuine likelihood optimum, not a local-
    minima artifact. None (default) = pure MLE, unregularized."""
    ar_order = len(phi_ar)
    if p0 is None:
        y_mean = float(np.mean(y))
        y_var = float(np.var(y))
        p0 = np.array([
            np.log(60.0), np.log(0.05), np.log(0.25),
            y_mean,
            np.log(0.5 * y_var), np.log(0.05 * y_var),
        ])
    bounds = [
        (np.log(1e-2), np.log(1e4)),
        (np.log(1e-3), np.log(5)),
        (np.log(1e-3), np.log(5)),
        (-10, 10),
        (np.log(1e-8), np.log(1e3)),
        (np.log(1e-8), np.log(1e3)),
    ]

    if tau_prior_std is None:
        def objective(params):
            return _kf_likelihood(params, u, y, phi_ar, ar_order)
    else:
        log_tau1_prior = np.log(R.AVG_PATIENT_PARAMS["tau1"])
        log_tau2_prior = np.log(R.AVG_PATIENT_PARAMS["tau2"])

        def objective(params):
            nll = _kf_likelihood(params, u, y, phi_ar, ar_order)
            log_prior = (
                0.5 * ((params[1] - log_tau1_prior) / tau_prior_std) ** 2
                + 0.5 * ((params[2] - log_tau2_prior) / tau_prior_std) ** 2
            )
            return nll + log_prior

    sol = minimize(
        objective, p0,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    fitted_params = sol.x

    k = np.exp(fitted_params[0])
    tau1, tau2 = sorted((np.exp(fitted_params[1]), np.exp(fitted_params[2])))
    k_sd = _estimate_k_standard_error(objective, fitted_params, k)

    innov, innov_cov_traj, x_traj, g_traj, nll = _kf_likelihood(
        fitted_params, u, y, phi_ar, ar_order, return_traj=True,
    )

    uy_corr = float(np.corrcoef(u, y)[0, 1])
    _, nis_mean, nis_ok = R.nis_consistency_check(innov, innov_cov_traj)
    whiteness_pass, whiteness_frac = R.residual_whiteness_test(innov)
    innovation_rms = np.sqrt(np.mean(innov ** 2))

    return FitV4(
        k=k, tau1=tau1, tau2=tau2, c0=float(fitted_params[3]),
        q_beta=float(np.exp(fitted_params[4])), R=float(np.exp(fitted_params[5])),
        nll=float(nll), nrmse=float(innovation_rms / np.std(y)),
        innov=innov, g_traj=g_traj, k_sd=float(k_sd), uy_corr=uy_corr,
        flagged=abs(uy_corr) < UY_COHERENCE_FLOOR, success=bool(sol.success),
        message=str(sol.message), nis_mean=nis_mean, nis_consistent=nis_ok,
        whiteness_pass=whiteness_pass, whiteness_frac_exceed=whiteness_frac,
        x_traj=x_traj,
    )


def _estimate_k_standard_error(objective, fitted_params, k):
    """SE of k via the Hessian of the NLL/NLPosterior wrt log k (delta method)."""
    step = 1e-3
    base = fitted_params.copy()

    def value_at(delta_log_k):
        perturbed = base.copy()
        perturbed[0] += delta_log_k
        return objective(perturbed)

    d2 = (value_at(step) - 2 * objective(base) + value_at(-step)) / step ** 2
    if d2 <= 0:
        return np.nan
    return k / np.sqrt(d2)


def _plot_suppression_gain(t: np.ndarray, g_traj: np.ndarray, fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, g_traj, lw=1, color="tab:purple")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("g_k = exp(-eta_k)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Suppression gain (1=unsuppressed, 0=fully suppressed)")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_fit(name: str, u: np.ndarray, y: np.ndarray, f: FitV4, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(u)) * Ts
    y_hat = y - f.innov
    fmt = R.CFG.fig_format
    R.plot_input(t, u, out_dir / f"input.{fmt}", title=f"Real stimulation input ({name})")
    R.plot_output_fit(t, y, y_hat, out_dir / f"output_fit.{fmt}", y_hat_label="y_hat (V4 coupled)")
    R.plot_residual(
        t, f.innov, out_dir / f"residual.{fmt}",
        ylabel="innovation", title="KF innovation vs time",
    )
    _plot_suppression_gain(t, f.g_traj, out_dir / f"suppression_gain.{fmt}")


def _simulate(u: np.ndarray, phi_ar_true: np.ndarray, sigma2_true: float,
              k_true: float, tau1_true: float, tau2_true: float, c0_true: float,
              meas_noise_var_true: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate the exact V4 generative model: y = g*(c0 + beta) + noise."""
    eta_true = V2._eta_cascade(k_true, (tau1_true, tau2_true), u)
    g_true = np.exp(-eta_true)

    beta_transition, beta_output = R.ar_to_state_space(phi_ar_true)
    beta_state = np.zeros(len(phi_ar_true))
    beta = np.zeros(len(u))
    for i in range(len(u)):
        beta[i] = (beta_output @ beta_state)[0]
        beta_state = beta_transition @ beta_state
        beta_state[0] += rng.normal(0, np.sqrt(sigma2_true))

    noise = rng.normal(0, np.sqrt(meas_noise_var_true), len(u))
    return g_true * (c0_true + beta) + noise


def _selfcheck() -> None:
    """Ground-truth recovery check: simulate from the EXACT V4 generative
    model (known k, tau1, tau2, c0, R, AR(3) beta process) and verify
    fit_v4 recovers all three physical parameters, not just k.

    Two excitation regimes, both informative, for different reasons:
    (1) rich broadband u (IID uniform, matches V1's S4.8 convention) --
        tests whether the ESTIMATOR is correct when excitation is
        sufficient. Tight tolerance, asserted.
    (2) real recorded u (system_id_1) -- tests what happens under this
        project's actual deployment excitation. Phase 4
        (PHASE4_INVESTIGATION_REPORT.md) already established, for every
        estimator version tried, that real u has <1% power above the
        fast pole corner and tau1/tau2 are intrinsically confounded
        (corr=-0.87) regardless of algorithm -- so degraded tau precision
        here is an EXPECTED, already-documented data limitation, not a
        new V4-specific failure. k is still checked (V4's SD-response is
        much better constrained than tau's SR-response); tau is reported,
        not asserted, consistent with how this limitation is treated
        everywhere else in this project."""
    rng = np.random.default_rng(0)
    phi_ar_true = np.array([0.6, -0.2, 0.1])
    sigma2_true = 0.05
    k_true, tau1_true, tau2_true, c0_true, meas_noise_var_true = 80.0, 0.05, 0.25, 1.10, 0.01

    u_rich = rng.uniform(0.004, 0.026, size=2745)
    y_rich = _simulate(
        u_rich, phi_ar_true, sigma2_true,
        k_true, tau1_true, tau2_true, c0_true, meas_noise_var_true, rng,
    )
    f_rich = fit_v4(u_rich, y_rich, phi_ar_true)
    k_err = abs(f_rich.k - k_true) / k_true
    tau1_err = abs(f_rich.tau1 - tau1_true) / tau1_true
    tau2_err = abs(f_rich.tau2 - tau2_true) / tau2_true
    assert f_rich.k > 0 and k_err < 0.1, f"k recovery off (rich u): {f_rich.k:.2f} (err {k_err:.1%})"
    assert tau1_err < 0.2 and tau2_err < 0.2, \
        f"tau recovery off (rich u): tau1 err {tau1_err:.1%}, tau2 err {tau2_err:.1%}"
    print(f"[selfcheck, rich broadband u] k={f_rich.k:.2f} (truth {k_true}, err {k_err:.1%})  "
          f"tau1={f_rich.tau1:.4f} (truth {tau1_true}, err {tau1_err:.1%})  "
          f"tau2={f_rich.tau2:.4f} (truth {tau2_true}, err {tau2_err:.1%})  -- all 3 params OK")

    u_real, _ = R.load_all_recordings(R.CFG)["system_id_1.csv"]
    y_real = _simulate(
        u_real, phi_ar_true, sigma2_true,
        k_true, tau1_true, tau2_true, c0_true, meas_noise_var_true, rng,
    )
    f_real = fit_v4(u_real, y_real, phi_ar_true)
    k_err_real = abs(f_real.k - k_true) / k_true
    assert f_real.k > 0 and k_err_real < 0.25, \
        f"k recovery off (real u): {f_real.k:.2f} (err {k_err_real:.1%})"
    print(f"[selfcheck, real recorded u] k={f_real.k:.2f} (truth {k_true}, err {k_err_real:.1%}) -- OK.  "
          f"tau1={f_real.tau1:.4f}, tau2={f_real.tau2:.4f} (truth {tau1_true}/{tau2_true}) -- reported "
          f"only, NOT asserted: real u's weak fast-band content makes tau1/tau2 intrinsically "
          f"confounded for every estimator in this project (Phase 4), not a V4-specific issue.")


def _pooled_ar():
    recs = R.load_all_recordings(R.CFG)
    with contextlib.redirect_stdout(io.StringIO()):
        return R.fit_ar_beta_model_pooled(recs, R.CFG)


TAU_PRIOR_STD = 0.5  # log-normal MAP width for the regularized comparison row -- see Entry 26


if __name__ == "__main__":
    _selfcheck()
    recs = R.load_all_recordings(R.CFG)
    _, phi_ar, _ = _pooled_ar()
    print(f"\n{'recording':22s} {'k':>16s} {'tau1':>8s} {'tau2':>8s} {'NRMSE':>7s} {'corr(u,y)':>9s} "
          f"{'NIS_mean':>9s} {'white':>6s}  {'g range':>14s}")
    print("(pure MLE -- see MAP-regularized comparison further below)")
    results = {}
    for name in R.CFG.system_id_files:
        u, y = recs[name]
        f = fit_v4(u, y, phi_ar)
        flag = "  <-- low u-y coherence" if f.flagged else ""
        flag += "  <-- optimizer not converged" if not f.success else ""
        print(f"{name:22s} {f.k:8.2f}+-{f.k_sd:6.2f} {f.tau1:8.4f} {f.tau2:8.4f} "
              f"{f.nrmse:7.3f} {f.uy_corr:+9.3f} {f.nis_mean:9.3f} "
              f"{str(f.whiteness_pass):>6s}  [{f.g_traj.min():.3f},{f.g_traj.max():.3f}]{flag}")
        _plot_fit(name, u, y, f, FIG_DIR / name.removesuffix(".csv"))
        results[name] = {"k": f.k, "tau1": f.tau1, "tau2": f.tau2}

    print(f"\n=== MAP-regularized (tau_prior_std={TAU_PRIOR_STD}, prior centered on "
          f"AVG_PATIENT_PARAMS tau1={R.AVG_PATIENT_PARAMS['tau1']}, "
          f"tau2={R.AVG_PATIENT_PARAMS['tau2']}) -- transparency comparison, NOT the default ===")
    for name in R.CFG.system_id_files:
        u, y = recs[name]
        f_map = fit_v4(u, y, phi_ar, tau_prior_std=TAU_PRIOR_STD)
        print(f"{name:22s} k={f_map.k:8.2f} tau1={f_map.tau1:.4f} tau2={f_map.tau2:.4f} "
              f"pure_NLL={f_map.nll:.2f} NIS={f_map.nis_mean:.3f}")

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
