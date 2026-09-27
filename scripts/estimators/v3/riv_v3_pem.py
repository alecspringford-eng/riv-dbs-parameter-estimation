"""
Version 3: Maximum-Likelihood / Prediction-Error grey-box identification.

The synthesis of everything the Phase 1-6 investigation learned:

  * PHYSICAL log-parameterisation p=(log k, log tau1, log tau2)  -> k>0 and both
    poles exp(-Ts/tau) in (0,1) BY CONSTRUCTION (V2's robustness; kills the V1
    negative-k / clamp-corner blowup structurally, no ARX->param inversion);
  * an explicit baseline offset c0 (fixes the V1 beta-baseline-hijacks-eta root
    cause);
  * a proper coloured-disturbance NOISE MODEL (the beta AR(p) + its variance
    q_beta) carried inside a Kalman filter, so the estimator is statistically
    EFFICIENT -- V1's one advantage, which pure output-error (V2) gave up;
  * ONE joint objective -- the exact Gaussian KF innovation likelihood -- solved
    by gradient optimisation. No two-stage KF<->RIV fixed point, hence no global
    attractor (V1's failure), and no discarding of the disturbance (V2's
    inefficiency).

Model (state x = [eta1, eta2, beta_k..beta_{k-p+1}]):
    eta      = two-lag plant response to u, gain k        (deterministic, physical)
    beta     = AR(p) disturbance, innovation variance q_beta
    y_k      = c0 + beta_k - eta1_k + e_k,   e_k ~ N(0, R)
Estimated jointly by ML: (log k, log tau1, log tau2, c0, log q_beta, log R).
The AR shape phi_ar is plugged in from the no-stim data (identifiable there);
q_beta and R are re-estimated so the noise model is data-matched, not guessed.
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
import riv_dual_estimation as R

Ts = R.CFG.Ts
Q_ETA = 1e-6          # tiny: eta follows the physical plant (grey-box), not free
UY_COHERENCE_FLOOR = 0.15
FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "real_data_v3"


@dataclass
class FitV3:
    k: float
    tau1: float
    tau2: float
    c0: float
    q_beta: float
    R: float
    nll: float
    nrmse: float
    innov: np.ndarray
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


def _build_state_space(k, tau1, tau2, phi_ar, n_eta, ar_order):
    """Augmented state-space matrices for x = [eta1, eta2, beta_k..beta_{k-p+1}].

    eta obeys the physical two-lag ARX plant (a1,a2,b1,b2); beta obeys the
    AR(p) disturbance model in companion form, appended below it.
    """
    n = n_eta + ar_order
    idx_beta = n_eta

    a1, a2, b1, b2 = R.params_to_theta(k, tau1, tau2, Ts)

    A = np.zeros((n, n))
    A[0, 0], A[0, 1] = a1, 1.0
    A[1, 0] = a2
    A[idx_beta, idx_beta:] = phi_ar
    if ar_order > 1:
        A[idx_beta + 1:, idx_beta:-1] = np.eye(ar_order - 1)

    B = np.zeros(n)
    B[0], B[1] = b1, b2

    H = np.zeros(n)
    H[0] = -1.0
    H[idx_beta] = 1.0

    return A, B, H, idx_beta


def _kf_likelihood(params, u, y, phi_ar, n_eta, ar_order, return_traj=False):
    """Gaussian negative log-likelihood of the KF innovations under `params`.

    With `return_traj=True`, returns the innovation/covariance/state
    trajectories instead (used once, at the fitted optimum, for diagnostics).
    """
    log_k, log_tau1, log_tau2, c0, log_q_beta, log_meas_noise = params
    k, tau1, tau2 = np.exp(log_k), np.exp(log_tau1), np.exp(log_tau2)
    q_beta = np.exp(log_q_beta)
    meas_noise_var = np.exp(log_meas_noise)

    A, B, H, idx_beta = _build_state_space(k, tau1, tau2, phi_ar, n_eta, ar_order)

    n = n_eta + ar_order
    process_noise_diag = np.full(n, Q_ETA)
    process_noise_diag[idx_beta] = q_beta
    Q = np.diag(process_noise_diag)

    x = np.zeros(n)
    P = 10.0 * np.eye(n)

    n_samples = len(y)
    nll = 0.0
    innov = np.zeros(n_samples) if return_traj else None
    innov_cov_traj = np.zeros(n_samples) if return_traj else None
    x_traj = np.zeros((n_samples, n)) if return_traj else None

    for sample in range(n_samples):
        if sample > 0:
            x = A @ x + B * u[sample - 1]
            P = A @ P @ A.T + Q

        innovation = y[sample] - (H @ x + c0)
        innov_cov = float(H @ P @ H + meas_noise_var)

        if innov_cov <= 0 or not np.isfinite(innov_cov):
            if return_traj:
                return innov, innov_cov_traj, x_traj, np.inf
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
        return innov, innov_cov_traj, x_traj, nll
    return nll


def _build_initial_guess(y, ar_order):
    y_mean = float(np.mean(y))
    y_var = float(np.var(y))
    return np.array([
        np.log(60.0), np.log(0.05), np.log(0.25),
        y_mean,
        np.log(0.5 * y_var), np.log(0.05 * y_var),
    ])


def _parameter_bounds():
    """Bounds on (log k, log tau1, log tau2, c0, log q_beta, log R); wide
    enough to never bind at a real optimum, just a runaway guard."""
    return [
        (np.log(1e-2), np.log(1e5)),
        (np.log(1e-3), np.log(5)),
        (np.log(1e-3), np.log(5)),
        (-10, 10),
        (np.log(1e-8), np.log(1e3)),
        (np.log(1e-8), np.log(1e3)),
    ]


def _estimate_k_standard_error(objective, fitted_params, k):
    """SE of k via the Hessian of the NLL wrt log k at the optimum (delta method)."""
    step = 1e-3
    base = fitted_params.copy()

    def nll_at(delta_log_k):
        perturbed = base.copy()
        perturbed[0] += delta_log_k
        return objective(perturbed)

    d2_nll = (nll_at(step) - 2 * objective(base) + nll_at(-step)) / step ** 2
    if d2_nll <= 0:
        return np.nan
    return k / np.sqrt(d2_nll)


def fit_v3(u, y, phi_ar, n_eta=2, p0=None):
    """Fit V3 by maximum likelihood (KF innovation NLL) and return a FitV3
    with the recovered plant, noise model, and diagnostics."""
    ar_order = len(phi_ar)
    if p0 is None:
        p0 = _build_initial_guess(y, ar_order)

    def objective(params):
        return _kf_likelihood(params, u, y, phi_ar, n_eta, ar_order)

    sol = minimize(
        objective, p0,
        method="L-BFGS-B",
        bounds=_parameter_bounds(),
        options={"maxiter": 300, "ftol": 1e-10},
    )
    fitted_params = sol.x

    k = np.exp(fitted_params[0])
    tau1, tau2 = sorted((np.exp(fitted_params[1]), np.exp(fitted_params[2])))
    k_sd = _estimate_k_standard_error(objective, fitted_params, k)

    innov, innov_cov_traj, x_traj, nll = _kf_likelihood(
        fitted_params, u, y, phi_ar, n_eta, ar_order, return_traj=True,
    )

    uy_corr = float(np.corrcoef(u, y)[0, 1])
    _, nis_mean, nis_ok = R.nis_consistency_check(innov, innov_cov_traj)
    whiteness_pass, whiteness_frac = R.residual_whiteness_test(innov)
    innovation_rms = np.sqrt(np.mean(innov ** 2))

    return FitV3(
        k=k, tau1=tau1, tau2=tau2, c0=float(fitted_params[3]),
        q_beta=float(np.exp(fitted_params[4])), R=float(np.exp(fitted_params[5])),
        nll=float(nll), nrmse=float(innovation_rms / np.std(y)),
        innov=innov, k_sd=float(k_sd), uy_corr=uy_corr,
        flagged=abs(uy_corr) < UY_COHERENCE_FLOOR, success=bool(sol.success),
        message=str(sol.message), nis_mean=nis_mean, nis_consistent=nis_ok,
        whiteness_pass=whiteness_pass, whiteness_frac_exceed=whiteness_frac,
        x_traj=x_traj,
    )


def _plot_fit(name: str, u: np.ndarray, y: np.ndarray, f: FitV3, out_dir: Path,
              n_eta: int = 2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(u)) * Ts
    y_hat = y - f.innov
    eta_hat = f.x_traj[:, 0]
    beta_hat = f.x_traj[:, n_eta]
    fmt = R.CFG.fig_format
    R.plot_input(
        t, u, out_dir / f"input.{fmt}",
        title=f"Real stimulation input ({name})",
    )
    R.plot_output_fit(
        t, y, y_hat, out_dir / f"output_fit.{fmt}",
        y_hat_label="y_hat (ML/PEM KF)",
    )
    R.plot_components(t, eta_hat, beta_hat, out_dir / f"components.{fmt}")
    R.plot_residual(
        t, f.innov, out_dir / f"residual.{fmt}",
        ylabel="innovation", title="KF innovation vs time",
    )


def _selfcheck():
    """Recover a known synthetic plant + AR disturbance -> k within a few %."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0.004, 0.026, size=3000)
    _, phi_ar, ar_innovation_var = _pooled_ar()
    ar_transition, ar_output = R.ar_to_state_space(phi_ar)

    k_true, tau1_true, tau2_true, c0_true = 80.0, 0.04, 0.30, 1.10
    eta = R.simulate_arx_recursion(R.params_to_theta(k_true, tau1_true, tau2_true, Ts), u)

    beta_state = np.zeros(len(phi_ar))
    beta = np.zeros(len(u))
    for i in range(len(u)):
        beta[i] = (ar_output @ beta_state)[0]
        beta_state = ar_transition @ beta_state
        beta_state[0] += rng.normal(0, np.sqrt(ar_innovation_var))
    y = c0_true + beta - eta + rng.normal(0, 0.05, len(u))

    f = fit_v3(u, y, phi_ar)
    assert f.k > 0 and abs(f.k - k_true) / k_true < 0.1, f"k recovery off: {f.k}"
    print(f"[selfcheck] k={f.k:.2f} (truth 80) tau1={f.tau1:.4f} tau2={f.tau2:.4f} c0={f.c0:.2f} -- OK")


def _pooled_ar():
    recs = R.load_all_recordings(R.CFG)
    with contextlib.redirect_stdout(io.StringIO()):
        return R.fit_ar_beta_model_pooled(recs, R.CFG)   # linear-domain (D1-fixed default)


if __name__ == "__main__":
    _selfcheck()
    recs = R.load_all_recordings(R.CFG)
    _, phi_ar, _ = _pooled_ar()
    print(f"\n{'recording':22s} {'k':>16s} {'tau1':>8s} {'tau2':>8s} {'NRMSE':>7s} {'corr(u,y)':>9s} "
          f"{'NIS_mean':>9s} {'white':>6s}")
    results = {}
    for name in R.CFG.system_id_files:
        u, y = recs[name]
        f = fit_v3(u, y, phi_ar)
        flag = "  <-- low u-y coherence" if f.flagged else ""
        flag += "  <-- optimizer not converged" if not f.success else ""
        print(f"{name:22s} {f.k:8.2f}+-{f.k_sd:6.2f} {f.tau1:8.4f} {f.tau2:8.4f} "
              f"{f.nrmse:7.3f} {f.uy_corr:+9.3f} {f.nis_mean:9.3f} "
              f"{str(f.whiteness_pass):>6s}{flag}")
        _plot_fit(name, u, y, f, FIG_DIR / name.removesuffix(".csv"))
        results[name] = {"k": f.k, "tau1": f.tau1, "tau2": f.tau2}

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
