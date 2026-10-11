
import contextlib
import io
from pathlib import Path
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "estimators/v1"))
import riv_dual_estimation as R

FIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v1_diagnostics"


def run_dual_estimator_carry_state(u, y, phi_ar, Q_diag, R_noise, x0, P0_kf, theta0, P0_rls,
                                    lam, n_eta, p, n_off, gamma_pole, a1_prior, a2_prior,
                                    n_epochs):
    """R.run_dual_estimator, except x_hat[-1] seeds the next epoch instead of
    resetting to x0."""
    n = n_eta + p + n_off
    theta_cur = np.ascontiguousarray(theta0, dtype=np.float64).copy()
    x_cur = np.ascontiguousarray(x0, dtype=np.float64).copy()
    P0_kf_c = np.ascontiguousarray(P0_kf, dtype=np.float64)
    P0_rls_c = np.ascontiguousarray(P0_rls, dtype=np.float64)
    dummy_checks_n = np.zeros((1, n, n))
    dummy_checks_4 = np.zeros((1, 4, 4))

    x_hat = theta_hat = None
    for _epoch in range(n_epochs):
        x_hat, theta_hat, y_hat, Pkf_final, Prls_final, innov, S = R._dual_estimator_core(
            u, y, phi_ar, Q_diag, float(R_noise), x_cur, P0_kf_c.copy(), theta_cur, P0_rls_c.copy(),
            float(lam), n_eta, p, n_off, float(gamma_pole), float(a1_prior), float(a2_prior),
            0, dummy_checks_n, dummy_checks_4)
        theta_cur = theta_hat[-1].copy()
        x_cur = x_hat[-1].copy()
    return theta_cur, innov, S


def main() -> None:
    cfg, dual_cfg = R.CFG, R.DUAL_CFG
    recordings = R.load_all_recordings(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        ar_fits = R.fit_ar_beta_model_per_recording(recordings, cfg)

    y_all = np.concatenate([recordings[f][1] for f in cfg.system_id_files])
    q_eta_global = 0.5 * float(np.var(y_all))

    print("=== 1. per-epoch KF state reset (current) vs carry-forward ===")
    for name in cfg.system_id_files:
        u, y = recordings[name]
        fit = ar_fits[name]
        R_noise = 0.05 * fit["sigma2_ar"]
        with contextlib.redirect_stdout(io.StringIO()):
            setup = R.build_kf_setup(u, y, fit["phi_coeffs"], fit["sigma2_ar"], q_eta_global, cfg, dual_cfg)
            baseline = R.run_identification(u, y, fit["phi_coeffs"], fit["sigma2_ar"], R_noise,
                                             q_eta_global, cfg, dual_cfg, FIG_DIR / "_t", make_plots=False)
            theta_carry, innov_c, S_c = run_dual_estimator_carry_state(
                setup["u_norm"], y, setup["phi_c"], setup["Q_diag"], R_noise, setup["x0"],
                setup["P0_kf"], setup["theta0"], setup["P0_rls"], dual_cfg.lambda_ff,
                R.N_ETA, setup["p"], setup["n_off"], dual_cfg.gamma_pole,
                setup["a1_prior"], setup["a2_prior"], dual_cfg.n_epochs)
            k_carry, tau1_c, tau2_c = R.theta_to_params_physical(theta_carry, setup["u_scale"], cfg.Ts)
            white_carry = R.residual_whiteness_test(innov_c)[1]
            nis_carry = R.nis_consistency_check(innov_c, S_c)[1]
        print(f"  {name}: k reset={baseline['k']:8.2f} (white={baseline['whiteness_frac_exceed']:.3f} "
              f"nis={baseline['nis_mean']:.3f})  ->  k carry-forward={k_carry:8.2f} "
              f"(white={white_carry:.3f} nis={nis_carry:.3f})")

    print("\n=== 2. q_eta global-pooled (current) vs per-recording ===")
    for name in cfg.system_id_files:
        u, y = recordings[name]
        fit = ar_fits[name]
        R_noise = 0.05 * fit["sigma2_ar"]
        q_eta_own = 0.5 * float(np.var(y))
        with contextlib.redirect_stdout(io.StringIO()):
            res_global = R.run_identification(u, y, fit["phi_coeffs"], fit["sigma2_ar"], R_noise,
                                               q_eta_global, cfg, dual_cfg, FIG_DIR / "_t", make_plots=False)
            res_own = R.run_identification(u, y, fit["phi_coeffs"], fit["sigma2_ar"], R_noise,
                                             q_eta_own, cfg, dual_cfg, FIG_DIR / "_t", make_plots=False)
        print(f"  {name}: q_eta_global={q_eta_global:.3e} -> k={res_global['k']:8.2f}  |  "
              f"q_eta_own={q_eta_own:.3e} -> k={res_own['k']:8.2f}  "
              f"(ratio q_eta_own/global={q_eta_own/q_eta_global:.2f})")


if __name__ == "__main__":
    main()
