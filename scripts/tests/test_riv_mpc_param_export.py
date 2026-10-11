import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "estimators/mpc_export"))
from riv_mpc_param_export import generate_synthetic_iv


def test_generate_synthetic_iv_shapes_and_range():
    u, y = generate_synthetic_iv(seed=0, k_true=62.11, tau1_true=0.05, tau2_true=0.25, N=500)
    assert u.shape == (500,)
    assert y.shape == (500,)
    assert np.all(u >= 0.004) and np.all(u <= 0.026)
    assert np.all(np.isfinite(y))


def test_run_theta_arm_recovers_reasonable_k():
    from riv_dual_estimation import CFG, DUAL_CFG, AVG_PATIENT_PARAMS
    from riv_mpc_param_export import run_theta_arm, PHI_AR_TRUE, SIGMA2_TRUE

    u, y = generate_synthetic_iv(seed=0, k_true=AVG_PATIENT_PARAMS["k"],
                                  tau1_true=AVG_PATIENT_PARAMS["tau1"],
                                  tau2_true=AVG_PATIENT_PARAMS["tau2"], N=500)
    q_eta = 0.5 * float(np.var(y))
    result = run_theta_arm(u, y, PHI_AR_TRUE, SIGMA2_TRUE, q_eta, CFG, DUAL_CFG)
    assert set(result) == {"k_final", "tau1_final", "tau2_final", "ok",
                            "k_online", "tau1_online", "tau2_online"}
    assert result["k_online"].shape == (500,)
    assert np.all(np.isfinite(result["k_online"]))
    assert np.all(np.isfinite([result["k_final"], result["tau1_final"], result["tau2_final"]]))


def test_run_logparam_arm_recovers_reasonable_k():
    from riv_dual_estimation import CFG, DUAL_CFG, AVG_PATIENT_PARAMS
    from riv_mpc_param_export import run_logparam_arm, PHI_AR_TRUE, SIGMA2_TRUE

    u, y = generate_synthetic_iv(seed=0, k_true=AVG_PATIENT_PARAMS["k"],
                                  tau1_true=AVG_PATIENT_PARAMS["tau1"],
                                  tau2_true=AVG_PATIENT_PARAMS["tau2"], N=500)
    q_eta = 0.5 * float(np.var(y))
    result = run_logparam_arm(u, y, PHI_AR_TRUE, SIGMA2_TRUE, q_eta, CFG, DUAL_CFG)
    assert set(result) == {"k_final", "tau1_final", "tau2_final", "ok",
                            "k_online", "tau1_online", "tau2_online"}
    assert result["k_online"].shape == (500,)
    assert np.all(result["k_online"] > 0)
    assert np.all(np.isfinite(result["k_online"]))


def test_export_synthetic_roundtrip(tmp_path):
    from riv_mpc_param_export import export_synthetic
    path = export_synthetic(n_seeds=2, N=300, out_dir=tmp_path)
    assert path.exists()
    data = np.load(path)
    for key in ("truth_k", "truth_tau1", "truth_tau2",
                "theta_k_final", "theta_tau1_final", "theta_tau2_final", "theta_ok",
                "theta_k_online", "theta_tau1_online", "theta_tau2_online",
                "logparam_k_final", "logparam_tau1_final", "logparam_tau2_final", "logparam_ok",
                "logparam_k_online", "logparam_tau1_online", "logparam_tau2_online"):
        assert key in data, f"missing key {key}"
    assert data["theta_k_final"].shape == (2,)
    assert data["theta_k_online"].shape == (2, 300)
    assert np.all(np.isfinite(data["theta_k_online"]))
    assert np.all(np.isfinite(data["logparam_k_online"]))
