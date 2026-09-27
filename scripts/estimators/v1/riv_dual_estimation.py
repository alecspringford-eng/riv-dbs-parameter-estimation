"""
Dual estimator (Kalman filter + recursive Instrumental-Variable RLS) for
joint state and parameter estimation of the DBS beta-suppression model.

Real-data pipeline: the plant (k, tau1, tau2) is identified directly from
real stimulation/beta recordings (real_recordings/<delay_dir>/), and
the beta AR disturbance model is fit from the pooled no-stimulation periods
across every recording. There is no synthetic "true" plant here -- only a
data-driven initial guess that the dual estimator refines epoch by epoch.

Run: python3 scripts/riv_dual_estimation.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import control
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ==========================================================================
# 0. Config
# ==========================================================================

@dataclass(frozen=True)
class Config:
    fs: float = 25.0
    Ts: float = 0.04
    seed: int = 0

    # Real recordings
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "real_recordings")
    # ponytail: two delay-compensated alignments are available (delay_5,
    # delay_6) with no basis in this pipeline to prefer one -- defaulting to
    # delay_5; switch cfg.delay_dir if the other alignment is known correct.
    delay_dir: str = "delay_5"
    system_id_files: tuple[str, ...] = (
        "system_id_1.csv", "system_id_2.csv", "system_id_3.csv",
        "system_id_4.csv", "system_id_5.csv", "system_id_long_run.csv",
    )
    controller_files: tuple[str, ...] = ("simple_controllers_1.csv", "simple_controllers_2.csv")
    # ponytail: 15% above the observed device floor -- separates genuine
    # "stim off" samples from the next PRBS amplitude level up; raise if a
    # recording's off-floor drifts higher than that.
    stim_off_tol_frac: float = 1.15

    # Beta AR(p) disturbance model (fit from pooled no-stim real data)
    settle_time_s: float = 5.0
    seg_ar_order_min: int = 3
    seg_ar_order_max: int = 10
    # D1 fix: fit the beta AR model in the LINEAR output domain (the domain the
    # Kalman filter operates in), not log. Phase 2 showed the log-domain fit put
    # the beta pole ~37% off and mis-scaled Q_beta; refitting linearly cut the
    # real-data k scatter (max k 388->84). Set True only to reproduce the old
    # log-domain behaviour.
    ar_log_domain: bool = False

    # Data / output paths
    fig_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "results" / "real_data")
    fig_format: str = "png"


@dataclass(frozen=True)
class DualConfig:
    """Dual-estimator (KF+RIV) tuning knobs, chosen empirically against the
    real per-recording system_id sequences (2745-6120 samples/pass):
    lambda_ff closer to 1 than 0.999 avoids P_rls growing by (1/lambda)**N
    each pass (verified: 0.999 caused epoch-to-epoch theta to blow up on a
    longer, now-removed concatenated run; kept at the safer 0.9999 here too).
    n_epochs=300: each recording is much shorter than the old concatenated
    run, so more passes are needed per file to settle (verified: some
    recordings, e.g. system_id_long_run, are still drifting slightly at
    300 -- short single recordings are inherently harder to fully converge
    on than the pooled AR data)."""
    lambda_ff: float = 0.9999
    p0_rls: float = 0.3
    p0_kf_state_var: float = 10.0
    n_epochs: int = 30
    # Offset (baseline) state: process-noise variance as a fraction of var(y).
    # Small -> the offset tracks only the slow beta DC baseline, leaving fast
    # fluctuation to beta/eta. This state is the real-data negative-k fix.
    # Tuned by synthetic sweep (median |k err| + positive-k fraction): 1e-2 gives
    # the most positive-k without letting the offset absorb u-driven suppression
    # (>=1 regresses to negative). See PHASE6_TUNING_REPORT.md.
    q_off_frac: float = 1e-2
    # Pole-regularisation leak (opt-in, default OFF): pulls a1,a2 toward prior
    # poles each step. Tested in Phase 8 and REJECTED -- anchoring to the
    # avg-patient prior (a1_prior=1.30, near the unit-sum region) pushes poles
    # toward the clamp corner and makes k *worse*. Kept available (verified) but
    # 0 by default. See PHASE8_V1_FURTHER_TUNING.md.
    gamma_pole: float = 0.0
    # ponytail: unvalidated heuristic threshold for epoch_convergence_check --
    # a numeric companion to the existing k_tau_vs_epoch plot, not itself tuned
    # against any ground truth. Tighten/loosen if it flags recordings the plot
    # visually shows as settled/unsettled.
    epoch_converge_rtol: float = 1e-3


CFG = Config()
DUAL_CFG = DualConfig()

N_ETA = 2
N_THETA = 4

# Average-patient reference plant (literature/prior, not fit from this
# data) -- the RIV estimator is initialized here rather than from a batch
# LS fit, since the only estimator in this script is the recursive IV one.
AVG_PATIENT_PARAMS = {"k": 62.11, "tau1": 0.05, "tau2": 0.25}

# Shared ceiling for the eta-recursion's real poles (project_stable_real_poles
# and _clamp_pole_pair below). 0.95 <-> tau~0.8s at Ts=0.04s. Originally 0.99
# (tau~4s), but a pole that close to 1 makes the DC-gain k=(b1+b2)/(1-a1-a2)
# divide by a near-zero denominator: on real data this let k's sign/magnitude
# swing wildly on tiny fit noise even though eta_hat itself tracked u with
# the physically correct (positive) sign -- verified by direct correlation.
POLE_MAX = 0.95

# Safety ceiling on the RIV update's b1,b2 (u-normalized-input gain terms).
# _clamp_pole_pair only bounds a1,a2 -- b1,b2 have no such constraint, and
# whenever a1,a2 sits at the clamp corner the RIV instrument (z1,z2, which
# is itself propagated forward using b1,b2 each step) goes ill-conditioned
# there and b1,b2 can explode to ~1e18 in a couple of epochs, after which
# the blown-up z1,z2 keeps every later step unstable too (verified: this is
# what made 3/6 recordings diverge under the avg-patient initial guess,
# which visits that corner far more than the old batch-LS warm start did).
# 10.0 is generous headroom over every converged fit seen so far (b1,b2 in
# u-normalized space have stayed under ~1) -- it's a runaway breaker, not a
# tuned physical bound.
B_MAX = 10.0


# ==========================================================================
# 1. Plant model math: continuous plant -> discrete ARX -> observer-canonical
#    (kept for the self-consistency checks and for theta<->(k,tau1,tau2)
#    conversion; there is no "true" plant for real data, only these helpers)
# ==========================================================================

def build_true_ss(k: float, tau1: float, tau2: float) -> control.StateSpace:
    Ac = np.array([[-1 / tau1, 0.0],
                   [1 / tau2, -1 / tau2]])
    Bc = np.array([[k / tau1], [0.0]])
    Cc = np.array([[0.0, 1.0]])
    Dc = np.array([[0.0]])
    return control.ss(Ac, Bc, Cc, Dc)


def discretize_zoh(sys_c: control.StateSpace, Ts: float) -> control.StateSpace:
    return control.c2d(sys_c, Ts, method="zoh")


def arx_theta_from_discrete(sys_d: control.StateSpace) -> np.ndarray:
    """G(z) = (b1 z + b2) / (z^2 + d1 z + d0)  ->  a1=-d1, a2=-d0."""
    tf_d = control.tf(sys_d)
    num = np.atleast_1d(np.squeeze(tf_d.num[0][0]))
    den = np.atleast_1d(np.squeeze(tf_d.den[0][0]))
    den = den / den[0]
    num = np.pad(num, (3 - len(num), 0))
    den = np.pad(den, (3 - len(den), 0), constant_values=0.0)
    den[0] = 1.0
    d1, d0 = den[1], den[2]
    b1, b2 = num[1], num[2]
    return np.array([-d1, -d0, b1, b2])


def params_to_theta(k: float, tau1: float, tau2: float, Ts: float) -> np.ndarray:
    sys_c = build_true_ss(k, tau1, tau2)
    sys_d = discretize_zoh(sys_c, Ts)
    return arx_theta_from_discrete(sys_d)


def theta_to_params(theta: np.ndarray, Ts: float) -> tuple[float, float, float]:
    """Analytic inverse of params_to_theta: recovers (k, tau1, tau2) from
    theta=[a1,a2,b1,b2]. Poles solve the discrete characteristic polynomial
    z^2-a1*z-a2=0, tau=-Ts/log(z)"""
    a1, a2, b1, b2 = theta
    poles = np.roots([1.0, -a1, -a2])

    if np.any(np.abs(poles.imag) > 1e-9):
        raise ValueError(f"theta_to_params requires two real poles, got {poles}")
    poles = poles.real
    if np.any((poles <= 0.0) | (poles >= 1.0)):
        raise ValueError(f"theta_to_params requires two stable poles in (0,1), got {poles}")

    tau1, tau2 = sorted(-Ts / np.log(poles))
    k = (b1 + b2) / (1.0 - a1 - a2)
    return float(k), float(tau1), float(tau2)


def theta_to_observer_canonical(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A_eta=[[a1,1],[a2,0]], B_eta=[[b1],[b2]], C_eta=[1,0]."""
    a1, a2, b1, b2 = theta
    A_eta = np.array([[a1, 1.0], [a2, 0.0]])
    B_eta = np.array([[b1], [b2]])
    C_eta = np.array([[1.0, 0.0]])
    return A_eta, B_eta, C_eta


def simulate_ss(A: np.ndarray, B: np.ndarray, C: np.ndarray, u: np.ndarray,
                 x0: np.ndarray | None = None) -> np.ndarray:
    u = np.atleast_2d(u.reshape(-1, 1))
    n = A.shape[0]
    N = u.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    y = np.zeros(N)
    for kk in range(N):
        y[kk] = (C @ x)[0]
        x = A @ x + (B @ u[kk])
    return y


def simulate_arx_recursion(theta: np.ndarray, u: np.ndarray) -> np.ndarray:
    a1, a2, b1, b2 = theta
    N = len(u)
    y = np.zeros(N)
    for kk in range(N):
        y_km1 = y[kk - 1] if kk - 1 >= 0 else 0.0
        y_km2 = y[kk - 2] if kk - 2 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        u_km2 = u[kk - 2] if kk - 2 >= 0 else 0.0
        y[kk] = a1 * y_km1 + a2 * y_km2 + b1 * u_km1 + b2 * u_km2
    return y


def run_self_consistency_checks(cfg: Config) -> None:
    """ARX recursion == forced_response == observer-canonical simulation,
    and theta's poles == analytic ZOH poles, for an arbitrary but
    physically-plausible test plant. This checks the math only -- it has
    no bearing on the real-data parameter estimates."""
    k_test, tau1_test, tau2_test = 60.0, 0.05, 0.25
    sys_c = build_true_ss(k_test, tau1_test, tau2_test)
    sys_d = discretize_zoh(sys_c, cfg.Ts)
    theta_test = arx_theta_from_discrete(sys_d)

    rng = np.random.default_rng(cfg.seed)
    u_test = rng.uniform(0, 2.5, size=500)

    y_arx = simulate_arx_recursion(theta_test, u_test)
    t = np.arange(len(u_test)) * cfg.Ts
    resp = control.forced_response(sys_d, T=t, U=u_test)
    y_ss = np.asarray(resp.outputs).squeeze()
    assert np.allclose(y_arx, y_ss, atol=1e-6, rtol=1e-6), \
        "ARX recursion does not match discretized state-space response"

    A_eta, B_eta, C_eta = theta_to_observer_canonical(theta_test)
    y_obs = simulate_ss(A_eta, B_eta, C_eta, u_test)
    assert np.allclose(y_obs, y_arx, atol=1e-9), \
        "Observer-canonical simulation does not match ARX recursion"

    poles_theta = np.roots([1.0, -theta_test[0], -theta_test[1]])
    poles_analytic = np.array([np.exp(-cfg.Ts / tau1_test), np.exp(-cfg.Ts / tau2_test)])
    assert np.allclose(sorted(poles_theta), sorted(poles_analytic), atol=1e-8), \
        "theta poles do not match analytic ZOH poles of the two lags"

    print("[4.1] Self-consistency checks passed (ARX/state-space/observer-canonical/poles).")


# ==========================================================================
# 2. Real recordings + beta AR(p) disturbance model pooled across all of them
# ==========================================================================

def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["inputs"].to_numpy(dtype=float), df["outputs"].to_numpy(dtype=float)


def load_all_recordings(cfg: Config) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rec_dir = cfg.data_dir / cfg.delay_dir
    names = cfg.system_id_files + cfg.controller_files
    return {name: load_recording(rec_dir / name) for name in names}


def find_bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where mask is True."""
    change_idx = np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1
    bounds = np.concatenate(([0], change_idx, [len(mask)]))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1) if mask[bounds[i]]]


def compute_epsilon_floor(beta: np.ndarray) -> float:
    return float(np.percentile(beta[beta > 0], 1))


def clean_beta_segments(beta_log: np.ndarray, segments: list[tuple[int, int]],
                         trim_samples: int, min_len: int) -> list[np.ndarray]:
    """Per segment: drop the first/last `trim_samples` (stim-transition
    transients) if enough data remain, then detrend by subtracting the
    (trimmed) segment mean. Segments still shorter than `min_len` after
    trimming are dropped -- too little data for a stable AR fit."""
    cleaned = []
    for s, e in segments:
        seg = beta_log[s:e]
        trimmed = seg[trim_samples:-trim_samples] if len(seg) > 2 * trim_samples else seg
        if len(trimmed) < min_len:
            continue
        cleaned.append(trimmed - trimmed.mean())
    return cleaned


def no_stim_segments_for_recording(u: np.ndarray, y: np.ndarray, thresh: float,
                                    trim_samples: int, min_len: int,
                                    log_domain: bool, epsilon: float) -> list[np.ndarray]:
    no_stim_runs = find_bool_runs(u <= thresh)
    beta_t = np.log(y + epsilon) if log_domain else y
    return clean_beta_segments(beta_t, no_stim_runs, trim_samples, min_len)


def fit_ar_ols(fluct: np.ndarray, order: int) -> tuple[np.ndarray, float, np.ndarray]:
    rows = [fluct[kk - order:kk][::-1] for kk in range(order, len(fluct))]
    target = fluct[order:]
    Phi = np.array(rows)
    phi, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    resid = target - Phi @ phi
    sigma2 = float(np.sum(resid ** 2) / len(resid))
    return phi, sigma2, resid


def aic_score(sigma2: float, order: int, n: int) -> float:
    return n * np.log(sigma2) + 2 * (order + 1)


ACF_MAX_LAG = 20
WHITENESS_CI = 1.96
WHITENESS_FAIL_FRAC = 0.05


def residual_whiteness_test(resid: np.ndarray) -> tuple[bool, float]:
    n = len(resid)
    max_lag = min(ACF_MAX_LAG, n - 1)
    resid = resid - resid.mean()
    denom = np.sum(resid ** 2)
    acf = np.array([np.sum(resid[:n - lag] * resid[lag:]) / denom for lag in range(1, max_lag + 1)])
    conf = WHITENESS_CI / np.sqrt(n)
    frac_exceed = float(np.mean(np.abs(acf) > conf))
    return frac_exceed <= WHITENESS_FAIL_FRAC, frac_exceed


def select_segment_ar_order(fluct: np.ndarray, order_min: int, order_max: int) -> dict:
    results = []
    for p in range(order_min, order_max + 1):
        if len(fluct) - p < p + 5:
            continue
        phi, sigma2, resid = fit_ar_ols(fluct, p)
        passed, frac_exceed = residual_whiteness_test(resid)
        results.append({"order": p, "phi": phi, "sigma2": sigma2,
                         "aic": aic_score(sigma2, p, len(resid)),
                         "passed": passed, "frac_exceed": frac_exceed})
    pool = [r for r in results if r["passed"]] or results
    return min(pool, key=lambda r: r["aic"])


def fit_ar_pooled(fluct_segments: list[np.ndarray], order: int) -> tuple[np.ndarray, float]:
    rows, targets = [], []
    for seg in fluct_segments:
        rows.extend(seg[kk - order:kk][::-1] for kk in range(order, len(seg)))
        targets.extend(seg[order:])
    Phi = np.array(rows)
    y = np.array(targets)
    phi, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    resid = y - Phi @ phi
    sigma2 = float(np.sum(resid ** 2) / len(y))
    return phi, sigma2


def _select_consensus_ar(segments: list[np.ndarray], order_min: int, order_max: int,
                          verbose: bool = True) -> tuple[int, np.ndarray, float]:
    """Per-segment AR order selection (AIC + whiteness) -> median consensus
    order -> final pooled OLS refit at that order."""
    selected_orders = []
    for i, fluct in enumerate(segments):
        best = select_segment_ar_order(fluct, order_min, order_max)
        selected_orders.append(best["order"])
        if verbose:
            print(f"  segment {i}: order={best['order']}  whiteness_pass={best['passed']}  "
                  f"resid_var={best['sigma2']:.4e}")
    consensus_order = int(round(np.clip(np.median(selected_orders), order_min, order_max)))
    if verbose:
        print(f"consensus AR order (median of segment selections): {consensus_order}")
    phi_coeffs, sigma2 = fit_ar_pooled(segments, consensus_order)
    if verbose:
        print(f"final AR({consensus_order}) coefficients: {np.array2string(phi_coeffs, precision=4)}")
        print(f"final innovation variance: {sigma2:.4e}")
    return consensus_order, phi_coeffs, sigma2


def fit_ar_beta_model_pooled(recordings: dict[str, tuple[np.ndarray, np.ndarray]],
                              cfg: Config) -> tuple[int, np.ndarray, float]:
    """Beta disturbance-model pipeline, pooled over every recording's
    no-stimulation periods: log-transform (one epsilon floor shared across
    all recordings), per-recording no-stim segment detection, per-segment
    trim+detrend, per-segment AR order selection (AIC + residual
    whiteness), then one final pooled refit at the consensus (median)
    order using every recording's cleaned segments together."""
    all_outputs = np.concatenate([y for _, y in recordings.values()])
    epsilon = compute_epsilon_floor(all_outputs)
    thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac
    domain = "log" if cfg.ar_log_domain else "linear"
    print(f"AR fit domain: {domain}; epsilon (log floor): {epsilon:.3e}; "
          f"no-stim input threshold: {thresh:.4e}")

    trim_samples = round(cfg.settle_time_s / cfg.Ts)
    min_len = cfg.seg_ar_order_max + 20
    fluct_segments = []
    for name, (u, y) in recordings.items():
        cleaned = no_stim_segments_for_recording(u, y, thresh, trim_samples, min_len,
                                                  cfg.ar_log_domain, epsilon)
        fluct_segments.extend(cleaned)
        print(f"  {name}: -> {len(cleaned)} usable segment(s) after trim/length filter")
    assert fluct_segments, "no usable no-stimulation segments across any recording"
    total_samples = sum(len(s) for s in fluct_segments)
    print(f"pooled {len(fluct_segments)} no-stim segments across {len(recordings)} "
          f"recordings, {total_samples} samples total ({total_samples * cfg.Ts:.1f}s)")

    return _select_consensus_ar(fluct_segments, cfg.seg_ar_order_min, cfg.seg_ar_order_max)


def _find_ar_donor(name: str, seg_pool: dict[str, list[np.ndarray]],
                    system_id_files: tuple[str, ...],
                    controller_files: tuple[str, ...]) -> tuple[list[np.ndarray], str]:
    """Borrow no-stim segments for a recording with none of its own, from the
    nearest other system_id recording (by filename-index adjacency) that has
    usable segments; simple_controllers recordings are a last-resort tier."""
    idx = system_id_files.index(name)
    candidates = sorted((p for p in range(len(system_id_files)) if p != idx),
                         key=lambda p: (abs(p - idx), p))
    for p in candidates:
        donor = system_id_files[p]
        if seg_pool[donor]:
            return list(seg_pool[donor]), donor
    for cf in controller_files:
        if seg_pool[cf]:
            return list(seg_pool[cf]), cf
    raise AssertionError(f"no donor with usable no-stim segments found for {name}")


def fit_ar_beta_model_per_recording(
    recordings: dict[str, tuple[np.ndarray, np.ndarray]], cfg: Config
) -> dict[str, dict]:
    """Per-recording beta AR(p) disturbance model: each system_id recording
    is fit on its own no-stim segments; a recording with no usable segments
    of its own borrows them from the nearest other recording that has some
    (see _find_ar_donor)."""
    all_outputs = np.concatenate([y for _, y in recordings.values()])
    epsilon = compute_epsilon_floor(all_outputs)
    thresh = min(u.min() for u, _ in recordings.values()) * cfg.stim_off_tol_frac
    trim_samples = round(cfg.settle_time_s / cfg.Ts)
    min_len = cfg.seg_ar_order_max + 20

    seg_pool = {name: no_stim_segments_for_recording(u, y, thresh, trim_samples, min_len,
                                                       cfg.ar_log_domain, epsilon)
                for name, (u, y) in recordings.items()}

    results = {}
    for name in cfg.system_id_files:
        own = seg_pool[name]
        segments, donor = (own, None) if own else _find_ar_donor(
            name, seg_pool, cfg.system_id_files, cfg.controller_files)

        order, phi_coeffs, sigma2 = _select_consensus_ar(
            segments, cfg.seg_ar_order_min, cfg.seg_ar_order_max, verbose=False)
        stable, eigvals = ar_stability_check(phi_coeffs)
        n_samples = sum(len(s) for s in segments)

        source = (f"own ({len(own)} seg, {n_samples} samples)" if donor is None
                  else f"borrowed from {donor} ({len(segments)} seg, {n_samples} samples)")
        stability_msg = "stable" if stable else f"UNSTABLE max|eig|={np.max(np.abs(eigvals)):.4f}"
        print(f"  {name}: {source}, AR({order}), sigma2={sigma2:.4e}, {stability_msg}")

        results[name] = {"order": order, "phi_coeffs": phi_coeffs, "sigma2_ar": sigma2,
                          "n_segments": len(segments), "n_samples": n_samples,
                          "donor": donor, "stable": stable, "eigvals": eigvals}
    return results


def ar_to_state_space(phi_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Companion form. State ordering x_beta=[beta_k,...,beta_{k-p+1}],
    C_beta=[1,0,...,0]."""
    p = len(phi_coeffs)
    A_beta = np.zeros((p, p))
    A_beta[0, :] = phi_coeffs
    if p > 1:
        A_beta[1:, :-1] = np.eye(p - 1)
    C_beta = np.zeros((1, p))
    C_beta[0, 0] = 1.0
    return A_beta, C_beta


def ar_stability_check(phi_coeffs: np.ndarray) -> tuple[bool, np.ndarray]:
    """Eigenvalues of the AR(p) companion matrix == characteristic-polynomial
    roots; stable iff all |eigenvalue| < 1. An unstable beta disturbance
    model would make the KF's beta sub-state diverge."""
    A_beta, _ = ar_to_state_space(phi_coeffs)
    eigvals = np.linalg.eigvals(A_beta)
    return bool(np.all(np.abs(eigvals) < 1.0)), eigvals


def resample_ar_model(phi_coeffs: np.ndarray, factor: int) -> np.ndarray:
    """Exact resampling of a discrete-time AR(p) model to `factor`x the
    original sampling period (e.g. factor=2 to go from 50 Hz to 25 Hz).

    If x_{k+1} = A x_k (companion form), observing every `factor`-th sample
    is exactly x_{k+factor} = A^factor x_k -- the new transition matrix is a
    matrix power of the original, no approximation. A companion matrix's
    characteristic polynomial IS the AR polynomial, so the new AR(p)
    coefficients are recovered from the characteristic polynomial of
    A^factor (its eigenvalues are exactly the original eigenvalues raised to
    `factor`). Verified numerically: eig(A^factor) == eig(ar_to_state_space(
    resample_ar_model(phi, factor))[0]) to 1e-8, for complex conjugate pairs
    too. Process-noise variance is NOT handled here -- propagate Q
    separately (Q_new = A Q A^T + Q for factor=2, generalizes to a sum over
    factor steps) since this codebase's Q is a diagonal approximation, not
    the exact companion-form noise covariance."""
    A_old, _ = ar_to_state_space(phi_coeffs)
    A_new = np.linalg.matrix_power(A_old, factor)
    charpoly = np.poly(np.linalg.eigvals(A_new))
    return (-charpoly[1:]).real


# ==========================================================================
# 3. Pole-projection helpers (initial guess is the average-patient prior via
#    params_to_theta; these exist only as a reporting-time safety net -- the
#    per-step clamp in the RIV core keeps theta in this family already)
# ==========================================================================

def project_stable_real_poles(a1: float, a2: float) -> tuple[float, float]:
    """Project (a1,a2) onto a stable, two-*positive*-real-pole cascade: the
    plant is physically two first-order lags in series (poles=exp(-Ts/tau),
    always in (0,1)), but an unconstrained ARX(2,2) fit on noisy real data
    routinely lands on complex, negative, or unstable poles. Complex/negative
    poles are replaced by two distinct positive real poles at the same
    natural magnitude (critically-damped approximation); anything outside
    the unit circle is then clipped back in. Used both to seed a physically
    valid initial guess and, where the recursive fit's poles fall outside
    this family, as a nearest-valid-cascade reading for reporting."""
    poles = np.roots([1.0, -a1, -a2])
    if np.any(np.abs(poles.imag) > 1e-9) or np.any(poles.real <= 0):
        # ponytail: split into (mag, mag**2) rather than a fixed +-5%/10%
        # spread -- a fixed spread saturates against the clip ceiling for
        # any near-unit dominant pole, collapsing every such fit onto the
        # same (tau1,tau2) and erasing real per-recording differences
        # (verified: every complex/negative-pole file mapped to the same
        # projected tau1/tau2 with the old fixed-ratio version).
        mag = float(np.clip(np.abs(poles[0]), 0.3, POLE_MAX))
        poles = np.array([mag, mag ** 2])
    else:
        poles = np.clip(poles.real, 1e-3, POLE_MAX)
    return float(poles.sum()), float(-np.prod(poles))


def theta_to_params_safe(theta: np.ndarray, Ts: float) -> tuple[float, float, float]:
    """theta_to_params, falling back to the nearest valid two-positive-
    real-pole cascade (see project_stable_real_poles) if the recursive
    estimator's current theta doesn't fall in that family."""
    try:
        return theta_to_params(theta, Ts)
    except ValueError:
        a1_new, a2_new = project_stable_real_poles(theta[0], theta[1])
        return theta_to_params(np.array([a1_new, a2_new, theta[2], theta[3]]), Ts)


# ==========================================================================
# 4. Dual estimator core
# ==========================================================================

def build_process_noise_diag(n_eta: int, p_beta: int, q_eta: float,
                              q_beta_nominal: float, q_beta_other: float) -> np.ndarray:
    q = np.full(n_eta + p_beta, q_eta)
    q[n_eta:] = q_beta_other
    q[n_eta] = q_beta_nominal
    return q


@numba.njit(cache=True)
def _clamp_pole_pair(a1: float, a2: float) -> tuple[float, float]:
    """Project the eta-recursion's (a1,a2) back onto a valid two-real-
    positive-pole cascade after every RIV update, not just at reporting
    time. Without this, the a1,a2 sub-map has a marginal-stability
    attractor at a1->0, a2->1 (poles -> +1,-1) for this real, weakly
    exciting data -- verified: theta drifted there smoothly over dozens of
    epochs, then bounced unpredictably on crossing the real/complex-pole
    boundary, which is what made epoch-end (k,tau1,tau2) look random.
    Clamping every step keeps the walk inside the physical region so it
    can no longer approach that boundary at all."""
    pole_min = 1e-3
    pole_max = POLE_MAX
    disc = a1 * a1 + 4.0 * a2
    if disc < 0.0:
        a2 = -a1 * a1 / 4.0
        disc = 0.0
    sq = disc ** 0.5
    p1 = (a1 + sq) / 2.0
    p2 = (a1 - sq) / 2.0
    if p1 < pole_min:
        p1 = pole_min
    elif p1 > pole_max:
        p1 = pole_max
    if p2 < pole_min:
        p2 = pole_min
    elif p2 > pole_max:
        p2 = pole_max
    return p1 + p2, -(p1 * p2)


@numba.njit(cache=True)
def _clamp_theta(a1: float, a2: float, b1: float, b2: float) -> tuple[float, float, float, float]:
    """_clamp_pole_pair for (a1,a2), plus a magnitude ceiling on b1,b2 --
    see B_MAX above for why b1,b2 need their own guard."""
    a1, a2 = _clamp_pole_pair(a1, a2)
    if b1 > B_MAX:
        b1 = B_MAX
    elif b1 < -B_MAX:
        b1 = -B_MAX
    if b2 > B_MAX:
        b2 = B_MAX
    elif b2 < -B_MAX:
        b2 = -B_MAX
    return a1, a2, b1, b2


@numba.njit(cache=True)
def _structured_kf_prop(P: np.ndarray, Q_diag: np.ndarray, phi_ar: np.ndarray,
                         a1: float, a2: float, n_eta: int, p: int, n_off: int,
                         temp: np.ndarray, Pminus: np.ndarray) -> None:
    # n_off (0 or 1): optional constant/slow-drift offset state appended last,
    # dynamics = identity (A_off = 1). It gives the beta DC baseline its own
    # degree of freedom so it stops being absorbed by the stim-response eta --
    # the root cause of the real-data negative-k failure (see run_identification).
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off

    for j in range(n):
        temp[0, j] = a1 * P[0, j] + P[1, j]
        temp[1, j] = a2 * P[0, j]
        s = 0.0
        for l in range(p):
            s += phi_ar[l] * P[idx_beta + l, j]
        temp[idx_beta, j] = s
        for m in range(1, p):
            temp[idx_beta + m, j] = P[idx_beta + m - 1, j]
        if n_off == 1:
            temp[idx_off, j] = P[idx_off, j]

    for i in range(n):
        Pminus[i, 0] = a1 * temp[i, 0] + temp[i, 1]
        Pminus[i, 1] = a2 * temp[i, 0]
        s2 = 0.0
        for l in range(p):
            s2 += phi_ar[l] * temp[i, idx_beta + l]
        Pminus[i, idx_beta] = s2
        for m in range(1, p):
            Pminus[i, idx_beta + m] = temp[i, idx_beta + m - 1]
        if n_off == 1:
            Pminus[i, idx_off] = temp[i, idx_off]

    for i in range(n):
        Pminus[i, i] += Q_diag[i]

    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (Pminus[i, j] + Pminus[j, i])
            Pminus[i, j] = avg
            Pminus[j, i] = avg


@numba.njit(cache=True)
def _dual_estimator_core(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, Q_diag: np.ndarray,
                          R: float, x0: np.ndarray, P0_kf: np.ndarray, theta0: np.ndarray,
                          P0_rls: np.ndarray, lam: float, n_eta: int, p: int, n_off: int,
                          gamma_pole: float, a1_prior: float, a2_prior: float,
                          cov_stride: int,
                          Pkf_checks: np.ndarray, Prls_checks: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = y.shape[0]

    x = x0.copy()
    Pkf = P0_kf.copy()
    theta = theta0.copy()
    Prls = P0_rls.copy()
    z1 = 0.0
    z2 = 0.0

    x_hat = np.zeros((N, n))
    theta_hat = np.zeros((N, 4))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    x_new = np.empty(n)
    temp = np.empty((n, n))
    Pminus = np.empty((n, n))
    HP = np.empty(n)
    Kkf = np.empty(n)
    phi_vec = np.empty(4)
    z_vec = np.empty(4)
    Pz = np.empty(4)
    phiP = np.empty(4)
    Krls = np.empty(4)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a1 = theta[0]
            a2 = theta[1]
            b1 = theta[2]
            b2 = theta[3]
            eta1 = x[0]
            eta2 = x[1]

            x_new[0] = a1 * eta1 + eta2 + b1 * u_prev
            x_new[1] = a2 * eta1 + b2 * u_prev
            s = 0.0
            for l in range(p):
                s += phi_ar[l] * x[idx_beta + l]
            x_new[idx_beta] = s
            for m in range(1, p):
                x_new[idx_beta + m] = x[idx_beta + m - 1]
            if n_off == 1:
                x_new[idx_off] = x[idx_off]          # constant/slow-drift offset
            for i in range(n):
                x[i] = x_new[i]

            _structured_kf_prop(Pkf, Q_diag, phi_ar, a1, a2, n_eta, p, n_off, temp, Pminus)
            for i in range(n):
                for j in range(n):
                    Pkf[i, j] = Pminus[i, j]

        # scalar KF measurement update. H = [-1,0,...,1(@beta),...,+1(@off)]:
        # y = beta - eta + offset. The offset carries the beta DC baseline so it
        # no longer leaks into eta.
        off = x[idx_off] if n_off == 1 else 0.0
        h_pred = x[idx_beta] - x[0] + off
        innov = y[kk] - h_pred

        for j in range(n):
            HP[j] = -Pkf[0, j] + Pkf[idx_beta, j]
            if n_off == 1:
                HP[j] += Pkf[idx_off, j]
        S = -HP[0] + HP[idx_beta] + R
        if n_off == 1:
            S += HP[idx_off]
        innov_hist[kk] = innov
        S_hist[kk] = S
        for i in range(n):
            Kkf[i] = HP[i] / S

        for i in range(n):
            x[i] = x[i] + Kkf[i] * innov
        for i in range(n):
            for j in range(n):
                Pkf[i, j] = Pkf[i, j] - Kkf[i] * HP[j]
        for i in range(n):
            for j in range(i + 1, n):
                avg = 0.5 * (Pkf[i, j] + Pkf[j, i])
                Pkf[i, j] = avg
                Pkf[j, i] = avg

        eta_hat_k = x[0]
        for i in range(n):
            x_hat[kk, i] = x[i]
        y_hat[kk] = x[idx_beta] - x[0] + (x[idx_off] if n_off == 1 else 0.0)

        # ARX regressor phi_k=[eta_{k-1},eta_{k-2},u_{k-1},u_{k-2}]
        eta_km1 = x_hat[kk - 1, 0] if kk - 1 >= 0 else 0.0
        eta_km2 = x_hat[kk - 2, 0] if kk - 2 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        u_km2 = u[kk - 2] if kk - 2 >= 0 else 0.0
        phi_vec[0] = eta_km1
        phi_vec[1] = eta_km2
        phi_vec[2] = u_km1
        phi_vec[3] = u_km2

        # Instrument z_k=[eta_aux_{k-1},eta_aux_{k-2},u_{k-1},u_{k-2}]: eta_aux
        # is a noise-free simulation (updated below using pre-update theta),
        # uncorrelated with the colored disturbance leaking into eta_hat.
        z_vec[0] = z1
        z_vec[1] = z2
        z_vec[2] = u_km1
        z_vec[3] = u_km2

        # RIV update (Soderstrom & Stoica): gain uses P@z (not P@phi)
        for i in range(4):
            s3 = 0.0
            for j in range(4):
                s3 += Prls[i, j] * z_vec[j]
            Pz[i] = s3
        denom = lam
        for i in range(4):
            denom += phi_vec[i] * Pz[i]
        for i in range(4):
            Krls[i] = Pz[i] / denom

        pred = 0.0
        for i in range(4):
            pred += phi_vec[i] * theta[i]
        err = eta_hat_k - pred

        # extend the auxiliary/instrument simulation one step using the
        # (still pre-update) theta, before theta itself gets updated below
        z_new = theta[0] * z1 + theta[1] * z2 + theta[2] * u_km1 + theta[3] * u_km2
        z2 = z1
        z1 = z_new

        for i in range(4):
            theta[i] = theta[i] + Krls[i] * err
        theta[0], theta[1], theta[2], theta[3] = _clamp_theta(theta[0], theta[1], theta[2], theta[3])
        # Prior-regularised (leaky) RIV on the POLE coefficients only: pull a1,a2
        # a fraction gamma_pole toward the physical prior each step. The pole
        # directions are weakly excited / confounded (Phase 4) and otherwise
        # drift to the clamp-corner attractor where (1-a1-a2)->0 blows k up; the
        # gain b1,b2 stays free to fit the data. gamma_pole=0 -> original RIV.
        if gamma_pole > 0.0:
            theta[0] += gamma_pole * (a1_prior - theta[0])
            theta[1] += gamma_pole * (a2_prior - theta[1])
            theta[0], theta[1] = _clamp_pole_pair(theta[0], theta[1])
        for i in range(4):
            s4 = 0.0
            for j in range(4):
                s4 += phi_vec[j] * Prls[j, i]
            phiP[i] = s4
        for i in range(4):
            for j in range(4):
                Prls[i, j] = (Prls[i, j] - Krls[i] * phiP[j]) / lam

        for i in range(4):
            theta_hat[kk, i] = theta[i]

        if cov_stride > 0 and kk % cov_stride == 0:
            idx_chk = kk // cov_stride
            if idx_chk < Pkf_checks.shape[0]:
                for i in range(n):
                    for j in range(n):
                        Pkf_checks[idx_chk, i, j] = Pkf[i, j]
                for i in range(4):
                    for j in range(4):
                        Prls_checks[idx_chk, i, j] = Prls[i, j]

    return x_hat, theta_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def run_dual_estimator(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray, Q_diag: np.ndarray,
                        R: float, x0: np.ndarray, P0_kf: np.ndarray, theta0: np.ndarray,
                        P0_rls: np.ndarray, lam: float, n_eta: int, p: int, n_off: int = 0,
                        gamma_pole: float = 0.0, a1_prior: float = 0.0, a2_prior: float = 0.0,
                        n_epochs: int = 1, n_snapshots: int = 0,
                        save_epochs: set[int] | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], float,
                                   np.ndarray, np.ndarray]:
    n = n_eta + p + n_off
    N = len(y)
    cov_stride = max(1, N // n_snapshots) if n_snapshots > 0 else 0
    n_checks = (N // cov_stride + 1) if cov_stride > 0 else 1
    Pkf_checks = np.zeros((n_checks, n, n))
    Prls_checks = np.zeros((n_checks, 4, 4))
    save_epochs = save_epochs or set()

    theta_cur = np.ascontiguousarray(theta0, dtype=np.float64).copy()
    P0_kf_c = np.ascontiguousarray(P0_kf, dtype=np.float64)
    P0_rls_c = np.ascontiguousarray(P0_rls, dtype=np.float64)
    theta_epoch_hist = np.zeros((n_epochs, 4))
    theta_hat_by_epoch: dict[int, np.ndarray] = {}

    x_hat = theta_hat = y_hat = Pkf_final = Prls_final = innov_final = S_final = None
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        # reuse the seeded initial state every epoch (all zeros except any offset
        # seed) rather than zeroing -- keeps the baseline seed across epochs.
        # (Carrying x forward across epochs instead was tried and measured
        # WORSE on the S4.8 synthetic ground-truth check -- 30-seed comparison,
        # 7/30 ok (reset) vs 5/30 ok (carry-forward), median err 0.883 vs 1.064
        # -- reverted. See INVESTIGATION.md Entry 22.)
        x0_epoch = np.ascontiguousarray(x0, dtype=np.float64).copy()
        x_hat, theta_hat, y_hat, Pkf_final, Prls_final, innov_final, S_final = _dual_estimator_core(
            u, y, phi_ar, Q_diag, float(R), x0_epoch, P0_kf_c.copy(), theta_cur, P0_rls_c.copy(),
            float(lam), n_eta, p, n_off, float(gamma_pole), float(a1_prior), float(a2_prior),
            cov_stride, Pkf_checks, Prls_checks,
        )
        theta_cur = theta_hat[-1].copy()
        theta_epoch_hist[epoch] = theta_cur
        if epoch in save_epochs:
            theta_hat_by_epoch[epoch] = theta_hat.copy()
    elapsed = time.perf_counter() - t0

    return (x_hat, theta_hat, y_hat, Pkf_final, Prls_final, Pkf_checks, Prls_checks,
            theta_epoch_hist, theta_hat_by_epoch, elapsed, innov_final, S_final)


def nis_consistency_check(innov: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """NIS_k = innov_k^2/S_k ~ chi2(1) if Q/R are correctly scaled (Bar-Shalom,
    Li & Kirubarajan, Estimation with Applications to Tracking and Navigation,
    sec 5.4); mean should be ~1. Systematically >1 means Q/R too small (filter
    overconfident); systematically <1 means Q/R too large (filter
    underconfident). Consistency band on the sample mean via the CLT
    (var(chi2(1))=2), reusing the existing WHITENESS_CI z-score rather than
    pulling in scipy.stats for one critical value."""
    nis = innov ** 2 / S
    band = WHITENESS_CI * np.sqrt(2.0 / len(nis))
    mean_nis = float(np.mean(nis))
    return nis, mean_nis, bool(abs(mean_nis - 1.0) <= band)


# ==========================================================================
# 5. Verification suite (math/estimator self-checks, independent of data)
# ==========================================================================

def rls_step_dense(theta: np.ndarray, P: np.ndarray, phi_vec: np.ndarray,
                    target: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Textbook (dense, no numba) RLS step -- spec eqs. 43-45."""
    Pphi = P @ phi_vec
    denom = lam + phi_vec @ Pphi
    K = Pphi / denom
    theta_new = theta + K * (target - phi_vec @ theta)
    P_new = (P - np.outer(K, Pphi)) / lam
    P_new = 0.5 * (P_new + P_new.T)
    return theta_new, P_new


def riv_step_dense(theta: np.ndarray, P: np.ndarray, phi_vec: np.ndarray, z_vec: np.ndarray,
                    target: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Dense recursive instrumental-variable (RIV) step -- the same
    Soderstrom & Stoica recursion as the jitted core's parameter update."""
    Pz = P @ z_vec
    denom = lam + phi_vec @ Pz
    K = Pz / denom
    theta_new = theta + K * (target - phi_vec @ theta)
    P_new = P - np.outer(K, phi_vec @ P)
    P_new = P_new / lam
    return theta_new, P_new


def test_rls_matches_batch_ls(rng: np.random.Generator) -> None:
    """S4.2: on a fixed (Phi, y), RLS with lambda=1 and a large P0 must
    reproduce the batch normal-equations solution theta=(Phi^T Phi)^-1 Phi^T y."""
    N, m = 500, 4
    Phi = rng.normal(size=(N, m))
    theta_star = rng.normal(size=m)
    y = Phi @ theta_star + rng.normal(scale=0.01, size=N)

    theta = np.zeros(m)
    P = 1e6 * np.eye(m)
    for kk in range(N):
        theta, P = rls_step_dense(theta, P, Phi[kk], y[kk], 1.0)

    theta_batch = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
    assert np.allclose(theta, theta_batch, atol=1e-6, rtol=1e-6), \
        "RLS (lambda=1) does not match batch least squares"
    print("[4.2] RLS (lambda=1, large P0) matches batch least squares.")


def test_structured_kf_prop_matches_dense(rng: np.random.Generator) -> None:
    """S4.3: structured P- = A P A^T + Q (the exact function used inside the
    jitted core) must match an explicitly-built dense blkdiag(A_eta,A_beta)
    @ P @ A.T + diag(Q)."""
    n_eta, p = N_ETA, 7
    n = n_eta + p

    a1, a2 = rng.uniform(-1, 1, size=2)
    phi_ar = rng.normal(scale=0.3, size=p)
    M = rng.normal(size=(n, n))
    P = M @ M.T  # random symmetric PSD
    Q_diag = rng.uniform(1e-6, 1e-2, size=n)

    A_eta = np.array([[a1, 1.0], [a2, 0.0]])
    A_beta = np.zeros((p, p))
    A_beta[0, :] = phi_ar
    if p > 1:
        A_beta[1:, :-1] = np.eye(p - 1)
    A = np.zeros((n, n))
    A[:n_eta, :n_eta] = A_eta
    A[n_eta:, n_eta:] = A_beta
    dense = A @ P @ A.T + np.diag(Q_diag)

    temp = np.empty((n, n))
    Pminus = np.empty((n, n))
    _structured_kf_prop(P, Q_diag, phi_ar, a1, a2, n_eta, p, 0, temp, Pminus)

    assert np.allclose(Pminus, 0.5 * (dense + dense.T), atol=1e-9), \
        "structured A P A^T + Q does not match the dense reference"
    print("[4.3] Structured KF covariance propagation matches dense A@P@A.T+Q.")


def reference_dual_estimator_dense(u: np.ndarray, y: np.ndarray, phi_ar: np.ndarray,
                                    Q_diag: np.ndarray, R: float, x0: np.ndarray,
                                    P0_kf: np.ndarray, theta0: np.ndarray, P0_rls: np.ndarray,
                                    lam: float, n_eta: int, p: int, n_off: int = 0,
                                    gamma_pole: float = 0.0, a1_prior: float = 0.0, a2_prior: float = 0.0,
                                    instrument: str = "aux_model"
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """S4.4: independent, readable, dense-matmul dual estimator (plain KF
    predict/update + textbook RIV), no numba, no structured tricks -- used
    to validate the jitted core's arithmetic, and (via `instrument`) to
    compare RIV instrument choices without touching the validated jitted
    production core. n_off appends an optional identity-dynamics offset
    state (H picks it up with +1).

    instrument="aux_model" (default, matches the jitted core): z_k=[eta_aux_{k-1},
    eta_aux_{k-2},u_{k-1},u_{k-2}], eta_aux a noise-free auxiliary simulation
    driven by u (Soderstrom & Stoica IV) -- uncorrelated with the colored
    disturbance leaking into eta_hat.
    instrument="lagged_u": z_k=[u_{k-3},u_{k-4},u_{k-1},u_{k-2}], the classic
    delayed-input instrument -- simpler, valid whenever u is persistently
    exciting and uncorrelated with the disturbance beyond the ARX lag
    structure, but ignores the plant-dynamics information the aux-model
    instrument exploits."""
    if instrument not in ("aux_model", "lagged_u"):
        raise ValueError(f"unknown instrument: {instrument!r}")
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + n_off
    N = len(y)

    x = x0.copy().astype(float)
    Pkf = P0_kf.copy().astype(float)
    theta = theta0.copy().astype(float)
    Prls = P0_rls.copy().astype(float)
    Qmat = np.diag(Q_diag)
    z1, z2 = 0.0, 0.0

    x_hat = np.zeros((N, n))
    theta_hat = np.zeros((N, 4))
    y_hat = np.zeros(N)
    innov_hist = np.zeros(N)
    S_hist = np.zeros(N)

    for kk in range(N):
        if kk > 0:
            u_prev = u[kk - 1]
            a1, a2, b1, b2 = theta
            A_eta = np.array([[a1, 1.0], [a2, 0.0]])
            A_beta = np.zeros((p, p))
            A_beta[0, :] = phi_ar
            if p > 1:
                A_beta[1:, :-1] = np.eye(p - 1)
            A = np.zeros((n, n))
            A[:n_eta, :n_eta] = A_eta
            A[n_eta:n_eta + p, n_eta:n_eta + p] = A_beta
            if n_off == 1:
                A[idx_off, idx_off] = 1.0
            B = np.zeros(n)
            B[0], B[1] = b1, b2

            x = A @ x + B * u_prev
            Pkf = A @ Pkf @ A.T + Qmat
            Pkf = 0.5 * (Pkf + Pkf.T)

        H = np.zeros(n)
        H[0] = -1.0
        H[idx_beta] = 1.0
        if n_off == 1:
            H[idx_off] = 1.0
        innov = y[kk] - float(H @ x)
        S = float(H @ Pkf @ H + R)
        innov_hist[kk] = innov
        S_hist[kk] = S
        K = (Pkf @ H) / S
        x = x + K * innov
        Pkf = Pkf - np.outer(K, H @ Pkf)
        Pkf = 0.5 * (Pkf + Pkf.T)

        eta_hat_k = x[0]
        x_hat[kk] = x
        y_hat[kk] = float(H @ x)

        eta_km1 = x_hat[kk - 1, 0] if kk - 1 >= 0 else 0.0
        eta_km2 = x_hat[kk - 2, 0] if kk - 2 >= 0 else 0.0
        u_km1 = u[kk - 1] if kk - 1 >= 0 else 0.0
        u_km2 = u[kk - 2] if kk - 2 >= 0 else 0.0
        phi_vec = np.array([eta_km1, eta_km2, u_km1, u_km2])

        if instrument == "aux_model":
            z_vec = np.array([z1, z2, u_km1, u_km2])
            z_new = theta[0] * z1 + theta[1] * z2 + theta[2] * u_km1 + theta[3] * u_km2
            z2, z1 = z1, z_new
        else:  # "lagged_u"
            u_km3 = u[kk - 3] if kk - 3 >= 0 else 0.0
            u_km4 = u[kk - 4] if kk - 4 >= 0 else 0.0
            z_vec = np.array([u_km3, u_km4, u_km1, u_km2])

        theta, Prls = riv_step_dense(theta, Prls, phi_vec, z_vec, eta_hat_k, lam)
        theta[0], theta[1], theta[2], theta[3] = _clamp_theta(theta[0], theta[1], theta[2], theta[3])
        if gamma_pole > 0.0:
            theta[0] += gamma_pole * (a1_prior - theta[0])
            theta[1] += gamma_pole * (a2_prior - theta[1])
            theta[0], theta[1] = _clamp_pole_pair(theta[0], theta[1])
        theta_hat[kk] = theta

    return x_hat, theta_hat, y_hat, Pkf, Prls, innov_hist, S_hist


def test_jit_core_matches_dense_reference_and_covariance_sanity(rng: np.random.Generator) -> None:
    n_eta, p = N_ETA, 5
    n = n_eta + p
    N = 150

    phi_ar = np.array([0.6, -0.1, 0.05, 0.02, -0.01])
    theta_true_local = np.array([1.2, -0.4, 0.3, 0.1])
    u = rng.uniform(0, 2.5, size=N)

    A_eta_t, B_eta_t, C_eta_t = theta_to_observer_canonical(theta_true_local)
    eta_true = simulate_ss(A_eta_t, B_eta_t, C_eta_t, u)
    A_beta, C_beta = ar_to_state_space(phi_ar)
    x_beta = np.zeros(p)
    beta_sim = np.zeros(N)
    w = rng.normal(0.0, 0.05, size=N)
    for kk in range(N):
        beta_sim[kk] = (C_beta @ x_beta)[0]
        x_beta = A_beta @ x_beta
        x_beta[0] += w[kk]
    y = beta_sim - eta_true + rng.normal(0.0, 0.05, size=N)

    theta0 = theta_true_local * 1.1
    P0_rls = 1e3 * np.eye(4)
    lam = 0.999
    cov_stride = 1
    n_checks = N // cov_stride + 1

    # Verify the jitted core matches the dense reference for BOTH the plain model
    # (n_off=0) and the offset-augmented model (n_off=1, the real-data fix).
    for n_off in (0, 1):
        n = n_eta + p + n_off
        Q_diag = np.concatenate((np.full(n_eta, 0.1), np.full(p, 0.1)))
        Q_diag[n_eta] = 5.0
        if n_off == 1:
            Q_diag = np.concatenate((Q_diag, [0.01]))
        x0 = np.zeros(n)
        P0_kf = 10.0 * np.eye(n)
        Pkf_checks = np.zeros((n_checks, n, n))
        Prls_checks = np.zeros((n_checks, 4, 4))
        # gamma_pole=0.05 on the second pass also exercises the pole-regularised
        # path; the jitted core and dense reference must still agree.
        gp = 0.0 if n_off == 0 else 0.05
        x_hat_fast, theta_hat_fast, y_hat_fast, Pkf_final, Prls_final, innov_fast, S_fast = _dual_estimator_core(
            u, y, phi_ar, Q_diag, 1e-4, x0, P0_kf, theta0, P0_rls, lam, n_eta, p, n_off,
            gp, 1.2, -0.35, cov_stride, Pkf_checks, Prls_checks)

        (x_hat_ref, theta_hat_ref, y_hat_ref, Pkf_final_ref, Prls_final_ref,
         innov_ref, S_ref) = reference_dual_estimator_dense(
            u, y, phi_ar, Q_diag, 1e-4, x0, P0_kf, theta0, P0_rls, lam, n_eta, p, n_off, gp, 1.2, -0.35)

        assert np.allclose(x_hat_fast, x_hat_ref, atol=1e-8), \
            f"jitted core state trajectory does not match dense reference (n_off={n_off})"
        assert np.allclose(theta_hat_fast, theta_hat_ref, atol=1e-8), \
            f"jitted core parameter trajectory does not match dense reference (n_off={n_off})"
        assert np.allclose(y_hat_fast, y_hat_ref, atol=1e-8), \
            f"jitted core output trajectory does not match dense reference (n_off={n_off})"
        assert np.allclose(Pkf_final, Pkf_final_ref, atol=1e-8), \
            f"jitted core final KF covariance does not match dense reference (n_off={n_off})"
        assert np.allclose(Prls_final, Prls_final_ref, atol=1e-8), \
            f"jitted core final RLS covariance does not match dense reference (n_off={n_off})"
        assert np.allclose(innov_fast, innov_ref, atol=1e-8), \
            f"jitted core innovation trajectory does not match dense reference (n_off={n_off})"
        assert np.allclose(S_fast, S_ref, atol=1e-8), \
            f"jitted core innovation covariance trajectory does not match dense reference (n_off={n_off})"
    print("[4.4] Jitted dual-estimator core matches dense NumPy reference (KF+RLS), n_off in {0,1}.")

    tol = 1e-8
    for idx in range(n_checks):
        Pc = Pkf_checks[idx]
        if np.all(Pc == 0.0):
            continue
        assert np.allclose(Pc, Pc.T, atol=1e-8), "KF covariance snapshot is not symmetric"
        assert np.linalg.eigvalsh(Pc).min() >= -tol, "KF covariance snapshot is not PSD"
    print("[4.6] KF covariance stayed symmetric PSD across the verification run.")


def test_theta_to_params_is_inverse(cfg: Config) -> None:
    """S4.7: theta_to_params must invert params_to_theta to numerical
    precision, for an arbitrary test plant (no real ground truth exists)."""
    k_test, tau1_test, tau2_test = 60.0, 0.05, 0.25
    theta_test = params_to_theta(k_test, tau1_test, tau2_test, cfg.Ts)
    k, tau1, tau2 = theta_to_params(theta_test, cfg.Ts)

    assert np.isclose(k, k_test, atol=1e-6, rtol=1e-6), \
        f"recovered k={k} does not match k_test={k_test}"
    assert np.isclose(tau1, tau1_test, atol=1e-9, rtol=1e-6), \
        f"recovered tau1={tau1} does not match tau1_test={tau1_test}"
    assert np.isclose(tau2, tau2_test, atol=1e-9, rtol=1e-6), \
        f"recovered tau2={tau2} does not match tau2_test={tau2_test}"

    theta_roundtrip = params_to_theta(k, tau1, tau2, cfg.Ts)
    assert np.allclose(theta_roundtrip, theta_test, atol=1e-8, rtol=1e-8), \
        "params_to_theta(theta_to_params(theta)) does not reproduce theta"
    print("[4.7] theta_to_params inverts params_to_theta to numerical precision.")


def test_v1_recovers_synthetic_plant(cfg: Config, dual_cfg: DualConfig) -> None:
    """S4.8: does the full V1 pipeline (heuristic R/q_eta + epoch KF+RIV loop)
    recover a KNOWN synthetic plant? No such ground-truth recovery test
    existed in this repo before -- only math self-consistency tests
    (S4.1-4.7) and separate, non-persisted scratchpad scripts.

    A maximum-likelihood/MAP replacement for the R/q_eta heuristic was
    investigated in depth here and found ill-posed for V1's architecture
    (calibrate_noise_mle, kept below as documented negative-result evidence,
    not used in production -- see INVESTIGATION.md). This test instead
    honestly characterizes what the ACTUAL production pipeline (heuristic
    noise) achieves: measured over a fixed set of synthetic seeds (IID
    uniform u, known AR(3) disturbance), only ~23% (7/30 tested) recover
    k within 30% of truth with the correct sign -- V1's epoch KF+RIV
    attractor (Entry 9) makes recovery seed-dependent, not reliable. A
    single-seed pass/fail assertion would misrepresent that reliability
    either way, so this asserts the pipeline runs cleanly (no crash/NaN)
    and that AT LEAST ONE seed in the fixed set recovers reasonably (a
    floor that catches genuine regressions, not a reliability claim)."""
    k_true, tau1_true, tau2_true, c0_true = 80.0, 0.05, 0.25, 1.10
    phi_ar_true = np.array([0.6, -0.2, 0.1])
    sigma2_true = 0.05
    N = 3000
    seeds = range(6)

    n_ok = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.004, 0.026, size=N)
        eta_true = simulate_arx_recursion(params_to_theta(k_true, tau1_true, tau2_true, cfg.Ts), u)
        A_beta, C_beta = ar_to_state_space(phi_ar_true)
        p = len(phi_ar_true)
        xb = np.zeros(p)
        beta_true = np.zeros(N)
        for kk in range(N):
            beta_true[kk] = (C_beta @ xb)[0]
            xb = A_beta @ xb
            xb[0] += rng.normal(0, np.sqrt(sigma2_true))
        y = c0_true + beta_true - eta_true + rng.normal(0, 0.05, N)

        R_heur = 0.05 * sigma2_true
        q_eta_heur = 0.5 * float(np.var(y))
        result = run_identification(u, y, phi_ar_true, sigma2_true, R_heur, q_eta_heur,
                                     cfg, dual_cfg, cfg.fig_dir / "_synthetic_selfcheck",
                                     make_plots=False)
        assert np.isfinite(result["k"]), f"seed={seed}: recovered k is not finite"
        k_err = abs(result["k"] - k_true) / k_true
        if result["k"] > 0 and k_err < 0.3:
            n_ok += 1

    assert n_ok > 0, f"0/{len(seeds)} seeds recovered k within 30% of truth -- pipeline regression"
    print(f"[4.8] V1 pipeline runs cleanly on synthetic data across {len(seeds)} seeds; "
          f"{n_ok}/{len(seeds)} recovered k within 30% of truth (attractor-limited, "
          f"see INVESTIGATION.md -- not a reliability guarantee).")


def run_verification_suite(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed + 1)
    run_self_consistency_checks(cfg)
    test_rls_matches_batch_ls(rng)
    test_structured_kf_prop_matches_dense(rng)
    test_jit_core_matches_dense_reference_and_covariance_sanity(rng)
    test_theta_to_params_is_inverse(cfg)
    test_v1_recovers_synthetic_plant(cfg, DUAL_CFG)


# ==========================================================================
# 6. Reporting / plotting
# ==========================================================================

def plot_input(t: np.ndarray, u: np.ndarray, path: Path,
                title: str = "Real stimulation input") -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, u, lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("u"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_output_fit(t: np.ndarray, y: np.ndarray, y_hat: np.ndarray, path: Path,
                     y_hat_label: str = "y_hat (dual KF)") -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, y, label="y (measured beta)", lw=1)
    ax.plot(t, y_hat, label=y_hat_label, lw=1, ls="--")
    ax.set_xlabel("time (s)"); ax.set_ylabel("y = beta - eta"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_residual(t: np.ndarray, resid: np.ndarray, path: Path,
                   ylabel: str = "residual", title: str = "Residual vs time") -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(t, resid, lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_input_excitation(u: np.ndarray, Ts: float, tau1_fit: float, tau2_fit: float,
                           tau1_ref: float, tau2_ref: float, path: Path) -> None:
    """Two-panel diagnostic: PSD of u vs plant pole corner frequencies, and autocorrelation
    of u vs plant impulse-response decay -- shows whether u has enough fast content to
    separately excite both lags (poor separation is why a1,a2 -- and hence
    k=(b1+b2)/(1-a1-a2) -- can drift toward the near-zero-denominator corner). Both this
    recording's own final fit (tau*_fit, can itself be corrupted if the fit diverged) and
    the fixed avg-patient prior (tau*_ref, AVG_PATIENT_PARAMS -- independent of any fit) are
    marked, since the fit-based corner is circular evidence on a diverged recording."""
    n = len(u)
    freqs = np.fft.rfftfreq(n, d=Ts)
    U = np.fft.rfft(u - u.mean())
    psd = (np.abs(U) ** 2) / n

    max_lag = min(n - 1, round(6 * max(tau1_fit, tau2_fit, tau1_ref, tau2_ref) / Ts))
    um = u - u.mean()
    denom = np.sum(um ** 2)
    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.sum(um[:n - lag] * um[lag:]) / denom for lag in lags])
    t_lag = lags * Ts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogy(freqs[1:], psd[1:], lw=1)
    for tau, color, tag in ((tau1_fit, "tab:red", "fit tau1"), (tau2_fit, "tab:green", "fit tau2"),
                            (tau1_ref, "tab:red", "avg-patient tau1"),
                            (tau2_ref, "tab:green", "avg-patient tau2")):
        ls = "--" if "fit" in tag else ":"
        f = 1.0 / (2 * np.pi * tau)
        ax1.axvline(f, color=color, ls=ls, label=f"{tag}: {f:.2f} Hz")
    ax1.set_xlabel("frequency (Hz)"); ax1.set_ylabel("PSD(u)"); ax1.legend(fontsize=7)
    ax1.set_title("Input power spectrum vs plant corner frequencies")

    ax2.plot(t_lag, acf, lw=1, label="autocorr(u)")
    ax2.plot(t_lag, np.exp(-t_lag / tau1_fit), lw=1, ls="--", color="tab:red", label="exp(-t/fit tau1)")
    ax2.plot(t_lag, np.exp(-t_lag / tau2_fit), lw=1, ls="--", color="tab:green", label="exp(-t/fit tau2)")
    ax2.plot(t_lag, np.exp(-t_lag / tau1_ref), lw=1, ls=":", color="tab:red", label="exp(-t/avg-patient tau1)")
    ax2.plot(t_lag, np.exp(-t_lag / tau2_ref), lw=1, ls=":", color="tab:green", label="exp(-t/avg-patient tau2)")
    ax2.set_xlabel("lag (s)"); ax2.set_ylabel("normalized autocorrelation"); ax2.legend(fontsize=7)
    ax2.set_title("Input decorrelation vs plant impulse-response decay")

    fig.suptitle("Excitation diagnostic")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_components(t: np.ndarray, eta_hat: np.ndarray, beta_hat: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, beta_hat, label="beta_hat (background)", lw=1)
    ax.plot(t, eta_hat, label="eta_hat (stim response)", lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("estimated component"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_theta_trajectory(t: np.ndarray, theta_hat: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, lab in enumerate(["a1", "a2", "b1", "b2"]):
        ax.plot(t, theta_hat[:, i], lw=1, label=lab)
    ax.set_xlabel("time (s)"); ax.set_ylabel("theta_hat_k")
    ax.set_title("ARX parameter trajectory (final epoch)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_k_tau_vs_epoch(epochs: np.ndarray, k_arr: list[float], tau1_arr: list[float],
                         tau2_arr: list[float], path: Path) -> None:
    """The final (k, tau1, tau2) recovered at the end of every epoch."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(epochs, k_arr, marker="o", ms=3, lw=1, color="tab:blue")
    axes[0].set_ylabel("k")
    axes[1].plot(epochs, tau1_arr, marker="o", ms=3, lw=1, color="tab:green")
    axes[1].set_ylabel("tau1 (s)")
    axes[2].plot(epochs, tau2_arr, marker="o", ms=3, lw=1, color="tab:red")
    axes[2].set_ylabel("tau2 (s)"); axes[2].set_xlabel("epoch")
    fig.suptitle("Final (k, tau1, tau2) at the end of each epoch")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def epoch_convergence_check(k_arr: list[float], tau1_arr: list[float], tau2_arr: list[float],
                             rtol: float) -> tuple[float, float, float, bool]:
    """Relative change in (k, tau1, tau2) between the last two epochs --
    numeric companion to plot_k_tau_vs_epoch. Undefined (nan, not converged)
    with fewer than 2 epochs -- there's no epoch-to-epoch change to measure."""
    if len(k_arr) < 2:
        return float("nan"), float("nan"), float("nan"), False
    def rel(a: list[float]) -> float:
        return abs(a[-1] - a[-2]) / max(abs(a[-2]), 1e-12)
    dk, dt1, dt2 = rel(k_arr), rel(tau1_arr), rel(tau2_arr)
    return dk, dt1, dt2, bool(max(dk, dt1, dt2) <= rtol)


def p_trace_plateau_check(Pkf_checks: np.ndarray, back_frac: float = 0.5,
                           rtol: float = 0.05) -> tuple[np.ndarray, bool]:
    """trace(P) over the back half of recorded snapshots should be flat if
    the KF covariance has reached steady state within one recording pass."""
    traces = np.trace(Pkf_checks, axis1=1, axis2=2)
    nz = traces[traces != 0.0]
    if len(nz) < 2:
        return traces, False
    back = nz[int(len(nz) * (1 - back_frac)):]
    return traces, bool((back.max() - back.min()) <= rtol * abs(back.mean()))


def plot_theta_epochs_stacked(t: np.ndarray, epochs: list[int],
                               theta_hat_by_epoch: dict[int, np.ndarray], path: Path) -> None:
    """One row per selected epoch: that epoch's full theta_hat_k trajectory."""
    fig, axes = plt.subplots(len(epochs), 1, figsize=(8, 2.6 * len(epochs)), sharex=True)
    if len(epochs) == 1:
        axes = [axes]
    labels = ["a1", "a2", "b1", "b2"]
    for row, epoch in enumerate(epochs):
        ax = axes[row]
        traj = theta_hat_by_epoch[epoch]
        for i, lab in enumerate(labels):
            ax.plot(t, traj[:, i], label=lab, lw=1)
        ax.set_ylabel("theta_hat_k")
        ax.set_title(f"epoch {epoch}")
        if row == 0:
            ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Parameter trajectories across selected epochs")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_final_params_bar(results: dict[str, dict], path: Path,
                           include_reference: bool = False) -> None:
    """Final (k, tau1, tau2) per recording, one bar-chart panel each, side
    by side on the same figure. Recordings whose |k| is wildly off the rest
    (the estimator failed to settle within its epoch budget) are dropped
    so one runaway fit doesn't crush the axis for everyone else.
    If include_reference, an extra "avg patient" bar (literature/prior
    values, not fit from this data) is appended in a distinct color."""
    all_names = list(results)
    k_all = np.array([results[name]["k"] for name in all_names])
    median_abs_k = float(np.median(np.abs(k_all))) or 1.0
    # ponytail: 20x the median |k| is a blunt divergence check, not a
    # statistical test -- flags an estimator that clearly never settled,
    # not genuine outliers; raise/lower if recordings legitimately vary
    # in gain by more than that.
    keep = np.abs(k_all) <= 20.0 * median_abs_k
    dropped = [n for n, k in zip(all_names, keep) if not k]
    if dropped:
        print(f"final_params_bar: excluding diverged recording(s) from the plot: {dropped}")

    names = [all_names[i].removesuffix(".csv") for i in range(len(all_names)) if keep[i]]
    k_vals = [results[all_names[i]]["k"] for i in range(len(all_names)) if keep[i]]
    tau1_vals = [results[all_names[i]]["tau1"] for i in range(len(all_names)) if keep[i]]
    tau2_vals = [results[all_names[i]]["tau2"] for i in range(len(all_names)) if keep[i]]

    bar_names = names + ["avg patient"] if include_reference else names
    bar_colors = ["tab:blue"] * len(names) + (["tab:orange"] if include_reference else [])
    title_suffix = ", vs. average-patient reference" if include_reference else ""

    fig, axes = plt.subplots(1, 3, figsize=(4 * len(bar_names), 4))
    for ax, vals, param, title in (
        (axes[0], k_vals, "k", "k"),
        (axes[1], tau1_vals, "tau1", "tau1 (s)"),
        (axes[2], tau2_vals, "tau2", "tau2 (s)"),
    ):
        bar_vals = vals + [AVG_PATIENT_PARAMS[param]] if include_reference else vals
        ax.bar(bar_names, bar_vals, color=bar_colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Final (k, tau1, tau2) per recording{title_suffix}")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ==========================================================================
# 7. Main
# ==========================================================================

def build_kf_setup(u: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray, sigma2_ar: float,
                    q_eta: float, cfg: Config, dual_cfg: DualConfig) -> dict:
    """Shared KF/RIV initial-condition setup (u-normalization, average-patient
    theta0 prior, Q_diag, x0, P0) -- reused by run_identification and any
    diagnostic harness (e.g. instrument-choice comparison) that needs the
    same starting point without duplicating this logic.

    n_off=1: appends a constant/slow-drift offset state so the beta DC
    baseline is no longer forced into the stim-response eta (root-cause fix
    for the real-data negative-k failure -- see PHASE4_ROOT_CAUSE_ANALYSIS.md).

    ponytail: the real input only spans ~0.004-0.026 while eta/beta sit at
    O(1) -- fitting b1,b2 directly against that made the ARX regression
    badly conditioned (verified empirically: RLS covariance would blow up
    in the u-driven direction). Normalize by std(u) for the estimator and
    undo it only when reading back the physical (k,tau1,tau2)."""
    p = len(phi_coeffs)
    n_off = 1
    idx_off = N_ETA + p
    n = N_ETA + p + n_off

    u_scale = float(np.std(u))
    u_norm = u / u_scale

    # Initial theta: the average-patient reference plant (not fit from this
    # recording) -- the RIV update is the only estimator, no batch LS warm start.
    theta0 = params_to_theta(AVG_PATIENT_PARAMS["k"], AVG_PATIENT_PARAMS["tau1"],
                              AVG_PATIENT_PARAMS["tau2"], cfg.Ts)
    theta0[2] *= u_scale
    theta0[3] *= u_scale

    q_off = dual_cfg.q_off_frac * float(np.var(y))
    Q_diag = np.concatenate((build_process_noise_diag(N_ETA, p, q_eta, sigma2_ar, 0.1 * sigma2_ar),
                             [q_off]))
    x0 = np.zeros(n)
    x0[idx_off] = float(np.mean(y))          # seed the offset at the beta baseline
    P0_kf = dual_cfg.p0_kf_state_var * np.eye(n)
    P0_rls = dual_cfg.p0_rls * np.eye(4)
    phi_c = np.ascontiguousarray(phi_coeffs, dtype=np.float64)
    Q_c = np.ascontiguousarray(Q_diag, dtype=np.float64)
    a1_prior, a2_prior = float(theta0[0]), float(theta0[1])

    return {"u_norm": u_norm, "u_scale": u_scale, "theta0": theta0, "Q_diag": Q_c,
            "x0": x0, "P0_kf": P0_kf, "P0_rls": P0_rls, "phi_c": phi_c,
            "a1_prior": a1_prior, "a2_prior": a2_prior, "n": n, "n_off": n_off, "p": p}


def theta_to_params_physical(theta: np.ndarray, u_scale: float, Ts: float) -> tuple[float, float, float]:
    k, tau1, tau2 = theta_to_params_safe(theta, Ts)
    return k / u_scale, tau1, tau2


# ==========================================================================
# Experimental / negative-result code, NOT used by main() -- kept as
# reproducible evidence for INVESTIGATION.md's noise-calibration entry.
# Decoupled (fixed-plant) MLE/MAP calibration of R and q_eta was tried and
# found ill-posed for V1's architecture: beta's own AR process can absorb
# almost any residual regardless of whether the fixed plant is right, so the
# likelihood keeps improving as q_eta->0 at every anchor point and every MAP
# prior width tested. main() still uses the R=0.05x/q_eta=0.5x heuristic.
# ==========================================================================

def _v1_forward_kf_nll(params: np.ndarray, u_norm: np.ndarray, y: np.ndarray,
                        phi_ar: np.ndarray, sigma2_ar: float, n_eta: int, p: int,
                        theta_fixed: np.ndarray, x0: np.ndarray, P0_kf: np.ndarray,
                        q_off: float, return_traj: bool = False
                        ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Single forward KF predict/update pass (no RIV -- the plant `theta_fixed`
    is held at the avg-patient prior) used to calibrate R and q_eta by maximum
    likelihood instead of the old 0.05x/0.5x heuristic multipliers. Same state
    layout as `_dual_estimator_core` (n_off=1 offset state); params=[log_R,
    log_q_eta]."""
    log_R, log_q_eta = params
    R = float(np.exp(log_R))
    q_eta = float(np.exp(log_q_eta))
    idx_beta = n_eta
    idx_off = n_eta + p
    n = n_eta + p + 1

    process_noise_diag = np.concatenate((
        build_process_noise_diag(n_eta, p, q_eta, sigma2_ar, 0.1 * sigma2_ar),
        [q_off],
    ))
    Q = np.diag(process_noise_diag)

    a1, a2, b1, b2 = theta_fixed
    A = np.zeros((n, n))
    A[0, 0], A[0, 1] = a1, 1.0
    A[1, 0] = a2
    A[idx_beta, idx_beta:idx_beta + p] = phi_ar
    if p > 1:
        A[idx_beta + 1:idx_beta + p, idx_beta:idx_beta + p - 1] = np.eye(p - 1)
    A[idx_off, idx_off] = 1.0

    B = np.zeros(n)
    B[0], B[1] = b1, b2

    H = np.zeros(n)
    H[0] = -1.0
    H[idx_beta] = 1.0
    H[idx_off] = 1.0

    x = x0.copy().astype(float)
    P = P0_kf.copy().astype(float)
    n_samples = len(y)
    nll = 0.0
    innov = np.zeros(n_samples) if return_traj else None
    S_traj = np.zeros(n_samples) if return_traj else None

    for sample in range(n_samples):
        if sample > 0:
            x = A @ x + B * u_norm[sample - 1]
            P = A @ P @ A.T + Q
            P = 0.5 * (P + P.T)

        innovation = y[sample] - float(H @ x)
        innov_cov = float(H @ P @ H + R)
        if innov_cov <= 0 or not np.isfinite(innov_cov):
            return 1e18, innov, S_traj

        nll += 0.5 * (innovation * innovation / innov_cov + np.log(2 * np.pi * innov_cov))
        kalman_gain = (P @ H) / innov_cov
        x = x + kalman_gain * innovation
        P = P - np.outer(kalman_gain, H @ P)
        P = 0.5 * (P + P.T)

        if return_traj:
            innov[sample] = innovation
            S_traj[sample] = innov_cov

    return nll, innov, S_traj


# ponytail: unvalidated width for the log-normal MAP prior on (R,q_eta) in
# calibrate_noise_mle -- log(10) puts "10x off the old heuristic" at 1 std,
# matching the plausible range the heuristic's own tuning comment considered
# ({0.5,1,10}x var(y)). Wide enough to let data move the estimate, narrow
# enough to block the R,q_eta->0 degenerate collapse (verified empirically:
# an unregularized fixed-theta MLE collapses to the numerical floor on every
# synthetic seed AND every real recording tested -- see INVESTIGATION.md).
NOISE_PRIOR_LOG_STD = np.log(10.0)


def calibrate_noise_mle(u: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray, sigma2_ar: float,
                         cfg: Config, dual_cfg: DualConfig, out_dir: Path) -> dict:
    """Replace the old R=0.05*sigma2_ar / q_eta=0.5*var(y) heuristics (picked
    to make epoch-to-epoch (k,tau1,tau2) settle visually, never checked
    against actual innovation statistics -- see RESEARCH_NOTEBOOK.md A6) with
    a maximum-a-posteriori calibration of R and q_eta.

    Plain (unregularized) fixed-theta MLE is ill-posed here, not just badly
    anchored: with theta held fixed at ANY point (raw avg-patient prior, or a
    warm-started converged theta -- both tried), beta's own AR process plus
    the offset state have enough flexibility to absorb almost any residual
    regardless of whether the fixed eta prediction is any good, so the
    likelihood keeps "improving" as R,q_eta->0 -- a numerically degenerate
    near-interpolating filter, not a meaningful noise estimate. Verified on
    every synthetic seed and every real recording tested (see
    INVESTIGATION.md). A weak log-normal MAP prior centered on the old
    heuristic (NOISE_PRIOR_LOG_STD) blocks the collapse while still letting
    data pull the estimate away from the heuristic.

    Two stages:
    1. Warm-start: one full run_identification pass with the OLD heuristic
       noise, to get theta off the (likely wrong) avg-patient prior before
       calibrating against it.
    2. MAP-calibrate R, q_eta with theta fixed at that warm-started anchor
       (still one calibration pass, not recoupled into the epoch loop)."""
    y_var = float(np.var(y))
    warm = run_identification(
        u, y, phi_coeffs, sigma2_ar, 0.05 * sigma2_ar, 0.5 * y_var,
        cfg, dual_cfg, out_dir, make_plots=False,
    )
    theta_anchor = warm["theta_final"]

    setup = build_kf_setup(u, y, phi_coeffs, sigma2_ar, 0.5 * y_var, cfg, dual_cfg)
    u_norm = setup["u_norm"]
    x0, P0_kf, p = setup["x0"], setup["P0_kf"], setup["p"]
    q_off = dual_cfg.q_off_frac * y_var

    R0 = 0.05 * sigma2_ar
    q_eta0 = 0.5 * y_var
    log_R0, log_q_eta0 = np.log(R0), np.log(q_eta0)
    p0 = np.array([log_R0, log_q_eta0])
    bounds = [
        (np.log(1e-8 * sigma2_ar), np.log(1e6 * sigma2_ar)),
        (np.log(1e-10), np.log(1e6 * y_var)),
    ]

    def objective(params):
        nll, _, _ = _v1_forward_kf_nll(
            params, u_norm, y, phi_coeffs, sigma2_ar, N_ETA, p, theta_anchor, x0, P0_kf, q_off,
        )
        log_prior = (
            0.5 * ((params[0] - log_R0) / NOISE_PRIOR_LOG_STD) ** 2
            + 0.5 * ((params[1] - log_q_eta0) / NOISE_PRIOR_LOG_STD) ** 2
        )
        return nll + log_prior

    sol = minimize(
        objective, p0,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    R_cal = float(np.exp(sol.x[0]))
    q_eta_cal = float(np.exp(sol.x[1]))

    _, innov, S_traj = _v1_forward_kf_nll(
        sol.x, u_norm, y, phi_coeffs, sigma2_ar, N_ETA, p,
        theta_anchor, x0, P0_kf, q_off, return_traj=True,
    )
    _, nis_mean, nis_ok = nis_consistency_check(innov, S_traj)
    white_pass, white_frac = residual_whiteness_test(innov)

    return {
        "R": R_cal, "q_eta": q_eta_cal, "success": bool(sol.success),
        "nis_mean": nis_mean, "nis_consistent": nis_ok,
        "whiteness_pass": white_pass, "whiteness_frac_exceed": white_frac,
        "R_heuristic": R0, "q_eta_heuristic": q_eta0,
    }


def _warm_up_jit_core(u_norm, y, phi_c, Q_c, R, x0, P0_kf, theta0, P0_rls,
                       dual_cfg, n_off, p, a1_prior, a2_prior, n):
    """Trigger numba compilation on a 5-sample slice before timing the real
    run, so the one-time JIT compile cost isn't charged to the estimate."""
    _dual_estimator_core(
        u_norm[:5], y[:5], phi_c, Q_c, R, x0.copy(), P0_kf.copy(),
        theta0, P0_rls.copy(), dual_cfg.lambda_ff, N_ETA, p, n_off,
        dual_cfg.gamma_pole, a1_prior, a2_prior, 0,
        np.zeros((1, n, n)), np.zeros((1, 4, 4)),
    )


def _compute_fit_diagnostics(theta_epoch_hist, Pkf_final, Prls_final, Pkf_checks,
                              innov_final, S_final, u_scale, cfg, dual_cfg):
    """Post-fit health checks: covariance conditioning, innovation whiteness,
    NIS calibration, epoch-to-epoch convergence, and covariance-trace plateau."""
    min_eig_kf = float(np.linalg.eigvalsh(0.5 * (Pkf_final + Pkf_final.T)).min())

    # RLS covariance conditioning: a large/uncollapsed a1,a2 sub-block means the
    # recursion genuinely never pinned those two directions down (weak excitation/
    # observability); a small, well-conditioned block that still lands theta at the
    # pole clamp corner instead means the recursion is confidently wrong (bias/attractor
    # in the dual KF+RIV loop itself, not an excitation problem) -- see the negative-k
    # investigation this diagnostic was added for.
    Prls_sym = 0.5 * (Prls_final + Prls_final.T)
    eig_a1a2 = np.linalg.eigvalsh(Prls_sym[:2, :2])
    cond_a1a2 = float(eig_a1a2.max() / eig_a1a2.min())
    eig_full = np.linalg.eigvalsh(Prls_sym)
    cond_full = float(eig_full.max() / eig_full.min())

    epoch_k, epoch_tau1, epoch_tau2 = [], [], []
    for theta in theta_epoch_hist:
        k_e, tau1_e, tau2_e = theta_to_params_physical(theta, u_scale, cfg.Ts)
        epoch_k.append(k_e)
        epoch_tau1.append(tau1_e)
        epoch_tau2.append(tau2_e)

    passed_white, frac_white = residual_whiteness_test(innov_final)
    _, nis_mean, nis_ok = nis_consistency_check(innov_final, S_final)
    dk, dt1, dt2, converged = epoch_convergence_check(
        epoch_k, epoch_tau1, epoch_tau2, dual_cfg.epoch_converge_rtol,
    )
    _, plateaued = p_trace_plateau_check(Pkf_checks)

    return {
        "eig_a1a2": eig_a1a2, "prls_a1a2_cond": cond_a1a2,
        "eig_full": eig_full, "prls_full_cond": cond_full,
        "min_eig_kf": min_eig_kf,
        "epoch_k": epoch_k, "epoch_tau1": epoch_tau1, "epoch_tau2": epoch_tau2,
        "whiteness_pass": passed_white, "whiteness_frac_exceed": frac_white,
        "nis_mean": nis_mean, "nis_consistent": nis_ok,
        "epoch_dk": dk, "epoch_dtau1": dt1, "epoch_dtau2": dt2, "epoch_converged": converged,
        "p_trace_plateau": plateaued,
    }


def run_identification(u: np.ndarray, y: np.ndarray, phi_coeffs: np.ndarray, sigma2_ar: float,
                        R: float, q_eta: float, cfg: Config, dual_cfg: DualConfig,
                        out_dir: Path, make_plots: bool = True) -> dict:
    if make_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    setup = build_kf_setup(u, y, phi_coeffs, sigma2_ar, q_eta, cfg, dual_cfg)
    u_norm, u_scale, theta0 = setup["u_norm"], setup["u_scale"], setup["theta0"]
    Q_c, x0, P0_kf, P0_rls = setup["Q_diag"], setup["x0"], setup["P0_kf"], setup["P0_rls"]
    phi_c, a1_prior, a2_prior = setup["phi_c"], setup["a1_prior"], setup["a2_prior"]
    n, n_off, p = setup["n"], setup["n_off"], setup["p"]

    print(f"initial theta (avg-patient prior, u-normalized): {np.array2string(theta0, precision=5)}")
    k0, tau1_0, tau2_0 = theta_to_params_physical(theta0, u_scale, cfg.Ts)
    print(f"initial (k,tau1,tau2): ({k0:.4f}, {tau1_0:.4f}, {tau2_0:.4f})")

    _warm_up_jit_core(
        u_norm, y, phi_c, Q_c, R, x0, P0_kf, theta0, P0_rls,
        dual_cfg, n_off, p, a1_prior, a2_prior, n,
    )

    # First 4 epochs + the final one, to plot individually.
    selected_epochs = sorted(set(range(min(4, dual_cfg.n_epochs))) | {dual_cfg.n_epochs - 1})

    x_hat, theta_hat, y_hat, Pkf_final, Prls_final, Pkf_checks, _, theta_epoch_hist, \
        theta_hat_by_epoch, elapsed, innov_final, S_final = run_dual_estimator(
            u_norm, y, phi_c, Q_c, R, x0, P0_kf, theta0, P0_rls, dual_cfg.lambda_ff,
            N_ETA, p, n_off=n_off, gamma_pole=dual_cfg.gamma_pole, a1_prior=a1_prior,
            a2_prior=a2_prior, n_epochs=dual_cfg.n_epochs, n_snapshots=20,
            save_epochs=set(selected_epochs),
        )

    theta_final = theta_hat[-1]
    print(f"final theta (u-normalized): {np.array2string(theta_final, precision=5)}")
    k_f, tau1_f, tau2_f = theta_to_params_physical(theta_final, u_scale, cfg.Ts)
    print(f"final (k,tau1,tau2): ({k_f:.4f}, {tau1_f:.4f}, {tau2_f:.4f})")

    diag = _compute_fit_diagnostics(
        theta_epoch_hist, Pkf_final, Prls_final, Pkf_checks,
        innov_final, S_final, u_scale, cfg, dual_cfg,
    )
    print(f"final P_kf min eigenvalue: {diag['min_eig_kf']:.3e} (PSD sanity)")
    print(f"final P_rls (a1,a2 block) eigenvalues: {np.array2string(diag['eig_a1a2'], precision=3)}, "
          f"condition number: {diag['prls_a1a2_cond']:.3e}")
    print(f"final P_rls (full 4x4) eigenvalues: {np.array2string(diag['eig_full'], precision=3)}, "
          f"condition number: {diag['prls_full_cond']:.3e}")
    print(f"dual-estimator wall-clock time ({len(u)} samples x {dual_cfg.n_epochs} epochs, "
          f"lambda={dual_cfg.lambda_ff}): {elapsed * 1e3:.2f} ms")
    print(f"innovation whiteness: pass={diag['whiteness_pass']}, "
          f"frac_exceed={diag['whiteness_frac_exceed']:.3f}")
    print(f"NIS: mean={diag['nis_mean']:.3f} (expect ~1.0), consistent={diag['nis_consistent']}")
    print(f"epoch convergence: d(k)={diag['epoch_dk']:.2e} d(tau1)={diag['epoch_dtau1']:.2e} "
          f"d(tau2)={diag['epoch_dtau2']:.2e} converged={diag['epoch_converged']}")
    print(f"P trace back-half plateau: {diag['p_trace_plateau']}")

    if make_plots:
        eta_hat = x_hat[:, 0]
        beta_hat = x_hat[:, N_ETA]
        t = np.arange(len(u)) * cfg.Ts

        plot_input(t, u, out_dir / f"input.{cfg.fig_format}",
                   title="Real stimulation input (concatenated system_id recordings)")
        plot_input_excitation(u, cfg.Ts, tau1_f, tau2_f,
                               AVG_PATIENT_PARAMS["tau1"], AVG_PATIENT_PARAMS["tau2"],
                               out_dir / f"excitation_diagnostic.{cfg.fig_format}")
        plot_output_fit(t, y, y_hat, out_dir / f"output_fit.{cfg.fig_format}")
        plot_components(t, eta_hat, beta_hat, out_dir / f"components.{cfg.fig_format}")
        plot_theta_trajectory(t, theta_hat, out_dir / f"theta_trajectory.{cfg.fig_format}")
        plot_k_tau_vs_epoch(np.arange(len(theta_epoch_hist)), diag["epoch_k"],
                             diag["epoch_tau1"], diag["epoch_tau2"],
                             out_dir / f"k_tau_vs_epoch.{cfg.fig_format}")
        plot_theta_epochs_stacked(t, selected_epochs, theta_hat_by_epoch,
                                   out_dir / f"theta_epochs_stacked.{cfg.fig_format}")

    return {
        "theta_final": theta_final, "k": k_f, "tau1": tau1_f, "tau2": tau2_f,
        "prls_a1a2_cond": diag["prls_a1a2_cond"], "prls_full_cond": diag["prls_full_cond"],
        "min_eig_kf": diag["min_eig_kf"], "elapsed": elapsed,
        "whiteness_pass": diag["whiteness_pass"], "whiteness_frac_exceed": diag["whiteness_frac_exceed"],
        "nis_mean": diag["nis_mean"], "nis_consistent": diag["nis_consistent"],
        "epoch_converged": diag["epoch_converged"], "epoch_dk": diag["epoch_dk"],
        "epoch_dtau1": diag["epoch_dtau1"], "epoch_dtau2": diag["epoch_dtau2"],
        "p_trace_plateau": diag["p_trace_plateau"],
    }


def main() -> None:
    cfg = CFG
    dual_cfg = DUAL_CFG
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running verification suite...")
    run_verification_suite(cfg)

    print(f"\nLoading real recordings from {cfg.data_dir / cfg.delay_dir} ...")
    recordings = load_all_recordings(cfg)
    print(f"loaded {len(recordings)} recordings: {list(recordings)}")

    print("\nFitting beta AR disturbance model pooled across every recording's "
          "no-stimulation periods (one consensus AR model shared by all recordings)...")
    ar_order, phi_coeffs, sigma2_ar = fit_ar_beta_model_pooled(recordings, cfg)

    # ponytail: no repeated-trials data to separately identify sensor noise
    # from process noise, so R and q_eta are set as data-scaled fractions of
    # the AR innovation variance / pooled real output variance. A maximum-
    # likelihood alternative to this heuristic was investigated in depth
    # (calibrate_noise_mle, kept below as documented negative-result
    # evidence, not wired in here) and found ill-posed for this architecture:
    # with the plant held fixed, beta's own AR process can absorb almost any
    # residual, so the likelihood keeps improving as q_eta->0 regardless of
    # anchor point or MAP regularization strength -- see INVESTIGATION.md.
    # R and the AR model are now both pooled (single value shared across
    # recordings, not per-recording); q_eta stays a single global heuristic
    # (0.5x pooled var(y)).
    R = 0.05 * sigma2_ar
    y_all_system_id = np.concatenate([recordings[f][1] for f in cfg.system_id_files])
    q_eta = 0.5 * float(np.var(y_all_system_id))
    print(f"pooled AR({ar_order}) sigma2_ar={sigma2_ar:.4e}, R={R:.4e}")
    print(f"data-driven q_eta={q_eta:.4e} (heuristic; see comment above -- MLE calibration ruled out)")

    results = {}
    for name in cfg.system_id_files:
        u, y = recordings[name]
        print(f"\n--- {name} ({len(u)} samples, {len(u) * cfg.Ts:.1f}s) ---")
        out_dir = cfg.fig_dir / name.removesuffix(".csv")
        results[name] = run_identification(u, y, phi_coeffs, sigma2_ar,
                                            R, q_eta, cfg, dual_cfg, out_dir)

    print("\n=== Per-recording final estimates ===")
    for name, result in results.items():
        print(f"  {name}: k={result['k']:.4f}, tau1={result['tau1']:.4f}, tau2={result['tau2']:.4f}, "
              f"P_rls(a1,a2) cond={result['prls_a1a2_cond']:.3e}")

    plot_final_params_bar(results, cfg.fig_dir / f"final_params_bar.{cfg.fig_format}")
    plot_final_params_bar(results, cfg.fig_dir / f"final_params_bar_vs_avg_patient.{cfg.fig_format}",
                           include_reference=True)

    print(f"\nFigures written under {cfg.fig_dir}")


if __name__ == "__main__":
    main()
