# Methods and Results

Why the original RIV estimator (V1) failed on real recordings, and the three
redesigns (V2, V3, V4) that fixed it.

Notation: $k$ = plant DC gain, $(\tau_1, \tau_2)$ = plant time constants,
$u$ = stimulation input, $y$ = measured output, $\eta$ = the
stimulation-induced component of $y$, $\beta$ = the background disturbance.

---

## 1. V1 architecture

`scripts/estimators/v1/riv_dual_estimation.py` is a dual state+parameter
estimator: a structured Kalman filter separates a recording into an AR
background $\beta$ and a stimulation-driven suppression $\eta$, and a
recursive instrumental-variable (RIV) update re-identifies the plant
$(k, \tau_1, \tau_2)$ from the extracted $\eta$, iterated over 30 epochs from
an average-patient prior.

Each real recording is two columns, `inputs` ($u$) and `outputs` ($y$),
sampled at $T_s = 0.04\,\text{s}$ (25 Hz). Real recordings are not included
in this repository — see the README's Data section.

**Plant** (two first-order lags in series, gain $k$):

$$G(s) = \frac{k}{(\tau_1 s + 1)(\tau_2 s + 1)}$$

**Discrete ARX(2,2)** after ZOH discretisation, $\theta = [a_1, a_2, b_1, b_2]$:

$$y_k = a_1 y_{k-1} + a_2 y_{k-2} + b_1 u_{k-1} + b_2 u_{k-2}$$

$$\text{poles} = \text{roots}(z^2 - a_1 z - a_2), \qquad \tau = \frac{-T_s}{\ln(\text{pole})}, \qquad k = \frac{b_1+b_2}{1-a_1-a_2}$$

`theta_to_params` requires two real poles in $(0,1)$; otherwise it projects
onto the nearest valid two-pole cascade.

**Measurement model:** $y_k = \beta_k - \eta_k + v_k$ — the output is the
background beta minus the stimulation-induced suppression, tracked by a
Kalman filter over the augmented state $x = [\eta_1, \eta_2, \beta_k \ldots
\beta_{k-p+1}]$.

**RIV update:** at each sample, an ARX model is re-fit to the KF's extracted
$\eta$ using a noise-free auxiliary simulation as the instrument
(Söderström–Stoica RIV recursion), then clamped back onto a valid two-pole
cascade. The whole recording is swept 30 times; $\theta$ carries over
between epochs, KF state resets each epoch. This is a coupled fixed-point
scheme — no explicit joint objective is minimised, and convergence isn't
guaranteed.

Open questions carried into the investigation: the AR beta model is fit in
log-space but consumed in a linear filter (§2); the $\beta/\eta$ split may
not be identifiable from one scalar signal; the epoch loop has no proven
objective; and near the pole clamp, $k=(b_1+b_2)/(1-a_1-a_2)$ can flip sign
on noise.

---

## 2. Mathematical verification

Every implementation claim was checked against an independent derivation —
where the code uses `python-control`, verification used a different path
(`scipy.signal.cont2discrete`, closed-form poles, a from-scratch KF+RIV) so
agreement isn't self-referential.

| Item verified | Independent method | Result |
|---|---|---|
| ZOH continuous→ARX(2,2) map | `scipy.signal.cont2discrete` | exact (Δθ = 1.7e-14) |
| discrete poles = $e^{-T_s/\tau}$ | closed form | exact (1e-9) |
| DC gain $(b_1+b_2)/(1-a_1-a_2) = k$ | closed form | exact |
| $\theta \leftrightarrow (k,\tau_1,\tau_2)$ inverse | round-trip, 3×3×3 grid | exact (7e-15 rel.) |
| KF predict/update + RIV recursion | from-scratch textbook KF+RIV | exact (Δ = 0.0) |
| $k_{phys} = k_{norm}/\text{std}(u)$ | closed form | exact |
| beta AR model domain | log vs. linear refit on real data | discrepancy (D1) |

The math-implementation chain is faithful everywhere tested. The one
substantive finding is a modelling inconsistency, not a coding bug:

**D1 — the beta AR model is fit in log domain but consumed in a linear
filter.** Refitting the same no-stim segments in each domain:

| quantity | log domain (used) | linear domain (correct here) |
|---|---|---|
| dominant β pole \|z\| | 0.681 | 0.592 |
| implied β time constant | 0.104 s | 0.076 s (~37% shorter) |
| innovation variance σ² | 3.03e-2 | 6.73e-2 (~2.2×) |

This biases every recording's $\beta/\eta$ split. Two corollaries: **D2**,
there's no DC-offset/level state even though linear $y$ has an O(1) mean
while $\beta$ and $\eta$ are both zero-mean by construction; **D3**, the
log-domain innovation variance seeds the linear filter's process noise,
mis-scaling it by ~2×. A fourth item, D4, is purely interpretive: `P_rls` is
read as a covariance in diagnostics but isn't symmetric under IV, so it's a
conditioning proxy, not a real covariance — no effect on estimates.

The ARX map, pole/gain algebra, the parameter inverse, KF recursion, and
u-normalisation are all exact as implemented — the real-data failure is not
an algebra bug. It lives in the modelling layer (D1–D3) and in the behaviour
the correct equations produce under real data.

---

## 3. Baseline performance and root cause

All six real recordings return negative $k$ (−18 to −688), with the ARX
poles pinned at the clamp corner ($a_1{\approx}0.95$, $a_2{\approx}0$). On
self-consistent synthetic data the same estimator recovers $k$ to <0.4%
error with the correct sign — the algorithm works when its assumptions hold;
the failure is real-data-specific.

| recording | k | τ1 (s) | τ2 (s) | final a1 | final a2 |
|---|---|---|---|---|---|
| system_id_1 | −39.54 | 0.0067 | 0.7103 | 0.9477 | −0.0024 |
| system_id_2 | −40.94 | 0.0069 | 0.7751 | 0.9528 | −0.0029 |
| system_id_3 | −687.6 | 0.0058 | 0.7798 | 0.9510 | −0.0010 |
| system_id_4 | −25.28 | 0.0071 | 0.7719 | 0.9530 | −0.0033 |
| system_id_5 | −23.03 | 0.0080 | 0.7620 | 0.9554 | −0.0062 |
| system_id_long_run | −18.31 | 0.0063 | 0.7784 | 0.9517 | −0.0017 |

τ2 sits at the pole clamp ceiling on every recording (a clamp artefact, not
an estimate); τ1 collapses toward 0; $1-a_1-a_2 \approx 0.05$ sits on the
edge of collapse. On system_id_1, output NRMSE is 0.002 (near-perfect fit)
yet 65% of innovation ACF lags exceed the whiteness band — an
over-parameterised model that fits any trace, so fit quality carries no
information about the parameters.

Eight initial guesses spanning $k_0 \in \{10,62,150,300\}$ all converge to
the same point to 3 significant figures — the estimator relaxes to a fixed
point set by the data and model structure, not a search that could be
rescued by better initialisation.

**Root cause, found by a single-variable intervention** (subtract mean(y)
before estimation):

| recording | k (as-is) | η̂ mean (as-is) | k (de-meaned y) | η̂ mean (de-meaned) |
|---|---|---|---|---|
| system_id_1 | −39.5 | −1.08 | +0.61 | 0.01 |
| system_id_2 | −40.9 | −1.19 | +0.94 | 0.00 |
| system_id_3 | −687.6 | −2.24 | −1.71 | 0.00 |
| system_id_4 | −25.3 | −0.99 | +6.43 | 0.00 |
| system_id_5 | −23.0 | −1.04 | +3.45 | 0.00 |
| system_id_long_run | −18.3 | −0.94 | +5.20 | 0.01 |

As-is, $\hat\eta$'s mean tracks $-\text{mean}(y)$ — $\eta$ is carrying the
beta baseline, not the stimulation response. The disturbance model de-means
each no-stim segment before its AR fit, discarding the O(1) positive
baseline; $\eta$ is the KF's only remaining state with a persistent offset
degree of freedom, so it absorbs the baseline instead. The RIV then finds no
real u-coherent gain (b1+b2≈0, sign set by noise) and pushes $a_1$ to the
clamp, collapsing the DC-gain denominator.

**Ranked contributors:**

| rank | contributor | evidence | confidence |
|---|---|---|---|
| 1 | beta baseline absorbed by η (no offset state) | de-mean flips sign 5/6 | 90% |
| 2 | pole clamp corner, near-zero DC denominator | $1-a_1-a_2\approx0.05$ | 90% |
| 3 | global attractor in KF↔RIV iteration | θ₀-independence | 90% |
| 4 | over-parameterisation / non-identifiable τ | NRMSE 0.002 + whiteness fail | 75% |
| 5 | log/linear disturbance-domain mismatch (D1) | β τ 37% off, σ² 2.2× | 95% present |

Ranks 1–3 are one causal chain (baseline → clamp → attractor): a structural
modelling failure, not a coding bug or an initialisation problem.

---

## 4. Is the plant even identifiable?

Fisher Information of the stimulation response w.r.t.
$p=(\log k, \log \tau_1, \log \tau_2)$, on each recording's real $u$:

| input | cond(J) | sv1 | sv2 | sv3 (weak dir) |
|---|---|---|---|---|
| system_id_1/2 | 4.6e1 | 27 | 4.7 | 0.59 |
| system_id_3/4/5 | 0.9–1.0e2 | ~29 | ~4.3 | 0.28–0.32 |
| long_run | 8.6e1 | 41 | 6.2 | 0.48 |
| rich white u | 5.8e1 | 24 | 1.7 | 0.41 |

CRB correlation on system_id_1: $\text{corr}(\log\tau_1,\log\tau_2)=-0.87$.

**Full rank, moderate conditioning (~50–100): the plant is identifiable.**
This rejects the hypothesis that the data can't support the estimate — the
failure is the estimator, not the data (confidence 85%). $k$ is
well-identified; $\tau_1,\tau_2$ are partially confounded but separable.

Rich white $u$ barely improves conditioning over real $u$ (58 vs 46) — the
confounding is intrinsic to the two-lag structure, not excitation-limited.
But excitation *is* weak in the fast band (<1% of input power above the fast
corner frequency), which specifically bounds $\tau_1$ precision: the
estimator's real-data $\tau_1{\approx}0.006\text{s}$ implies a corner above
Nyquist — direct evidence it's a clamp artefact (confidence 80%).

Under the physical log-parameterisation, poles $z_i=e^{-T_s/\tau_i}\in(0,1)$
and $k=e^{\log k}>0$ **by construction** — negative $k$, unstable poles, and
complex poles become structurally impossible (confidence 90%).

Every element of a fix is now evidence-motivated: model the baseline, use a
physical log-parameterisation, and replace the two-stage attractor with a
direct grey-box estimator.

---

## 5. Why grey-box output-error (V2)

The problem needs: physical/constrained parameters ($k>0$, stable real
poles), an explicit baseline, robustness to a wrong disturbance model (D1),
one attractor-free objective, and honest uncertainty given the weak
$\tau_1,\tau_2$ identifiability.

| Method | k>0/poles | Baseline | Robust to wrong noise model | Verdict |
|---|---|---|---|---|
| **Grey-box Output-Error (OE)** | via log-param | explicit offset | yes, if $u\perp$ disturbance | **chosen** |
| Prediction-Error Method | yes | yes | model-dependent | strong; OE is its low-assumption special case |
| Maximum Likelihood / EM | yes | yes | yes, if noise modelled | excellent once a disturbance model is trusted |
| Grey-box + Bayesian prior | yes | yes | yes | best route to usable τ; recommended extension |
| Subspace ID (N4SID/MOESP) | no | needs de-trend | moderate | can't enforce k>0 |
| EKF/UKF joint state+param | weak | needs offset | no | reinherits V1's tuning burden |
| ARX/ARMAX (V1 family) | no (fit then invert) | no | biased under coloured noise | the inversion is the blowup's origin |

OE estimates the plant from $u \to y$ alone and stays consistent whenever
$u$ is uncorrelated with the disturbance (Ljung, *System Identification*,
ch. 8) — removing D1's dependence entirely. Empirically: $k = 80.15 \pm
0.87$ with no disturbance, $78.3 \pm 5.4$ under the full disturbance —
variance grows, bias doesn't. Log-parameterisation gives the constraints for
free, with no attractor (8 random starts converge to the same optimum).

References: L. Ljung, *System Identification: Theory for the User* (2nd
ed., 1999); T. Söderström & P. Stoica, *System Identification* (1989); R.
Pintelon & J. Schoukens, *System Identification: A Frequency Domain
Approach* (2012); T. Schön, A. Wills, B. Ninness, *Automatica* (2011); R.
Bellman & K. J. Åström, *Math. Biosci.* (1970); G. Pillonetto et al.,
*Automatica* (2014); C. V. Rao, J. B. Rawlings, D. Q. Mayne, *IEEE TAC*
(2003).

### V2 design (`scripts/estimators/v2/riv_v2_greybox.py`)

Model: $\eta_k$ = cascade of first-order lags (gain $k$) driven by $u$;
$y_k = c_0 - \eta_k + d_k + e_k$ ($c_0$ = baseline offset, $d_k$ = coloured
disturbance left in the residual, $e_k$ = sensor noise). Parameters
$p = [\log k, \log \tau_1, (\log \tau_2), c_0]$, fit by output-error least
squares (Levenberg–Marquardt). No Kalman filter, no RIV, no clamps, no
epoch loop.

- **Order selection:** AIC between one and two lags — a single lag wins on
  5/6 recordings.
- **Uncertainty:** CRB covariance from the Gauss–Newton Jacobian, plus a
  block bootstrap for coloured residuals. A coherence flag
  ($|corr(u,y)| < 0.15$) marks recordings where $u$ barely drives $y$.
- **Limitations:** $(\tau_1,\tau_2)$ remain weakly identifiable on this
  data; OE's residual-domain disturbance means wider CIs than a correct
  full-noise ML (inefficient, not biased); $c_0$ is constant (a drifting
  baseline would need a trend term).

---

## 6. V1 vs V2 validation

**Synthetic recovery** (truth $k=80,\tau_1=0.04,\tau_2=0.30$):

| estimator | k | τ1 | τ2 |
|---|---|---|---|
| V1 dual KF+RIV | 80.2 | 0.007 (clamp) | 0.37 |
| **V2 grey-box OE** | **80.07** | **0.040** | **0.303** |

**Monte-Carlo recovery** (72 fits, median relative error):

| condition | k | τ1 | τ2 |
|---|---|---|---|
| ideal (rich white u, 0.2× disturbance) | 2.7% | 4.6% | 0.7% |
| realistic (real u, 1.0× disturbance) | 2.1% | 9.9% | 5.7% |

$k$ stays around 2% error across the whole grid; τ degrades with SNR, τ1
first, τ2 next, k last — matching §4's identifiability analysis.

**Head-to-head** (12 truths × 3 seeds, real u):

| case | est | k | τ1 | τ2 | negative-k runs |
|---|---|---|---|---|---|
| no baseline (V1's favourable case) | V1 | **2.7** | 74.5 | 14.9 | 0/36 |
| | V2 | 6.6 | **38.5** | **9.9** | 0/36 |
| with baseline (c0=1.1, like real data) | V1 | **79.4** | 58.2 | 63.6 | **12/36** |
| | V2 | **6.6** | **38.5** | **9.9** | 0/36 |

When V1's disturbance model is exactly correct and there's no baseline, it's
statistically more efficient on $k$ (2.7% vs 6.6%) — the classic
efficiency/robustness tradeoff. But add the baseline real physiology has,
and V1 collapses even on synthetic data (79% k error, a third of runs
negative) while V2 is unchanged — the root cause reproduced end-to-end
under known ground truth.

**Real data:**

| recording | V1 k | V2 k (95% bootstrap CI) | V2 order | corr(u,y) |
|---|---|---|---|---|
| system_id_1 | −39.5 | 44.3 [22.6, 61.6] | 2 | +0.08 ⚠ |
| system_id_2 | −40.9 | 45.0 [22.4, 65.3] | 2 | +0.00 ⚠ |
| system_id_3 | −687.6 | 29.3 [24.3, 33.9] | 1 | −0.37 |
| system_id_4 | −25.3 | 44.3 [39.2, 50.0] | 1 | −0.56 |
| system_id_5 | −23.0 | 47.4 [43.7, 51.7] | 1 | −0.59 |
| system_id_long_run | −18.3 | 40.5 [37.2, 43.0] | 1 | −0.56 |

Every V1 estimate is negative; every V2 estimate is positive. On the four
recordings where $u$ genuinely drives $y$, V2's $k$ clusters tightly (median
≈44); the two flagged recordings get wide CIs instead of confident
nonsense. V2 is a strictly superior methodology for recovering $k$ here
(90% confidence); $(\tau_1,\tau_2)$ recoverable from this data by any method
is ~30% — a data limit, not an algorithm limit.

---

## 7. Tuning the repaired V1

Sensitivity analysis on realistic synthetic data (known truth, $c_0=1.1$,
real $u$, full disturbance):

| knob | default | sensitivity |
|---|---|---|
| lambda_ff | 0.9999 | high |
| q_eta_frac | 0.5 | high |
| p0_kf | 10 | high |
| p0_rls | 0.3 | high |
| q_off_frac | 1e-3 | medium |
| R_frac | 0.05 | low |
| qb_nom / qb_oth | 1.0 / 0.1 | low |
| n_epochs | 30 | low (plateaus ≥10) |

The covariance/forgetting knobs dominate; noise-scale knobs barely matter,
and there's a k-accuracy vs. positive-k tradeoff. Refitting the beta AR
model in the linear domain instead of log (the D1 fix) sharply reduces
scatter (max |k| across recordings: 388 → 84).

Adopted: `q_off_frac` 1e-3→1e-2, AR domain log→linear.

| recording | V1 original | +offset state | +tuned |
|---|---|---|---|
| system_id_1 | −39.5 | 74.4 | 65.5 |
| system_id_2 | −40.9 | 48.9 | 43.4 |
| system_id_3 | −687.6 | −10.7 | −4.6 (150× better) |
| system_id_4 | −25.3 | 12.5 | 19.8 |
| system_id_5 | −23.0 | 4.5 | 3.0 |
| system_id_long_run | −18.3 | 62.1 | 14.1 |

0/6 → 5/6 positive. The residual scatter traces to the ARX→params inversion
itself, whose near-singular denominator no amount of tuning removes.

---

## 8. V3: maximum-likelihood grey-box

`scripts/estimators/v3/riv_v3_pem.py` adds back what V2 left out — a
coloured-disturbance noise model — inside one joint objective:

$$y_k = c_0 + \beta_k - \eta_k + e_k$$

minimised by L-BFGS over $(\log k, \log\tau_1, \log\tau_2, c_0, \log
q_\beta, \log R)$ using the Kalman filter's one-step innovation likelihood.
No RIV, no ARX inversion, no epoch loop.

| condition | V1-tuned | V2 (OE) | V3 (PEM/ML) |
|---|---|---|---|
| clean (c0=0, rich u, 0.2× disturbance) | 63.8 (75%) | 3.1 (100%) | 3.6 (100%) |
| realistic (c0=1.1, real u, 1× disturbance) | 27.6 (100%) | 12.8 (100%) | 14.2 (100%) |

V2 and V3 are statistically tied on accuracy; both beat V1 by a wide margin.

| recording | V1 orig | V1 tuned | V2 | V3 |
|---|---|---|---|---|
| system_id_1 | −39.5 | 65.5 | 44.3 | 71.3 |
| system_id_2 | −40.9 | 43.4 | 45.0 | 48.8 |
| system_id_3 | −687.6 | −4.6 | 29.3 | 29.3 |
| system_id_4 | −25.3 | 19.8 | 44.3 | 44.3 |
| system_id_5 | −23.0 | 3.0 | 47.4 | 47.4 |
| system_id_long_run | −18.3 | 14.1 | 40.5 | 39.9 |
| positive-k | 0/6 | 5/6 | 6/6 | 6/6 |
| NRMSE (informative recs) | — | — | ~0.80 | **~0.33** |

V3 lands in the same cluster as V2 (29–48) with 2.4× lower prediction error,
since it models the disturbance instead of leaving it in the residual.
Recommended as the identification method going forward (85% confidence it's
the best of the four for recovering $k$ on this data).

---

## 9. Further V1 tuning attempts (mostly negative)

Kept because they bound what's achievable inside the KF+RIV architecture:

- **Pole-regularised RIV — rejected.** Pulling $a_1,a_2$ toward the physical
  prior trades real for synthetic performance: the prior itself sits near
  the unit-sum region, so anchoring toward it pushes $a_1$ toward the clamp
  corner, not away from it.
- **Lower POLE_MAX — rejected as default.** 0.85 is better on real data
  (6/6 positive) but worse on synthetic (positive rate 100%→91%); the
  default stays 0.95 since the goal is best-on-both.
- **Warm start from a batch fit — not pursued.** The fixed point is
  θ₀-independent (§3), so a better warm start can't change the outcome.
- **Numba cache hazard.** `@njit(cache=True)` served stale binaries across
  rapid edits during tuning; clear the cache whenever the core changes.

Every additional lever trades real-data sign for synthetic accuracy — the
residual failure is structural to the ARX inversion and the two-stage
attractor, removable only by leaving that architecture (V2 or V3).

---

## 10. Why V1 works on semi-synthetic data but not real data

**Ruled out as the dominant cause:** real beta's own statistics (a
nonparametric resample of real beta recovers $k$ the same as a parametric
AR draw, 16/36 vs 17/36 "ok" runs); real excitation weakness (semi-synthetic
uses the same real $u$ and still recovers well); baseline drift magnitude
alone (added at up to 50% of std(y), median $k$ stays in 55–95 across the
whole sweep — nowhere near real data's actual failure); R/q_eta
miscalibration (present even in fully self-consistent synthetic data, so it
doesn't explain the real-vs-synthetic gap specifically).

**The standout finding — beta-eta coupling.** Using V2 to construct a beta
proxy ($y - (c_0 - \hat\eta)$) and comparing its variance stim-ON vs no-stim:

| recording | var(beta_proxy) on/off | AR(3)-resid-var on/off |
|---|---|---|
| system_id_3 | 0.51 | 0.20 |
| system_id_4 | 0.37 | 0.20 |
| system_id_5 | 0.40 | 0.27 |
| system_id_long_run | 0.41 | 0.31 |
| median | 0.40 | 0.24 |

Beta's own variance is roughly half as large during stimulation, and its
short-range unpredictability drops to roughly a quarter — consistent across
every recording. This matches DBS's known mechanism (suppressing beta-band
oscillatory power, not just adding a counter-signal): the additive split
$y=\beta-\eta$, shared by V1/V2/V3, may be the wrong functional form. Two
de-confounding designs — an out-of-sample half-split (ratios 0.411/0.268)
and an alternating-block design spanning each full recording (0.401/0.282)
— both converge within ~5% of the in-sample numbers, ruling out both the
model-artifact and non-stationarity explanations (confidence 85%, up from
65% across the three rounds).

**Inconclusive:** non-stationarity. A mild same-direction trend on 3/6
recordings (R²≈0.25) doesn't survive a pooled early-vs-late test across the
4 recordings with enough data (n=4, p=0.25) — a sample-size ceiling.

**Architectural items tested:** per-epoch KF state reset vs. carry-forward
is a real, moderate-to-large lever (up to 6× k change on one recording), but
adopting carry-forward was tried and rejected — a 30-seed ground-truth
comparison showed it performs worse (7/30 ok vs 5/30 ok). Reverted. q_eta
pooling (global vs. per-recording) is confirmed minor (<15% change).

**A non-additive architecture (V4)**
(`scripts/estimators/v4/riv_v4_coupled_suppression.py`), $y_k =
g_k(c_0+\beta_k)$, $g_k=e^{-\eta_k}$: ground-truth self-check passes on all
three parameters (k err 2.0%, τ1 err 2.3%, τ2 err 0.2%); on real data, $k$ is
positive and consistent (23–40, matching V2/V3) with NIS 0.999–1.000 on all
6 recordings — the first architecture besides V3 to hit well-calibrated
innovations. But τ1/τ2 collapse to near-zero on the informative recordings —
confirmed as a genuine global optimum, not a bug. A follow-up semi-synthetic
test (same real $u$ and bootstrapped beta, generated under V4's own correct
model) recovers τ close to truth (τ1 median 0.049–0.071 vs. truth 0.05; τ2
median 0.183–0.234 vs. truth 0.25) — a different regime from real data's
collapse (0.002–0.016), confirming the collapse is caused by real data's own
residual misspecification, not the architecture (85% confidence, up from
30%). The whiteness failure and the τ-collapse are now two independent
symptoms pointing at the same gap: some structure — plant order, or missing
periodic content during active stimulation — is missing from every
architecture tried. Identifying it is the highest-value remaining work, not
testing further architectures.

---

## 11. Does the RIV instrument need to update online?

V1's RIV update needs an auxiliary instrument $z_k$, normally re-simulated
every sample from the live $\theta$ estimate. Tested on a 50 Hz
semi-synthetic setup (rich multisine $u$, true plant $k=62.11,
\tau_1=0.05,\tau_2=0.25$, 80,000 samples × 30 epochs), varying only how
often the instrument refreshes:

| Variant | Refresh period | Success (/10 seeds) | Typical error |
|---|---|---|---|
| Adaptive (baseline) | every sample | 10/10 | 0.034 |
| Decimated | 2–1000 samples | 10/10 (flat) | 0.027–0.034 |
| Frozen, initial guess (fully offline) | never | 10/10 | 0.032 |
| Frozen, true parameters | never | 10/10 | 0.034 |
| Lagged input (no plant info, control) | — | 0/10 | 4.3 (~130× worse) |

From every sample up to never again after the first, the estimator converges
with essentially the same accuracy. Only the instrument that dropped all
plant-dynamics information failed. What makes it work is that it's *some*
reasonable simulation of the plant's response, not that it tracks the
estimator's current belief in lockstep. Caveat: tested only under favourable
conditions (rich excitation, good initial guess, abundant data); an earlier
related result didn't carry over from favourable to harder conditions, so
this should be re-checked on real data before treating it as general.

---

## 12. Recommendations

**Verified findings:** the V1 recursion math is exact (§2) — the failure is
not an algebra bug. The negative-$k$ failure root-causes to a missing
baseline/offset state (§3), producing a global attractor independent of
initialisation. The data are sufficient to identify $k$ (§4, FIM full rank);
$(\tau_1,\tau_2)$ are intrinsically confounded — a data limit, not an
estimator one. V2/V3 fix the failure: $k>0$ on 6/6, consistent, recovering
the full plant on synthetic ground truth (§6).

**Adopt for practice:** V2 or V3 for $k$; report per-recording data-quality
flags and bootstrap CIs (system_id_1/2 have no usable u–y coherence);
treat the plant as first-order unless AIC says otherwise; report a single
effective $\tau$ with a CI rather than two separately-trusted lags.

**Future work:** a Bayesian/MAP prior on the fast lag to regularise the
τ1–τ2 confounding; excitation redesign (fast-band content) to improve τ
precision, though it won't remove the intrinsic confounding; investigate the
missing structure implicated by §10's whiteness failure and V4's
τ-collapse — the highest-value open question; investigate the beta-eta
coupling finding's implication for the additive model shared by all three
estimators.

**Bottom line:** V1 failed because it forced the beta baseline into the
stimulation state and then inverted an unconstrained ARX fit through a
near-singular map — a structural error, not a bug, and inescapable by
tuning. V2/V3 remove that chain by construction and recover a physically
valid, uncertainty-quantified gain on the recordings that carry the
information, while honestly reporting that the two time constants aren't
identifiable from the present excitation.
