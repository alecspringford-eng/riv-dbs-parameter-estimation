# Methods and Results

Why the original RIV estimator (V1) failed on real recordings.

Notation: $k$ = plant DC gain, $(\tau_1,\tau_2)$ = plant time constants,
$u$ = stimulation input, $y$ = measured output, $\eta$ = stimulation-induced
component of $y$, $\beta$ = background disturbance.

## V1 and its failure

`scripts/estimators/v1/riv_dual_estimation.py` splits each recording into
background $\beta$ and stimulation response $\eta$ via a Kalman filter, then
re-fits an ARX(2,2) plant $(k,\tau_1,\tau_2)$ from $\eta$ with a recursive
instrumental-variable (RIV) update, swept over 30 epochs.

Every equation was independently re-derived and matches exactly (ZOH
mapping, pole/gain algebra, KF+RIV recursion), the implementation is not
buggy. But all six real recordings return **negative** $k$ (−18 to −688),
while the same code recovers $k$ to <0.4% error on synthetic data.

**Root cause:** the disturbance model de-means $\beta$'s no-stim segments,
discarding the real O(1) positive baseline. $\eta$ is the only state left
with a free offset, so it absorbs the baseline instead of the true
stimulation response. RIV then finds no real $u$-coherent gain, pushes the
ARX pole $a_1$ to its clamp, and the DC-gain denominator $1-a_1-a_2\to0$,
flipping the sign. Verified directly: de-meaning $y$ before estimation flips
$k$ positive on 5/6 recordings. A secondary issue (D1) compounds it, the
beta AR model is fit in log space but used in a linear filter, biasing the
$\beta/\eta$ split by ~2× in noise variance.

**Is the plant identifiable at all?** Yes: Fisher Information analysis
shows full rank, moderate conditioning (~50–100) on real $u$. $k$ is
well-identified; $\tau_1,\tau_2$ are partially confounded
($\text{corr}(\log\tau_1,\log\tau_2)=-0.87$) because real excitation has
<1% power above the fast corner frequency, a data limit, not an estimator
bug.

## Partial fix

A partial V1 fix (add an offset state + linear-domain AR refit) recovers
5/6 positive-$k$, but residual scatter remains because the ARX→parameter
inversion itself is near-singular, not fixable by further tuning.

## Open question

$\beta$'s own variance drops ~2× during stimulation, inconsistent with the
additive $y=\beta-\eta$ split V1 assumes. Some structure in the disturbance
model is still missing; this is the highest-value remaining question.

## Recommendation

Report per-recording coherence flags and bootstrap CIs; treat the plant as
first-order unless AIC says otherwise; report one effective $\tau$ with a CI
rather than two separately-trusted lags. Full derivations, tables, and
rejected tuning attempts are in git history (this file was condensed from a
longer working log).
