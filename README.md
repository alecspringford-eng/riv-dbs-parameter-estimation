# RIV: Recursive Instrumental-Variable Plant Identification

System identification of a two-lag stimulation/response plant (gain $k$, time
constants $\tau_1, \tau_2$) from input/output recordings, using a recursive
instrumental-variable (RIV) Kalman-filter estimator and three grey-box
redesigns that fix its failure modes on real data.

The problem: a stimulation input $u$ drives a response through a plant

$$G(s) = \frac{k}{(\tau_1 s + 1)(\tau_2 s + 1)}$$

observed as

$$y = \beta - \eta + \text{noise}$$

where $\beta$ is a background disturbance and $\eta$ is the plant's response
to $u$. The goal is to recover $(k, \tau_1, \tau_2)$ from $(u, y)$ alone.

## Key findings

The original estimator (V1, a dual Kalman-filter + RIV design) works on
idealized synthetic data but returns physically impossible (negative) gains
on every real recording tested. The root cause, confirmed by a chain of
targeted experiments: the disturbance model has no baseline/offset state, so
the real signal's DC level gets absorbed into the wrong internal state,
which drives the parameter estimate into a near-singular region of its own
inversion:

$$k = \frac{b_1+b_2}{1-a_1-a_2}$$

This is a modelling error, not a numerical bug — the underlying recursion
algebra is independently verified exact (§2 of the methods doc).

Three redesigns were built and benchmarked against it:

- **V2** (`scripts/estimators/v2/riv_v2_greybox.py`) — output-error grey-box with
  a physical log-parameterisation and explicit offset. Fixes the sign failure
  on 6/6 real recordings.
- **V3** (`scripts/estimators/v3/riv_v3_pem.py`) — adds a maximum-likelihood
  noise model on top of V2, for the best prediction accuracy of the four
  methods.
- **V4** (`scripts/estimators/v4/riv_v4_coupled_suppression.py`) — an
  exploratory multiplicative (rather than additive) coupling model, motivated
  by a finding that the disturbance's own variance drops under stimulation.

Full methodology, evidence tables, and confidence levels: **[docs/METHODS_AND_RESULTS.md](docs/METHODS_AND_RESULTS.md)**.

## Requirements

Python 3.12+, and:

```
pip install -r requirements.txt
```

(`numpy`, `scipy`, `matplotlib`, `numba`, `pandas`, `control`, `pytest`.)

## Usage

`scripts/` is organized by role, then by topic:

```
scripts/
  estimators/            the 8 estimator implementations -- run these directly
    v1/                  original dual Kalman-filter + RIV estimator, and its log-param variants
    v2/                  grey-box output-error estimator
    v3/                  maximum-likelihood grey-box estimator
    v4/                  multiplicative-coupling estimator
    single_pole/         single-lag baseline variant of V1
    mpc_export/          exports recovered parameters for downstream MPC use
  experiments/           ~35 one-off analysis/validation scripts, grouped by what they investigate
    v1_semisynth/        why V1 works on semi-synthetic but not real data (§10)
    beta_eta_coupling/   the beta-variance-under-stimulation finding (§10)
    v1_logparam/         V1's log-parameterised variant, tuning experiments
    sampling_rate/       sample-rate sweep experiments
    single_pole/         single-pole baseline checks
    real_data_checks/    real-recording-specific reconstruction/convergence checks
    instrument_update_rate/  does the RIV instrument need to update online? (§11)
    excitation_design/   input/excitation protocol checks
    external_ar_model/   derives a cached AR disturbance model from an external dataset
    v4/                  V4-specific validation experiments
  tests/                 pytest suite
```

Every script is standalone and directly runnable — there is no package/CLI
wrapper. For example:

```bash
cd scripts
python estimators/v2/riv_v2_greybox.py    # grey-box output-error estimator + self-check
python estimators/v3/riv_v3_pem.py        # maximum-likelihood grey-box estimator
python estimators/v1/riv_dual_estimation.py  # original V1 KF+RIV estimator (for comparison)
```

Experiment scripts that analyze real recordings (e.g.
`experiments/real_data_checks/riv_real_recon_multisine_sweep_check.py`,
`experiments/real_data_checks/riv_raw_real_replay_sweep_check.py`) expect a
`real_recordings/` directory with the same `inputs,outputs` CSV format the
estimators use — see **Data** below.

## Testing

```bash
pytest scripts/
```

## Project structure

```
scripts/
  estimators/       V1-V4 estimator implementations, by version
  experiments/      one-off analysis, validation, and tuning scripts, by topic
  tests/            pytest suite
docs/
  METHODS_AND_RESULTS.md   synthesized methodology, math verification, benchmarks
  figures/                 a few representative diagnostic plots
results/            generated plots from running the scripts (gitignored; regenerate locally)
```

## Data

This repository does not include real recordings — the original experiments
used real DBS-adjacent stimulation/suppression recordings that are not
publishable for privacy reasons. Scripts that reference real data expect a
`real_recordings/` directory with per-recording CSVs (`inputs`, `outputs`
columns); supply your own in that format to use them. Synthetic and
semi-synthetic experiments (the majority of `scripts/experiments/`) run
standalone with no external data.

## Limitations

- $(\tau_1, \tau_2)$ are only weakly identifiable from the available
  excitation — this is a data/excitation limitation (confirmed by a Fisher
  Information analysis), not something any estimator here resolves. Report a
  single effective $\tau$ with a confidence interval, not two
  separately-trusted lags.
- The additive disturbance model ($y = \beta - \eta$) shared by all four
  estimator versions may itself be the wrong functional form — see §10 of
  the methods doc for the evidence and the open question it leaves.

## References

The recovered plant parameters are intended for use downstream in a
model-predictive controller for closed-loop deep brain stimulation:

S. Steffen, M. Cannon, H. Tan, and J. Debarros, "Model predictive control
for closed-loop deep brain stimulation," in *2024 IEEE 63rd Conference on
Decision and Control (CDC)*, 2025, pp. 4034–4039, doi:
[10.1109/CDC56724.2024.10885888](https://doi.org/10.1109/CDC56724.2024.10885888).

## License

MIT — see [LICENSE](LICENSE).
