# Lorenz-96: ATR constraints and model reduction

Forty periodically indexed states obey

\[
\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+8+u(t).
\]

The scalar input acts uniformly on all sites. The quadratic term conserves
energy, while linear damping and bounded forcing give a trapping region.
See [Kerin and Engler](https://arxiv.org/abs/2005.07767) for Lorenz-96 and
[Schlegel and Noack](https://arxiv.org/abs/1310.0053) for trapping-region analysis.

Install from the repository root with `python -m pip install -e '.[plot]'`.
Run from this directory, or launch scripts by absolute path from elsewhere:

```bash
python generate_data.py
python pod_energy.py --energy 0.9 0.95 0.99
python train_models.py --r 10
python plot_results.py
```

The energy targets and `--r 10` above are illustrative choices, not accuracy
recommendations. **Training requires `--r`; there is no default dimension.**
`pod_energy.py` is independent of training: without `--energy` it still saves
the singular spectrum and plots cumulative centered-state energy. It never
selects or changes the training dimension. Training computes its own POD basis
and does not read the energy-analysis outputs.

Append `--smoke` to each command for a small workflow check. Use
`--output-dir /path/to/run` consistently across stages for a separate run.
Figures work without LaTeX; `--tex` enables it for either plotting script.

## Data and models

`settings.py` contains the physical and numerical settings. The full profile
generates eight trajectories: two independent initial states near `8 * ones(40)`
for each constant input `-2, -1, 0, 2`. Every trajectory starts in a transient
regime and evolves toward its long-time dynamics; none receives a spin-up.
They run through `t=30`, with samples every `0.02` and RK4 step `0.002`.
Derivatives are evaluated directly from the equations.

The mean over all complete training trajectories is saved and subtracted before
POD and fitting. Ordinary, unweighted POD is used, so the reported energy is
centered snapshot energy, not per-trajectory normalized energy. Physical
predictions restore the saved mean. POD energy alone does not establish that
a dimension is sufficient for autonomous dynamics; fewer than three autonomous
states cannot reproduce a chaotic attractor.

The four methods share the same initial rank-r POD trial/test bases:

1. **OpInf:** constant, linear, quadratic, and scalar-input operators fitted by
   regularized least squares, with quadratic regularization `1e-6`.
2. **ATR-OpInf:** retract the ordinary fit, initialize the shift from the latent
   training mean, and optimize all ATR parameters and the input operator.
3. **NiTROM:** refine OpInf, including its dynamics, input operator, and bases.
4. **ATR-NiTROM:** refine ATR-OpInf with the same trajectory objective.

POD and OpInf use the complete training histories. NiTROM fits only the initial
prefix through `t=1`, avoiding a long chaotic trajectory-matching objective.
Both objectives average squared errors over trajectories and samples, normalized
by each trajectory's centered-state energy; OpInf uses its full history and
NiTROM its prefix to compute that normalization. Defaults are 200 iterations per
optimized fit, L-BFGS, discrete adjoints, and four RK4 substeps per measurement
interval. NiTROM uses Grassmann trial and Stiefel test bases. Its optimized bases
need not retain the initial POD energy fraction.

Training automatically continues saved NiTROM and ATR-NiTROM parameters and
appends histories. Baselines are recomputed. `--fresh` restores the listed
initializations; optimizer state is not restored. Requested dimension and the
centering offset are checked before overwriting fits. Use `--fresh` after other
changes to the data or physical problem, even if these checks still pass.

## Evaluation and bounds

Independent test initial conditions use unseen constant inputs `-1.5, 0.5, 1.5`
and `u(t) = 2 sin(2*pi*t/5)`. Short predictions span two time units; long rollouts
span 100, with the first 20 excluded from statistics. Short errors are RMS
fluctuation-energy-normalized state errors; projection-only errors use each
method's own bases. Constant-input statistics compare per-site means and
variances, pooled site marginals, and site-averaged temporal autocorrelations.
The sinusoid is evaluated as a driven response, not a stationary attractor.

ATR bounds account for **every** scalar input with `abs(u) <= 2`, using
`max(norm(Bhat - 2*B), norm(Bhat + 2*B))` in place of `norm(Bhat)`.
For `Qtil = inv(Q) @ inv(Q).T`, `Rtil = inv(R) @ inv(R).T`, and `w = z - m`,

\[
V=\tfrac12 w^\top\widetilde Qw,\qquad
\rho=\frac{\max_{|u|\le2}\|\widehat B+Bu\|}
 {\lambda_{\min}(\widetilde R)\lambda_{\min}(\widetilde Q)}.
\]

The certified ellipsoid is `V <= 0.5*lambda_max(Qtil)*rho**2`. Its containing
ball has radius `rho*sqrt(cond(Qtil))`. The smaller radius rho only bounds the
region where V can increase; it need not contain the trapping ellipsoid.
Diagnostics report these quantities, sampled ellipsoid occupancy, and sign
checks on `Vdot` outside the rho-ball. Positive `Vdot` inside that ball is allowed.
Initial states can lie outside the certified region. Sampled checks support the
implementation; the bound follows from the ATR structure. These are continuous
ODE guarantees, so a fixed numerical step can still fail for a stiff model.

Failed rollouts remain NaN after their first failed sample and are counted.
Their error curves stop, and their constant-input statistics are unavailable.
Histograms use common reference-derived bins; any predicted mass outside those
bins is reported separately. Zero-variance trajectories have undefined
autocorrelations. A bounded model can still have the wrong attractor or dynamics.

## Outputs and checks

Outputs live in `outputs/full/` or `outputs/smoke/`:

- `data/`: physical training/test arrays, training mean, settings, and optional
  standalone POD spectrum.
- `models/`: four NPZ checkpoints, including the mean, model parameters, bases,
  run counts, loss histories, and initial POD retained energy.
- `results/`: cached physical predictions, projection errors, statistics,
  Lyapunov diagnostics, and JSON metrics; optional POD energy reports.
- `figures/`: prediction errors, representative trajectories and state
  projections, constant-input statistics, ATR diagnostics, and optional POD energy.

The smoke profile retains all eight training trajectories and 40 physical states,
uses `t=2` training histories, a `t=0.2` NiTROM prefix, `t=0.4` predictions, and
`t=4` long rollouts with a one-unit statistics exclusion. It performs two optimizer
iterations and still requires an explicit `--r`. It checks the workflow and
does not establish converged attractor statistics or full-profile accuracy.

Run `pytest -q tests/examples/test_lorenz96.py` for the physics, diagnostics,
gradient, standalone analysis, and smoke workflow checks. This example uses
public package APIs and is single-process; it has no MPI companion.
