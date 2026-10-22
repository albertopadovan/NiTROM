# Complex Ginzburg–Landau model reduction

This example compares cubic ROMs of the stable complex Ginzburg–Landau equation:
**balanced OpInf**, **POD OpInf**, and **NiTROM refinements of each**. It follows
section 4 of Padovan, Vollmer and Bodony, *Data-Driven Model Reduction via
Non-intrusive Optimization of Projection Operators and Reduced-Order Dynamics*,
SIADS 23 (2024), [DOI: 10.1137/24M1628414](https://doi.org/10.1137/24M1628414).
The settings preserve this repository's benchmark rather than exactly reproducing
the paper's dataset or its TrOOP comparison.

After installing this checkout with `python -m pip install -e '.[plot]'` from the
repository root, run from this directory:

```bash
python generate_data.py
python train_models.py
python plot_results.py
```

`cgl.py` defines the full-order solver, actuator, and sensor. The full profile uses
256 grid points on `[-50, 50]`, a 512-entry real state `[Re(q); Im(q)]`, and
CN/AB2 integration with `dt=0.01`. Snapshots are spaced by `0.2`. There are eight
training impulses: four signed amplitudes `{-1, 0.01, 0.1, 1}` on the actuator's
real and imaginary columns. Impulse data extend to `t=500`; NiTROM trains through
`t=200`. Testing includes unseen impulses, 50 seeded ensemble impulses, and six
sinusoidal inputs through `t=300`. These parameters live in `settings.py`.

`train_models.py` exposes basis construction, regularization selection, and the
NiTROM setup. It uses public NiTROM APIs for POD, balancing, polynomial dynamics,
projection, OpInf, and training. Small array-backed views in `_utils.py` provide
in-memory data to those APIs without requiring a derivative cache for NiTROM.
DOP853 evaluates ROM predictions with an explicit blow-up event.

Data and fitting conventions:

- All saved states and derivatives are in **physical units**. Signed impulse
  amplitudes are stored separately. Gaussian actuator profiles have unit norm.
- POD divides physical trajectories by their absolute impulse amplitudes, preserving
  the covariance of the former amplitude-normalized data. Balancing uses physical
  trajectories, ten checkpoints spaced by five samples, and families determined
  by each parent's signed amplitude. Its input retains the old divide/multiply
  roundtrip: phase-paired singular values make the rank-5 cutoff sensitive to
  tiny rounding changes.
- OpInf omits two derivative samples at each endpoint. Its unaveraged derivative
  residuals are weighted by inverse mean output energy. Cubic tensor penalties
  are selected from `10**arange(-2, 9)` using training trajectories only.
- NiTROM averages energy-normalized output errors over trajectories and samples.
  Its default uses one full window per trajectory, three RK4 substeps, the discrete
  adjoint, and 300 optimizer iterations. Trial and test bases are optimized on the
  Grassmann and Stiefel manifolds; the test-basis QR also transforms the operators.

Outputs in `outputs/full/`:

| Directory | Contents |
| --- | --- |
| `data/` | Physical trajectories, finite-difference derivatives, amplitudes, reference responses, and settings |
| `models/` | Four checkpoints with bases, dense operators, selected regularization, histories, and run metadata |
| `results/` | Cached impulse/forced/ensemble predictions, error curves, energies, and failure metrics |
| `figures/` | Initial modes, impulse responses, sinusoidal responses, ensemble error, and training-energy plots |

Matching NiTROM checkpoints continue automatically. `--fresh` restarts both
refinements from their OpInf initializations. This is a parameter warm start;
optimizer state is not restored. Data/settings mismatches are rejected with an
instruction to use `--fresh`. Use `--output-dir /path/to/run` on all stages for an
independent run.

Append `--smoke` to every command for 64 grid points, data through `t=40`, NiTROM
and forced responses through `t=20`, two ensemble trajectories, and two optimizer
iterations. Every model and figure remains included. During validation, generation
took **1.1 s**, training/evaluation **3.2 s**, and first-time plotting **13 s**,
including process startup and font-cache construction. Full training is much more
expensive and depends on optimizer convergence; it is excluded from default pytest.

Plotting reads cached results and works without LaTeX; `--tex` enables publication
styling. Failed rollouts remain NaN, with failure counts in metrics and legends.
For the ensemble error curve, failures contribute zero predicted output after
failure rather than being dropped. POD OpInf can perform poorly with these
settings; the workflow reports that result without retuning the baseline.

Full-data migration checks used all eight 256-point training trajectories. Relative
prediction differences against the compact reference fit were below `1e-10` for
both bases; the representative NiTROM loss and gradients also agreed. This checks
the fitting/data migration without running the two full 300-iteration refinements.
