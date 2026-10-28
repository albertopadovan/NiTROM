# Complex Ginzburg–Landau model reduction

This example compares **balanced OpInf**, **POD OpInf**, and **NiTROM refinements
of each** for the stable complex Ginzburg–Landau equation. It follows section 4
of Padovan, Vollmer and Bodony, *Data-Driven Model Reduction via Non-intrusive
Optimization of Projection Operators and Reduced-Order Dynamics*, SIADS 23
(2024), [DOI: 10.1137/24M1628414](https://doi.org/10.1137/24M1628414).
The settings define this repository's experiment rather than reproducing the
paper's dataset or its TrOOP comparison.

Install from the repository root with `python -m pip install -e '.[plot]'`, then
run from this directory:

```bash
python generate_data.py
python train_models.py
python plot_results.py
```

Append `--smoke` to every command for a short run with every model and figure.
The scripts are single-process; see the [MPI companion](../cgl_mpi/README.md)
for distributed NiTROM training using the same data and plotting.

`cgl.py` defines the full-order solver, actuator, and sensor. The full profile
uses 256 grid points on `[-50, 50]`, real state `[Re(q); Im(q)]`, and CN/AB2
integration with `dt=0.01`. Snapshots are spaced by `0.2`. Eight training impulses
use signed amplitudes `{-1, 0.01, 0.1, 1}` on the real and imaginary actuator
columns. Impulse data extend to `t=500`; NiTROM uses one full trajectory window
through `t=200`. Testing includes unseen impulses, 50 seeded ensemble impulses,
and six sinusoidal inputs through `t=300`. Edit these controls in `settings.py`.

`train_models.py` separates basis and OpInf initialization from NiTROM setup and
training. All states and derivatives are stored in physical units. POD divides
trajectories by absolute impulse amplitude. Balancing uses physical trajectories,
ten checkpoints spaced by five samples, and families grouped by signed parent
amplitude. There is no historical amplitude divide/multiply roundtrip, so
rounding-sensitive balanced modes and fitted results can differ from older runs.

OpInf trims two derivative samples from each endpoint. Its unaveraged derivative
residuals are weighted by inverse mean output energy. Cubic penalties are selected
from `10**arange(-2, 9)` using training rollouts. NiTROM averages energy-normalized
output errors over trajectories and samples, using three RK4 substeps, the
discrete adjoint, and 300 optimizer iterations. Trial and test bases use the
Grassmann and Stiefel manifolds. Orthonormalizing the test basis also transforms
the operators to preserve the initialized ROM.

Training automatically continues each NiTROM refinement from its latest
checkpoint, appending loss and gradient histories. `--fresh` restarts from
OpInf. This restores parameters, not optimizer state. Only checkpoint dimensions
are checked; use `--fresh` after changing the physical problem or training data.

Outputs are saved under `outputs/full/` or `outputs/smoke/`:

- `data/`: physical training arrays in `training.npz`, test responses, and settings.
- `models/`: one current NPZ checkpoint per method, including bases, dense operators,
  selected regularization where applicable, and loss/gradient histories.
- `results/`: cached predictions, error curves, and energies.
- `figures/`: modes, impulse responses, sinusoidal responses, ensemble error,
  and training energies.

The public `nitrom.solve_ivp` evaluates ROMs with RK4 and five integration steps
per saved-sample interval, including during regularization selection. The
polynomial model returns a zero RHS when the latent-state norm reaches `1e6`;
integration continues and errors use the returned trajectories directly.
POD OpInf can perform poorly; the comparison retains that result.

The smoke profile uses 64 grid points, training data through `t=40`, NiTROM and
forced responses through `t=20`, two ensemble trajectories, and two optimizer
iterations. Full optimization is excluded from default pytest. Plotting reads
cached results and works without LaTeX; `--tex` enables it. Use
`--output-dir /path/to/run` consistently across stages for another run.
Regenerate data from the old per-trajectory layout.

Checkpoints use the public `nitrom.save_checkpoint` / `nitrom.load_checkpoint`
API. Old `.pkl` checkpoints are not resumed; rerun training to create NPZ
checkpoints. Plot styling lives in this example's `plot.mplstyle` and is scoped
to its plotting call.
