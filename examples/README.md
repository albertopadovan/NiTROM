# Examples

Start with [toymodel](toymodel/README.md), a three-state forced ODE and five ROMs.
Then use [cgl](cgl/README.md) for a spatially distributed system, empirical
balancing, cubic dynamics, and generalization to unseen inputs.
Use [lorenz96](lorenz96/README.md) for chaotic dynamics, ATR constraints,
and an independent POD energy analysis before choosing a latent dimension.

Install from the repository root:

```bash
python -m pip install -e '.[plot]'
```

Each single-process example has three stages, run from its directory:

```bash
python generate_data.py
python train_models.py
python plot_results.py
```

Lorenz-96 additionally provides `python pod_energy.py` and requires an explicit
`--r <dimension>` for training. Its POD energy analysis is optional and does not
control training. Lorenz-96 checkpoints also validate the training mean.

Append `--smoke` to all three commands for a small installation check. Outputs
live in `outputs/full/` or `outputs/smoke/`. Use `--output-dir /path/to/run` on
all stages to select another directory. Scripts also work when launched by
absolute path from another working directory.

Training automatically continues each NiTROM fit from its latest
`models/<method>.npz`, using the public `nitrom.checkpointing` API and appending
loss and gradient histories. The optimizer
starts anew. `--fresh` ignores those parameters and uses the documented
initializations. Baseline fits are recomputed on each run. Checkpoints check
the reduced and full dimensions, plus the training mean for Lorenz-96.
After changing the data or physical problem, use `--fresh`.
There is no `--resume` flag.

Training data are plain arrays in `data/training.npz`; constant input functions
are reconstructed from amplitudes. Regenerate data created by the older
per-trajectory file layout. Training caches predictions and metrics, so plotting
requires no new solves. Figures work without LaTeX; `--tex` enables it.
Each example owns its `plot.mplstyle`; styles are scoped to its plotting call.
Old `.pkl` checkpoints are left untouched and are not resumed. Rerun training
to create the new NPZ checkpoints.

For distributed training, use the separate [toy MPI companion](toymodel_mpi/README.md)
or [CGL MPI companion](cgl_mpi/README.md). They reuse the single-process data,
model construction, output directory, and plotting. The companions show trajectory
sharding, collective training, and rank-zero saving explicitly.

Smoke runs include every model and figure, but do not establish the accuracy of
full optimization runs. Run their checks with `pytest -q tests/examples`.
Generated outputs are ignored by Git.
