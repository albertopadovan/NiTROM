# MPI training companion: CGL

Read the [single-process example](../cgl/README.md) first. This companion
reuses its data generation, model definitions, initialization, checkpoints, and
plotting. Only NiTROM training is distributed; baseline fits run on rank zero.

Install an MPI runtime and launcher, then install this checkout from the
repository root:

```bash
python -m pip install -e '.[mpi,plot]'
```

From the repository root, run:

```bash
python examples/cgl/generate_data.py --smoke
mpiexec -n 2 python examples/cgl_mpi/train_models.py --smoke
python examples/cgl/plot_results.py --smoke
```

Remove `--smoke` from all stages for the full experiment. A serial training run
is not required first. The default output directory is the single-process
example's `outputs/smoke/` or `outputs/full/`. To use another run directory, pass
`--output-dir /path/to/run` to all three commands.

The training script exposes the MPI steps directly:

1. Rank zero builds the baselines with `MPI.COMM_SELF`, then broadcasts their
   parameters and any saved NiTROM checkpoints.
2. Each rank takes trajectories `rank::size`, retaining the normalization
   computed on the full dataset. These examples load the small dataset on each
   rank before slicing it for clarity.
3. `data.pool.comm` tells NiTROM which communicator to use. Its optimizer sums
   local costs and gradients internally, so every rank follows the same updates.
4. Rank zero saves the trained models and evaluates test responses.

Use at most 8 ranks so every rank has a trajectory. Changing the number of
ranks does not change the global loss normalization. More ranks may be slower
for the smoke profile because communication dominates these short fits.

Rerunning automatically continues the latest NiTROM parameters, including
checkpoints from the single-process example. `--fresh` restarts from the
baseline initializations. Optimizer state is not restored. Only checkpoint
dimensions are checked; use `--fresh` after changing the physical problem or
data. Run one training job at a time per output directory.

Checkpoints use the public NPZ checkpoint format. Every rank loads its own
components from the shared output directory; only rank zero writes archives.
Old `.pkl` checkpoints are not resumed; rerun training to create NPZ checkpoints.
