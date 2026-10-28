# Forced three-state ODE

A forced quadratic ODE with three states is reduced to two latent coordinates.
A constant scalar input drives all states; the measured output is their sum.
The coefficients and full-order projection are defined in `model.py`.

Install from the repository root with `python -m pip install -e '.[plot]'`, then
run from this directory:

```bash
python generate_data.py
python train_models.py
python plot_results.py
```

Append `--smoke` to every command for a short run with all five methods:

1. **POD-Galerkin:** project the known full-order operators.
2. **OpInf:** fit projected derivatives with weighted least squares.
3. **GAS-OpInf:** retract the ordinary fit and optimize stable dynamics.
4. **NiTROM:** optimize dynamics and trial/test bases, initialized from Galerkin.
5. **GAS-NiTROM:** optimize stable dynamics and bases, initialized from GAS-OpInf.

The reduced input operator is fixed throughout training. `initialize_models()`
shows the baseline fits, `make_nitrom()` constructs the two refinements, and
`main()` trains and saves them. These scripts are single-process; see the
[distributed-training companion](../toymodel_mpi/README.md) for MPI.

Training automatically continues the latest saved NiTROM and GAS-NiTROM
parameters and appends their histories. `--fresh` restores the listed
initializations. Optimizer state is not restored. Only checkpoint dimensions
are checked; use `--fresh` after changing the physical problem or data.

Settings live in `settings.py`. The default uses the continuous adjoint,
15 integration substeps, 400 iterations per NiTROM fit, and 1000 GAS-OpInf
iterations. Four training inputs have amplitudes `0.01, 0.1, 0.2, 0.248`,
sampled 100 times through `t=10`. POD normalizes by steady-state output energy;
inference uses the same energy normalization and averages over trajectories
and snapshots. Ordinary OpInf uses `reg={2: 2.983647e-7}`. Testing uses
100 seeded amplitudes in `[0, 0.24)`.

The smoke profile uses two optimizer iterations, 21 training samples through
`t=2`, and four test inputs. Full optimization is excluded from default pytest.

Outputs are saved under `outputs/full/` or `outputs/smoke/`:

- `data/`: training arrays and amplitudes in `training.npz`, test references,
  and a settings record.
- `models/`: one current NPZ checkpoint per method, with bases, operators, GAS parameters,
  settings, and loss/gradient histories.
- `results/`: cached predictions, error curves, and JSON metrics.
- `figures/rom_step_error.pdf`: held-out step-response errors.

Failed rollouts remain NaN in predictions, contribute zero predicted output to
the error curves after failure, and are counted in legends. Plotting needs no
LaTeX; `--tex` enables it. Use `--output-dir /path/to/run` consistently across
stages for an independent run. Regenerate data from the old per-trajectory layout.

Checkpoints use the public `nitrom.save_checkpoint` / `nitrom.load_checkpoint`
API and preserve the concrete model type. Old `.pkl` checkpoints are not resumed;
rerun training to create NPZ checkpoints. Plot styling lives in this example's
`plot.mplstyle` and is scoped to its plotting call.
