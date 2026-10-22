# Forced three-state ODE

This is the starting example. A quadratic ODE with three states is reduced to two
latent coordinates. A constant scalar input drives all states, and the measured
output is their sum. The coefficients are defined once in `model.py`.

From this directory, after installing the checkout with `python -m pip install -e
'.[plot]'` from the repository root:

```bash
python generate_data.py
python train_models.py
python plot_results.py
```

`train_models.py` shows five comparisons in order:

1. **POD-Galerkin:** project the known full-order operators.
2. **OpInf:** fit projected derivatives with `solve_opinf(..., solver="lstsq")`.
3. **GAS-OpInf:** retract the ordinary fit and iteratively optimize stable dynamics.
4. **NiTROM:** jointly optimize dynamics and trial/test bases, starting from Galerkin.
5. **GAS-NiTROM:** optimize stable dynamics and bases, starting from GAS-OpInf.

The reduced input operator is fixed throughout training. Each method owns its
parameters and projection. Saved NiTROM parameters take precedence over the listed
initializations on subsequent runs; `--fresh` restores those initializations.
The default uses the continuous adjoint, 15 integration substeps, 400 iterations
for each NiTROM fit, and 1000 for GAS-OpInf. Edit `settings.py` to change these.

Training uses four step inputs with amplitudes `0.01, 0.1, 0.2, 0.248`, sampled
100 times through `t=10`. POD normalizes each trajectory by its steady-state
output energy. The inference losses use the same energy weights and average
over trajectories and snapshots. Ordinary OpInf penalizes the quadratic tensor
with `reg={2: 2.983647e-7}`. Testing uses 100 seeded amplitudes in `[0, 0.24)`.

Outputs in `outputs/full/`:

| Directory | Contents |
| --- | --- |
| `data/` | Physical training states/derivatives, weights, input callables, and testing reference solutions |
| `models/` | One self-contained pickle per method, including bases, physical operators, constrained parameters, settings, and histories |
| `results/` | Cached predictions, normalized error curves, and JSON metrics |
| `figures/` | `rom_step_error.pdf`, comparing held-out step-response errors |

Failed rollouts remain NaN in cached predictions. Error curves use zero output
after failure and report the number of failed trajectories in the legend.

For a quick check, append `--smoke` to every command. It keeps all five methods,
uses two optimizer iterations, 21 training samples through `t=2`, and four test
inputs. In validation on the development machine, generation took **0.3 s**,
training/evaluation **0.9 s**, and first-time plotting **11 s**, including startup
and font-cache construction. Timings vary by machine and cache state.
The full profile was also run: data generation took **0.6 s** and all five fits
plus evaluation took **452 s** (about 7.5 minutes), with BLAS limited to one thread.

Use `--output-dir /path/to/run` consistently across stages for a separate run.
Plotting needs no LaTeX; `--tex` enables publication styling. Full optimization is
not part of default pytest; the normal commands above run it explicitly.
