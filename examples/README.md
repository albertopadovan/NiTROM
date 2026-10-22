# Examples

Start with **[toymodel](toymodel/README.md)**: a three-state forced ODE and five
ROMs, with their construction and training visible in one script. Then use
**[cgl](cgl/README.md)** for a spatially distributed system, empirical balancing,
cubic dynamics, and generalization to unseen impulses and sinusoidal inputs.

Install this checkout from the repository root:

```bash
python -m pip install -e '.[plot]'
```

Each example has the same three commands, run from its directory:

```bash
python generate_data.py
python train_models.py
python plot_results.py
```

Append `--smoke` to **all three commands** for a small installation check. Outputs
are separated into `outputs/full/` and `outputs/smoke/`; `--output-dir /path/to/run`
overrides the run directory. Scripts can also be launched by absolute path from
another working directory. They import the installed package; an editable install
ensures that package is this checkout.

`train_models.py` automatically continues matching NiTROM checkpoints. Use
`--fresh` to start those fits from their documented initializations. Continuation
restores parameters and appends histories; the optimizer starts anew. Data or
settings mismatches require `--fresh`. Ordinary OpInf and GAS-OpInf are fit again.

Training saves checkpoints and predictions; plotting only reads saved results.
Figures work without LaTeX. Add `--tex` to `plot_results.py` for publication styling.
Generated outputs are ignored by Git. No downloaded dataset is required.

The smoke profiles exercise the whole workflow, including every model and figure;
they do not establish the accuracy of the full scientific comparison. Run their
integration tests with `pytest -q tests/examples`. Full optimization runs remain
outside default pytest.
