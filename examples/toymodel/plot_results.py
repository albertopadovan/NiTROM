"""Plot the cached average step-response error for all five ROMs."""

import json
from pathlib import Path

import numpy as np
from settings import LABELS, arguments


def main(argv=None):
    _, directory, args = arguments(__doc__, argv, plotting=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (
        plt.style.context(Path(__file__).with_name("plot.mplstyle")),
        plt.rc_context(
            {"text.usetex": args.tex, "text.latex.preamble": r"\usepackage{amsmath}"}
        ),
    ):
        metrics = json.loads((directory / "results" / "metrics.json").read_text())
        with np.load(directory / "results" / "predictions.npz") as results:
            fig, ax = plt.subplots(figsize=(6, 4))
            for kind, label in LABELS.items():
                failed = metrics[kind]["diverged"]
                note = f" [{failed} diverged]" if failed else ""
                ax.semilogy(
                    results["time"], results[f"error_{kind}"], label=label + note
                )
            ax.set(
                xlabel=r"Time $t$",
                ylabel=r"Average error $e(t)$",
                title="Held-out step responses",
                xlim=(0, results["time"][-1]),
            )
            ax.legend(fontsize=9)
            fig.tight_layout()
            path = directory / "figures" / "rom_step_error.pdf"
            fig.savefig(path)
            plt.close(fig)
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
