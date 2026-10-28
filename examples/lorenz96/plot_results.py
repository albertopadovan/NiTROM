"""Plot cached prediction, attractor-statistic, and input-aware ATR diagnostics."""

import json
from pathlib import Path

import numpy as np
from settings import LABELS, arguments


def main(argv=None):
    _, directory, args = arguments(__doc__, argv, plotting=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads((directory / "results" / "metrics.json").read_text())
    with np.load(directory / "results" / "predictions.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    colors = {kind: f"C{i}" for i, kind in enumerate(LABELS)}

    def save(fig, name):
        fig.tight_layout()
        path = directory / "figures" / name
        fig.savefig(path)
        plt.close(fig)
        print(f"Figure: {path}")

    with (
        plt.style.context(Path(__file__).with_name("plot.mplstyle")),
        plt.rc_context({"text.usetex": args.tex}),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, indices, title in (
            (axes[0], data["constant_indices"], "Unseen constant inputs"),
            (axes[1], np.array([3]), "Unseen sinusoidal input"),
        ):
            for kind, label in LABELS.items():
                failures = sum(metrics[kind]["short_failed"][i] for i in indices)
                note = f" [{failures} failed]" if failures else ""
                # A failed curve stops; it is never silently scored as a valid prediction.
                error = data[f"error_{kind}"][indices].mean(axis=0)
                projection = data[f"projection_error_{kind}"][indices].mean(axis=0)
                ax.semilogy(
                    data["time"],
                    np.maximum(error, 1e-15),
                    color=colors[kind],
                    label=label + note,
                )
                ax.semilogy(
                    data["time"],
                    np.maximum(projection, 1e-15),
                    "--",
                    color=colors[kind],
                    alpha=0.6,
                )
            ax.set(xlabel="Time", ylabel="Relative fluctuation error", title=title)
            ax.legend(fontsize=8)
        fig.suptitle("Rollout error; dashed: projection-only error")
        save(fig, "prediction_error.pdf")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        tail = data["long_time"] >= float(data["statistics_start"])
        for row, case in enumerate((0, 3)):
            axes[row, 0].plot(data["time"], data["truth"][case, 0], "k", label="FOM")
            axes[row, 1].plot(
                data["long_truth"][case, 0, tail],
                data["long_truth"][case, 1, tail],
                "k",
                lw=0.8,
                label="FOM",
            )
            for kind, label in LABELS.items():
                long_note = " [failed]" if metrics[kind]["long_failed"][case] else ""
                axes[row, 0].plot(
                    data["time"],
                    data[f"prediction_{kind}"][case, 0],
                    color=colors[kind],
                    label=label,
                )
                axes[row, 1].plot(
                    data[f"long_prediction_{kind}"][case, 0, tail],
                    data[f"long_prediction_{kind}"][case, 1, tail],
                    color=colors[kind],
                    lw=0.8,
                    alpha=0.7,
                    label=label + long_note,
                )
            axes[row, 0].set(
                xlabel="Time", ylabel=r"$x_1$", title=str(data["labels"][case])
            )
            axes[row, 1].set(
                xlabel=r"$x_1$", ylabel=r"$x_2$", title="Long-time state projection"
            )
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=7)
        axes[1, 1].legend(fontsize=7)
        save(fig, "trajectories.pdf")

        fig, axes = plt.subplots(len(data["constant_indices"]), 4, figsize=(15, 9))
        centers = 0.5 * (data["density_edges"][1:] + data["density_edges"][:-1])
        for row, case in enumerate(data["constant_indices"]):
            for column, (name, coordinate, xlabel, ylabel) in enumerate(
                (
                    ("mean", np.arange(1, data["truth"].shape[1] + 1), "Site", "Mean"),
                    (
                        "variance",
                        np.arange(1, data["truth"].shape[1] + 1),
                        "Site",
                        "Variance",
                    ),
                    ("density", centers, "State value", "Pooled density"),
                    ("autocorrelation", data["lags"], "Lag", "Autocorrelation"),
                )
            ):
                ax = axes[row, column]
                ax.plot(coordinate, data[f"{name}_truth"][row], "k", label="FOM")
                for kind, label in LABELS.items():
                    missing = metrics[kind]["constant_statistics"][row] is None
                    ax.plot(
                        coordinate,
                        data[f"{name}_{kind}"][row],
                        color=colors[kind],
                        label=label + (" [unavailable]" if missing else ""),
                    )
                ax.set(xlabel=xlabel, ylabel=ylabel, title=str(data["labels"][case]))
        axes[0, 0].legend(fontsize=7)
        save(fig, "constant_statistics.pdf")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for row, kind in enumerate(("atr_opinf", "atr_nitrom")):
            bound = metrics[kind]["atr"]
            level = bound["ellipsoid_level"]
            for case, label in enumerate(data["labels"]):
                value = data[f"lyapunov_{kind}"][case]
                axes[row, 0].semilogy(
                    data["long_time"],
                    np.maximum(value / level if level > 0 else value, 1e-30),
                    label=str(label),
                )
                axes[row, 1].plot(
                    data["long_time"], data[f"lyapunov_derivative_{kind}"][case]
                )
            axes[row, 0].axhline(1 if level > 0 else 0, color="k", ls="--")
            axes[row, 0].set(
                xlabel="Time",
                ylabel="V / ellipsoid boundary" if level > 0 else "V",
                title=LABELS[kind],
            )
            axes[row, 1].axhline(0, color="k", ls="--")
            axes[row, 1].set_yscale("symlog", linthresh=1e-3)
            axes[row, 1].set(
                xlabel="Time",
                ylabel=r"$\dot V$",
                title=f"Outside samples: {bound['outside_samples']}; violations: {bound['derivative_violations']}",
            )
        axes[0, 0].legend(fontsize=8)
        save(fig, "atr_diagnostics.pdf")


if __name__ == "__main__":
    main()
