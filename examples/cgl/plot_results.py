"""Render all CGL figures from cached results; no training or time integration."""

import json

import numpy as np
from _utils import plot_style
from settings import LABELS, arguments


def main(argv=None):
    settings, directory, args = arguments(__doc__, argv, plotting=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_style(args.tex)
    with np.load(directory / "results" / "predictions.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    metrics = json.loads((directory / "results" / "metrics.json").read_text())
    figures = directory / "figures"
    time, tf = data["time"], data["time_forced"]

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)

    # Training and held-out impulses, normalized only for display by IC amplitude.
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for j, ax in enumerate(axes.flat):
        amplitude = data["impulse_amplitudes"][j]
        truth = data["impulse_truth"][j, 0] / amplitude
        ax.plot(time, truth, "k", label="FOM")
        for kind, label in LABELS.items():
            error = metrics[kind]["impulse_relative_errors"][j]
            note = "diverged" if error is None else f"error {error:.2e}"
            ax.plot(
                time,
                data[f"impulse_{kind}"][j, 0] / amplitude,
                label=f"{label} ({note})",
            )
        ax.set(
            title=str(data["impulse_labels"][j]), xlabel="$t$", ylabel=r"Re $y/\beta$"
        )
        bound = max(float(np.max(np.abs(truth))), 1e-15)
        ax.set_ylim(-1.5 * bound, 1.5 * bound)
        ax.legend(fontsize=7)
    save(fig, f"impulse_responses_r{settings.r}.png")

    # Sinusoidal forcing: retain all amplitude/frequency combinations.
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for j, ax in enumerate(axes.flat):
        truth = data["force_truth"][j, 0]
        ax.plot(tf, truth, "k", label="FOM")
        for kind, label in LABELS.items():
            prediction = data[f"forced_{kind}"][j, 0]
            note = " [diverged]" if not np.isfinite(prediction).all() else ""
            ax.plot(tf, prediction, label=label + note)
        ax.set(
            title=f"Amplitude {data['force_amplitudes'][j]:g}, frequency {data['force_frequencies'][j]:g} omega",
            xlabel="$t$",
            ylabel=r"Re $y$",
        )
        bound = max(float(np.max(np.abs(truth))), 1e-15)
        ax.set_ylim(-1.5 * bound, 1.5 * bound)
        ax.legend(fontsize=7)
    save(fig, f"sinusoidal_responses_r{settings.r}.png")

    # Ensemble error: failures remain counted, with zero-output error thereafter.
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, label in LABELS.items():
        failed = metrics[kind]["ensemble_diverged"]
        note = f" [{failed} diverged]" if failed else ""
        ax.semilogy(time, data[f"error_{kind}"], label=label + note)
    ax.set(xlabel="$t$", ylabel="$e(t)$", title="Average held-out impulse error")
    ax.legend(fontsize=8)
    save(fig, f"avg_testing_error_r{settings.r}.png")

    # Initial balanced trial/test modes and POD modes.
    bases = [
        (r"Balanced $\Psi$", data["Psi_balanced_opinf"]),
        (r"Balanced $\Phi$", data["Phi_balanced_opinf"]),
        ("POD", data["Phi_pod_opinf"]),
    ]
    fig, axes = plt.subplots(
        settings.r, 3, figsize=(11, 1.6 * settings.r), squeeze=False
    )
    nx = len(data["x"])
    for column, (label, modes) in enumerate(bases):
        for j in range(settings.r):
            ax = axes[j, column]
            mode = modes[:, j] / np.max(np.abs(modes[:, j]))
            ax.plot(data["x"], mode[:nx], label="Re")
            ax.plot(data["x"], mode[nx:], label="Im")
            ax.plot(
                data["x"],
                np.hypot(mode[:nx], mode[nx:]),
                "k",
                lw=0.7,
                label="magnitude",
            )
            for branch in (data["xbar"], -data["xbar"]):
                ax.axvline(branch, color="gray", ls="--", lw=0.6)
            ax.set(xlim=(-30, 30), yticks=[])
            if j == 0:
                ax.set_title(label)
            if column == 0:
                ax.set_ylabel(f"Mode {j + 1}")
            if j == settings.r - 1:
                ax.set_xlabel("$x$")
    axes[0, 0].legend(fontsize=6)
    save(fig, f"modes_r{settings.r}.png")

    # State/output energy and the normalization used in the NiTROM objective.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    energies = [data["state_energy"], data["output_energy"]]
    amplitudes = data["training_amplitudes"]
    unique = np.unique(amplitudes)
    colors = dict(
        zip(unique, plt.cm.viridis(np.linspace(0, 0.85, len(unique))), strict=True)
    )
    for row, (energy, label) in enumerate(
        zip(energies, ("State", "Output"), strict=True)
    ):
        seen = set()
        for j, curve in enumerate(energy):
            amplitude = amplitudes[j]
            style = "--" if amplitude < 0 else "-"
            label_amplitude = (
                rf"$\beta={amplitude:g}$" if amplitude not in seen else None
            )
            seen.add(amplitude)
            axes[row, 0].semilogy(
                data["training_time"],
                curve,
                ls=style,
                color=colors[amplitude],
                label=label_amplitude,
            )
            axes[row, 1].semilogy(
                data["training_time"],
                curve / curve.mean(),
                ls=style,
                color=colors[amplitude],
            )
        axes[row, 0].set_title(f"{label} energy")
        axes[row, 1].set_title(f"{label} energy / time average")
        axes[row, 0].set_ylabel("Energy")
        for ax in axes[row]:
            ax.grid(alpha=0.25)
            maximum = max(float(np.max(line.get_ydata())) for line in ax.lines)
            ax.set_ylim(maximum * 1e-8, maximum * 3)
    for ax in axes[-1]:
        ax.set_xlabel("$t$")
    axes[0, 0].legend(fontsize=8, ncol=2)
    save(fig, "training_energy.png")
    print(f"Five figures: {figures}")


if __name__ == "__main__":
    main()
