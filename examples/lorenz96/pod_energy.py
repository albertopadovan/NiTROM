"""Inspect centered training-state POD energy, independently of model training."""

from pathlib import Path

import numpy as np
from _utils import load_training, prepare, write_json
from settings import arguments


def energy_spectrum(X, mean):
    centered = X - mean[None, :, None]
    if not np.isfinite(centered).all():
        raise ValueError("POD requires finite, nonzero centered training energy.")
    snapshots = centered.transpose(1, 0, 2).reshape(X.shape[1], -1)
    singular = np.linalg.svd(snapshots, compute_uv=False)
    energy = singular**2
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        raise ValueError("POD requires finite, nonzero centered training energy.")
    cumulative = np.cumsum(energy) / energy.sum()
    cumulative[-1] = 1.0
    return singular, cumulative


def rank_for_energy(cumulative, target):
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Energy fractions must be finite and in (0, 1].")
    return int(np.searchsorted(cumulative, target, side="left") + 1)


def main(argv=None):
    _, directory, args = arguments(__doc__, argv, analysis=True)
    X, _, _, _, mean = load_training(directory)
    singular, cumulative = energy_spectrum(X, mean)
    prepare(directory)
    np.savez(
        directory / "data" / "pod_energy.npz",
        singular_values=singular,
        cumulative_energy=cumulative,
        mean=mean,
    )
    targets = [
        dict(
            target=e,
            r=rank_for_energy(cumulative, e),
            retained_energy=float(cumulative[rank_for_energy(cumulative, e) - 1]),
        )
        for e in args.energy
    ]
    write_json(directory / "results" / "pod_energy.json", dict(targets=targets))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (
        plt.style.context(Path(__file__).with_name("plot.mplstyle")),
        plt.rc_context({"text.usetex": args.tex}),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        ranks = np.arange(1, len(singular) + 1)
        axes[0].semilogy(ranks, singular, "o-", markersize=3)
        axes[0].set(xlabel="Rank", ylabel="Singular value")
        axes[1].plot(ranks, cumulative, "o-", markersize=3)
        for result in targets:
            axes[1].plot(
                result["r"],
                result["retained_energy"],
                "x",
                label=f"{result['target']:g}: r={result['r']}",
            )
            print(
                f"Energy {result['target']:g}: r={result['r']} ({result['retained_energy']:.6f} retained)"
            )
        axes[1].set(
            xlabel="Rank", ylabel="Cumulative centered-state energy", ylim=(0, 1.02)
        )
        if targets:
            axes[1].legend()
        fig.tight_layout()
        path = directory / "figures" / "pod_energy.pdf"
        fig.savefig(path)
        plt.close(fig)
    print(f"POD spectrum: {path}; choose --r explicitly when training.")


if __name__ == "__main__":
    main()
