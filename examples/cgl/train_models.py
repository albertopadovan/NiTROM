"""Compare POD/balanced cubic OpInf and NiTROM refinements using public APIs."""

import time as clock

import numpy as np
from _utils import (
    basis_pool,
    checkpoint,
    checkpoint_windows,
    data_signature,
    evaluate,
    global_loss,
    load_training,
    orthonormalize_psi,
    prepare,
    read_data_settings,
    restore,
    resume_checkpoint,
    save_checkpoint,
    selection_cost,
    training_view,
)
from cgl import CGL, OutputMap, input_output
from settings import arguments

from nitrom.backend import set_backend
from nitrom.bases import compute_data_driven_balancing, compute_pod
from nitrom.distributed import mpi_rank_size
from nitrom.latent_space_models import PolynomialModel
from nitrom.optimization import NitromModule, OpInfModule, solve_opinf, train
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry


def main(argv=None):
    settings, directory, args = arguments(__doc__, argv, training=True)
    set_backend("numpy")
    start = clock.perf_counter()
    prepare(directory)
    read_data_settings(directory, settings)
    signature = data_signature(directory)
    previous = {
        kind: resume_checkpoint(directory, kind, settings, signature, args.fresh)
        for kind in ("nitrom_lall", "nitrom_pod")
    }
    X, derivatives, time, amplitudes = load_training(directory)
    fom = CGL(nx=settings.nx)
    _, _, _, C = input_output(fom, settings.seed)
    rank, _ = mpi_rank_size()

    # Bases: balancing uses physical trajectories and families by parent amplitude.
    families = [list(np.where(amplitudes == a)[0]) for a in np.unique(amplitudes)]
    # Preserve the former amplitude-normalized storage roundtrip for balancing.
    # Phase-paired singular values make the odd-rank cutoff sensitive to rounding.
    balancing_states = (X / amplitudes[:, None, None]) * amplitudes[:, None, None]
    Phi, Psi, Sigma, _ = compute_data_driven_balancing(
        basis_pool(balancing_states, time),
        n_checkpoints=settings.balancing_checkpoints,
        checkpoint_stride=settings.balancing_stride,
        families=families,
    )
    # X/|beta| has the same covariance as the former X/beta storage convention.
    U, _, _ = compute_pod(basis_pool(X, time, amplitudes**2), normalize=True)
    bases = {
        "balanced_opinf": (Phi[:, : settings.r], Psi[:, : settings.r]),
        "pod_opinf": (U[:, : settings.r], U[:, : settings.r]),
    }

    # OpInf: trim two endpoint samples and retain the unaveraged derivative fit.
    # Unlike TrainingData's averaged loss, these weights contain no ntraj*nt factor.
    weights = np.mean(np.sum((C @ X) ** 2, axis=1), axis=1)
    data = training_view(X[:, :, 2:-2], time[2:-2], weights, derivatives[:, :, 2:-2])
    models = {}
    for kind, (phi, psi) in bases.items():
        best, best_cost = None, np.inf
        for reg in 10.0 ** np.arange(-2, 9):
            model = PolynomialModel(settings.r, [1, 3], instability_threshold=np.inf)
            projection = LinearProjection([phi.copy(), psi.copy()])
            opinf = OpInfModule(data, model, projection, reg={3: float(reg)})
            solve_opinf(opinf, solver="lstsq")
            cost = selection_cost(model, projection, X, time, C)
            if cost < best_cost:
                best_cost = cost
                best = checkpoint(
                    kind, model, projection, settings, signature, module=opinf
                )
                best.update(selected_reg=float(reg), selection_cost=cost)
        if best is None:
            raise RuntimeError(f"Every regularization candidate diverged for {kind}.")
        best["Sigma"] = Sigma.copy() if kind == "balanced_opinf" else None
        models[kind] = best
        save_checkpoint(directory, best)
        if rank == 0:
            print(
                f"{kind}: reg={best['selected_reg']:g}, selection cost={best_cost:.6g}"
            )

    # NiTROM: independent refinements of the balanced and POD OpInf initializations.
    count = int(np.searchsorted(time, settings.nitrom_end)) + 1
    windows, length = checkpoint_windows(
        X[:, :, :count], settings.nitrom_checkpoints, settings.nitrom_stride
    )
    training_time = time[:length]
    energy = np.mean(np.sum((C @ windows) ** 2, axis=1), axis=1)
    weights = energy * len(windows) * length  # averaged output loss, equation (3.7)
    data = training_view(windows, training_time, weights)
    for kind, initialization in (
        ("nitrom_lall", "balanced_opinf"),
        ("nitrom_pod", "pod_opinf"),
    ):
        saved = previous[kind]
        seed = saved or models[initialization]
        model, projection = restore(seed, instability_threshold=1e6)
        # QR changes latent coordinates; transform the operators along with Psi.
        if (
            np.linalg.norm(projection.Psi.T @ projection.Psi - np.eye(settings.r))
            > 1e-10
        ):
            psi, A, H = orthonormalize_psi(
                projection.Phi, projection.Psi, model.A_1, model.A_3
            )
            model.update_params([A, H])
            projection.update([projection.Phi.copy(), psi])
        registry = ParamRegistry(model, projection)
        nitrom = NitromModule(
            data,
            registry,
            fom=OutputMap(C),
            n_substeps=settings.n_substeps,
            adjoint_method=settings.adjoint_method,
        )
        nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
        initial = global_loss(nitrom)
        tick = clock.perf_counter()
        train(
            nitrom,
            n_epochs=settings.nitrom_epochs,
            lr=1.0,
            optimizer_type="lbfgs",
            print_every=5,
            tol=1e-12,
        )
        registry.scatter([getattr(nitrom, name) for name in registry.names])
        models[kind] = checkpoint(
            kind,
            model,
            projection,
            settings,
            signature,
            module=nitrom,
            previous=saved,
            training_seconds=clock.perf_counter() - tick,
        )
        models[kind]["initial_loss"] = initial
        save_checkpoint(directory, models[kind])

    if rank == 0:
        evaluate(directory, models, X, time, C, settings)
        print(f"Models and results: {directory} ({clock.perf_counter() - start:.2f} s)")


if __name__ == "__main__":
    main()
