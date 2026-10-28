"""Compare POD/balanced cubic OpInf and NiTROM refinements using public APIs."""

import numpy as np
from _utils import (
    basis_pool,
    checkpoint,
    evaluate,
    load_training,
    orthonormalize_psi,
    prepare,
    resume_checkpoint,
    selection_cost,
    training_view,
)
from cgl import OutputMap
from settings import arguments

from nitrom.backend import set_backend
from nitrom.bases import compute_data_driven_balancing, compute_pod
from nitrom.checkpointing import load_checkpoint, save_checkpoint
from nitrom.latent_space_models import PolynomialModel
from nitrom.optimization import NitromModule, OpInfModule, solve_opinf, train
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry

REFINEMENTS = {"nitrom_lall": "balanced_opinf", "nitrom_pod": "pod_opinf"}


def opinf_data(X, derivatives, time, C):
    # The derivative fit is energy-normalized but unaveraged. Trim FD endpoints.
    energy = np.mean(np.sum((C @ X) ** 2, axis=1), axis=1)
    return training_view(X[:, :, 2:-2], time[2:-2], energy, derivatives[:, :, 2:-2])


def nitrom_data(settings, X, time, C):
    # One full trajectory window; average over trajectories and saved times.
    keep = time <= settings.nitrom_end + 1e-10
    X, time = X[:, :, keep], time[keep]
    energy = np.mean(np.sum((C @ X) ** 2, axis=1), axis=1)
    return training_view(X, time, energy * len(X) * len(time))


def initialize_models(settings, X, time, amplitudes, C, data):
    """Construct both bases and select cubic regularization on training rollouts."""
    families = [list(np.where(amplitudes == a)[0]) for a in np.unique(amplitudes)]
    Phi, Psi, Sigma, _ = compute_data_driven_balancing(
        basis_pool(X, time),
        n_checkpoints=settings.balancing_checkpoints,
        checkpoint_stride=settings.balancing_stride,
        families=families,
    )
    U, _, _ = compute_pod(basis_pool(X, time, amplitudes**2), normalize=True)
    bases = {
        "balanced_opinf": (Phi[:, : settings.r], Psi[:, : settings.r]),
        "pod_opinf": (U[:, : settings.r], U[:, : settings.r]),
    }
    models = {}
    for kind, (phi, psi) in bases.items():
        best, best_cost = None, np.inf
        for reg in 10.0 ** np.arange(-2, 9):
            model = PolynomialModel(settings.r, [1, 3])
            projection = LinearProjection([phi.copy(), psi.copy()])
            opinf = OpInfModule(data, model, projection, reg={3: float(reg)})
            solve_opinf(opinf, solver="lstsq")
            cost = selection_cost(model, projection, X, time, C)
            if cost < best_cost:
                best_cost = cost
                best = checkpoint(kind, model, projection, settings, module=opinf)
                best.metadata.update(selected_reg=float(reg), selection_cost=cost)
        if best is None:
            raise RuntimeError(f"No finite regularization-selection cost for {kind}.")
        best.metadata["Sigma"] = Sigma.copy() if kind == "balanced_opinf" else None
        models[kind] = best
        print(
            f"{kind}: reg={best.metadata['selected_reg']:g}, selection cost={best_cost:.6g}"
        )
    return models


def make_nitrom(settings, data, C, seed):
    model, projection = seed.model, seed.projection
    # QR changes latent coordinates: the operators must move with the test basis.
    if not np.allclose(
        projection.Psi.T @ projection.Psi, np.eye(settings.r), atol=1e-10
    ):
        psi, A, H = orthonormalize_psi(projection.Psi, model.A_1, model.A_3)
        model.update_params([A, H])
        projection.update([projection.Phi.copy(), psi])
    module = NitromModule(
        data,
        ParamRegistry(model, projection),
        fom=OutputMap(C),
        n_substeps=settings.n_substeps,
        adjoint_method=settings.adjoint_method,
    )
    module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    return module


def main(argv=None):
    settings, directory, args = arguments(__doc__, argv, training=True)
    set_backend("numpy")
    prepare(directory)
    previous = {
        kind: resume_checkpoint(directory, kind, settings, args.fresh)
        for kind in REFINEMENTS
    }
    X, derivatives, time, amplitudes = load_training(directory)
    with np.load(directory / "data" / "testing.npz") as testing:
        C = testing["C"]
    models = initialize_models(
        settings, X, time, amplitudes, C, opinf_data(X, derivatives, time, C)
    )
    for kind, saved in models.items():
        save_checkpoint(directory / "models" / f"{kind}.npz", saved)
    data = nitrom_data(settings, X, time, C)
    for kind, initialization in REFINEMENTS.items():
        saved = previous[kind]
        seed = saved or load_checkpoint(directory / "models" / f"{initialization}.npz")
        module = make_nitrom(settings, data, C, seed)
        initial = float(module())
        train(
            module,
            n_epochs=settings.nitrom_epochs,
            lr=1.0,
            optimizer_type="lbfgs",
            print_every=5,
            tol=1e-12,
        )
        module.registry.scatter(
            [getattr(module, name) for name in module.registry.names]
        )
        models[kind] = checkpoint(
            kind,
            module.model,
            module.projection,
            settings,
            module=module,
            previous=saved,
        )
        models[kind].metadata["initial_loss"] = initial
        save_checkpoint(directory / "models" / f"{kind}.npz", models[kind])
    evaluate(directory, models, X, time, C, settings)
    print(f"Models and results: {directory}")


if __name__ == "__main__":
    main()
