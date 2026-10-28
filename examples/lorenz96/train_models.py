"""Fit OpInf, ATR-OpInf, NiTROM, and ATR-NiTROM at an explicitly chosen rank."""

import numpy as np
from _utils import (
    basis_pool,
    checkpoint,
    evaluate,
    load_training,
    prepare,
    resume_checkpoint,
    training_view,
)
from model import Lorenz96
from settings import arguments

from nitrom.backend import set_backend
from nitrom.bases import compute_pod
from nitrom.checkpointing import load_checkpoint, save_checkpoint
from nitrom.latent_space_models import AtrPolynomialModel, PolynomialModel
from nitrom.optimization import NitromModule, OpInfModule, solve_opinf, train
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry

REFINEMENTS = {"nitrom": "opinf", "atr_nitrom": "atr_opinf"}


def initialize_models(settings, X, dX, time, amplitudes, mean):
    U, singular, _ = compute_pod(basis_pool(X))
    if settings.r > U.shape[1]:
        raise ValueError("--r exceeds the number of available POD modes.")
    Phi = U[:, : settings.r].copy()
    data = training_view(X, dX, time, amplitudes)
    forcing = {"forcing_exists": True, "m": 1}
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    model = PolynomialModel(settings.r, [0, 1, 2], forcing_config=forcing)
    opinf = OpInfModule(data, model, projection, reg={2: settings.opinf_reg})
    solve_opinf(opinf, solver="lstsq")
    models = {"opinf": checkpoint("opinf", model, projection, settings, mean, opinf)}

    # Retract the unconstrained fit, preserving its learned input operator.
    r = settings.r
    params = [
        np.zeros((r, r)),
        np.eye(r),
        np.eye(r),
        np.zeros((r, r, r)),
        np.zeros(r),
        np.zeros(r),
        model.B.copy(),
    ]
    atr = AtrPolynomialModel(r, [1, 2], atr_params=params, forcing_config=forcing)
    atr.set_shift_from_data(opinf.Z)
    atr.retract_general_tensors_to_atr_tensors(model.inner_params()[:3])
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    module = OpInfModule(data, atr, projection, reg={2: settings.opinf_reg})
    train(
        module,
        n_epochs=settings.atr_opinf_epochs,
        lr=1.0,
        optimizer_type="lbfgs",
        print_every=5,
        tol=1e-12,
    )
    atr.update_params([getattr(module, name) for name in atr.param_names])
    models["atr_opinf"] = checkpoint(
        "atr_opinf", atr, projection, settings, mean, module
    )
    for saved in models.values():
        saved.metadata["pod_retained_energy"] = float(
            np.sum(singular[:r] ** 2) / np.sum(singular**2)
        )
    return models


def make_nitrom(settings, data, seed):
    module = NitromModule(
        data,
        ParamRegistry(seed.model, seed.projection),
        fom=Lorenz96(settings.n, settings.F),
        n_substeps=settings.n_substeps,
        adjoint_method=settings.adjoint_method,
    )
    module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    return module


def main(argv=None):
    settings, directory, args = arguments(__doc__, argv, training=True)
    set_backend("numpy")
    X, dX, time, amplitudes, mean = load_training(directory)
    if X.shape[1] != settings.n or mean.shape != (settings.n,):
        raise ValueError("Training data dimensions do not match the settings.")
    previous = {
        kind: resume_checkpoint(directory, kind, settings, mean, args.fresh)
        for kind in REFINEMENTS
    }
    X = X - mean[None, :, None]
    prepare(directory)
    models = initialize_models(settings, X, dX, time, amplitudes, mean)
    for kind, saved in models.items():
        save_checkpoint(directory / "models" / f"{kind}.npz", saved)
    keep = time <= settings.nitrom_end + 1e-10
    if keep.sum() < 2:
        raise ValueError("The NiTROM prefix must contain at least two snapshots.")
    data = training_view(X[:, :, keep], dX[:, :, keep], time[keep], amplitudes)
    for kind, initialization in REFINEMENTS.items():
        saved = previous[kind]
        seed = saved or load_checkpoint(directory / "models" / f"{initialization}.npz")
        module = make_nitrom(settings, data, seed)
        initial_loss = float(module())
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
            mean,
            module,
            previous=saved,
        )
        models[kind].metadata.update(
            initial_loss=initial_loss,
            pod_retained_energy=seed.metadata["pod_retained_energy"],
        )
        save_checkpoint(directory / "models" / f"{kind}.npz", models[kind])
    evaluate(directory, models, settings, mean)
    print(f"Models and results: {directory}")


if __name__ == "__main__":
    main()
