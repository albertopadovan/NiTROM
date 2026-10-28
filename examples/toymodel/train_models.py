"""Compare Galerkin, OpInf, GAS-OpInf, NiTROM, and GAS-NiTROM on the same data."""

from _utils import (
    checkpoint,
    evaluate,
    load_training,
    prepare,
    resume_checkpoint,
)
from model import ToyModel
from settings import arguments

from nitrom.backend import set_backend
from nitrom.bases import compute_pod
from nitrom.checkpointing import load_checkpoint, save_checkpoint
from nitrom.latent_space_models import GasPolynomialModel, PolynomialModel
from nitrom.optimization import NitromModule, OpInfModule, solve_opinf, train
from nitrom.projections import LinearProjection
from nitrom.roms import ParamRegistry

REFINEMENTS = {"nitrom": "galerkin", "gas_nitrom": "gas_opinf"}


def initialize_models(settings, pool, data, fom):
    """Build the three baselines; each owns its operators and projection."""
    U, _, _ = compute_pod(pool, normalize=True)
    Phi = U[:, : settings.r].copy()
    tensors = fom.project(Phi, Phi)
    forcing = {"forcing_exists": True, "m": 1, "B": tensors[-1].copy()}
    models = {}

    # 1. Galerkin: project the known full-order operators onto the POD basis.
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    model = PolynomialModel(settings.r, [1, 2], tensors=tensors, forcing_config=forcing)
    models["galerkin"] = checkpoint("galerkin", model, projection, settings)

    # 2. OpInf: weighted least squares with a fixed reduced input operator.
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    model = PolynomialModel(settings.r, [1, 2], forcing_config=forcing)
    opinf = OpInfModule(data, model, projection, reg={2: settings.opinf_reg})
    opinf.set_unlearnable("B")
    solve_opinf(opinf, solver="lstsq")
    models["opinf"] = checkpoint("opinf", model, projection, settings, module=opinf)

    # 3. GAS-OpInf: retract the ordinary fit, then optimize stable operators.
    seed = GasPolynomialModel(settings.r, [1, 2])
    seed.retract_general_tensors_to_gas_tensors(model.get_params()[:2])
    model = GasPolynomialModel(
        settings.r,
        [1, 2],
        gas_params=[*seed.get_params(), forcing["B"].copy()],
        forcing_config=forcing,
    )
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    gas = OpInfModule(data, model, projection)
    gas.set_unlearnable("B")
    train(
        gas,
        n_epochs=settings.gas_opinf_epochs,
        lr=1.0,
        optimizer_type="lbfgs",
        print_every=5,
        tol=1e-14,
    )
    model.update_params([getattr(gas, name) for name in model.param_names])
    models["gas_opinf"] = checkpoint(
        "gas_opinf", model, projection, settings, module=gas
    )
    return models


def make_nitrom(settings, data, fom, seed):
    """Use the same training setup for ordinary and GAS-constrained dynamics."""
    model, projection = seed.model, seed.projection
    module = NitromModule(
        data,
        ParamRegistry(model, projection),
        fom=fom,
        n_substeps=settings.n_substeps,
        adjoint_method=settings.adjoint_method,
    )
    module.set_unlearnable("B")
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
    pool, data = load_training(directory)
    fom = ToyModel(settings.beta)
    models = initialize_models(settings, pool, data, fom)
    for kind, saved in models.items():
        save_checkpoint(directory / "models" / f"{kind}.npz", saved)

    # 4-5. Refine Galerkin and GAS-OpInf, or continue their latest checkpoints.
    for kind, initialization in REFINEMENTS.items():
        saved = previous[kind]
        seed = saved or load_checkpoint(directory / "models" / f"{initialization}.npz")
        module = make_nitrom(settings, data, fom, seed)
        initial = float(module())
        train(
            module,
            n_epochs=settings.nitrom_epochs,
            lr=1.0,
            optimizer_type="lbfgs",
            print_every=5,
            tol=1e-14,
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
    evaluate(directory, models, fom)
    print(f"Models and results: {directory}")


if __name__ == "__main__":
    main()
