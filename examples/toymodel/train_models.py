"""Compare Galerkin, OpInf, GAS-OpInf, NiTROM, and GAS-NiTROM on the same data."""

import time as clock

from _utils import (
    checkpoint,
    data_signature,
    evaluate,
    global_loss,
    load_training,
    prepare,
    read_data_settings,
    resume_checkpoint,
    save_checkpoint,
)
from model import ToyModel
from settings import arguments

from nitrom.backend import set_backend
from nitrom.bases import compute_pod
from nitrom.distributed import mpi_rank_size
from nitrom.latent_space_models import GasPolynomialModel, PolynomialModel
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
        for kind in ("nitrom", "gas_nitrom")
    }
    pool, data = load_training(directory)
    fom = ToyModel(settings.beta)
    U, _, _ = compute_pod(pool, normalize=True)
    Phi = U[:, : settings.r].copy()
    tensors = fom.project(Phi, Phi)
    forcing = {"forcing_exists": True, "m": 1, "B": tensors[-1].copy()}
    models = {}
    rank, _ = mpi_rank_size()

    # 1. Galerkin: project the known full-order operators onto the POD basis.
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    model = PolynomialModel(
        settings.r, [1, 2], tensors=[x.copy() for x in tensors], forcing_config=forcing
    )
    models["galerkin"] = checkpoint(
        "galerkin", model, projection, settings, signature, forcing_config=forcing
    )
    save_checkpoint(directory, models["galerkin"])

    # 2. Ordinary OpInf: weighted least squares, with a fixed reduced input B.
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    model = PolynomialModel(settings.r, [1, 2], forcing_config=forcing)
    opinf = OpInfModule(data, model, projection, reg={2: settings.opinf_reg})
    opinf.set_unlearnable("B")
    solve_opinf(opinf, solver="lstsq")
    models["opinf"] = checkpoint(
        "opinf",
        model,
        projection,
        settings,
        signature,
        module=opinf,
        forcing_config=forcing,
    )
    save_checkpoint(directory, models["opinf"])

    # 3. GAS-OpInf: retract the ordinary fit, then optimize stable operators.
    seed = GasPolynomialModel(settings.r, [1, 2])
    seed.retract_general_tensors_to_gas_tensors(model.get_params()[:2])
    gas_params = [*seed.get_params(), forcing["B"].copy()]
    gas_model = GasPolynomialModel(
        settings.r, [1, 2], gas_params=gas_params, forcing_config=forcing
    )
    projection = LinearProjection([Phi.copy(), Phi.copy()])
    gas = OpInfModule(data, gas_model, projection)
    gas.set_unlearnable("B")
    initial = global_loss(gas)
    tick = clock.perf_counter()
    train(
        gas,
        n_epochs=settings.gas_opinf_epochs,
        lr=1.0,
        optimizer_type="lbfgs",
        print_every=5,
        tol=1e-14,
    )
    gas_model.update_params([getattr(gas, name) for name in gas_model.param_names])
    models["gas_opinf"] = checkpoint(
        "gas_opinf",
        gas_model,
        projection,
        settings,
        signature,
        module=gas,
        forcing_config=forcing,
        training_seconds=clock.perf_counter() - tick,
    )
    models["gas_opinf"]["initial_loss"] = initial
    save_checkpoint(directory, models["gas_opinf"])

    # 4. NiTROM: jointly learn the bases and unconstrained operators.
    saved = previous["nitrom"]
    initial_tensors = saved["tensors"] if saved else tensors
    initial_phi, initial_psi = (saved["Phi"], saved["Psi"]) if saved else (Phi, Phi)
    model = PolynomialModel(
        settings.r,
        [1, 2],
        tensors=[x.copy() for x in initial_tensors],
        forcing_config=forcing,
    )
    projection = LinearProjection([initial_phi.copy(), initial_psi.copy()])
    registry = ParamRegistry(model, projection)
    nitrom = NitromModule(
        data,
        registry,
        fom=fom,
        n_substeps=settings.n_substeps,
        adjoint_method=settings.adjoint_method,
    )
    nitrom.set_unlearnable("B")
    nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    initial = global_loss(nitrom)
    tick = clock.perf_counter()
    train(
        nitrom,
        n_epochs=settings.nitrom_epochs,
        lr=1.0,
        optimizer_type="lbfgs",
        print_every=5,
        tol=1e-14,
    )
    registry.scatter([getattr(nitrom, name) for name in registry.names])
    models["nitrom"] = checkpoint(
        "nitrom",
        model,
        projection,
        settings,
        signature,
        module=nitrom,
        previous=saved,
        forcing_config=forcing,
        training_seconds=clock.perf_counter() - tick,
    )
    models["nitrom"]["initial_loss"] = initial
    save_checkpoint(directory, models["nitrom"])

    # 5. GAS-NiTROM: warm start from its checkpoint, otherwise from GAS-OpInf.
    saved = previous["gas_nitrom"]
    seed = saved or models["gas_opinf"]
    model = GasPolynomialModel(
        settings.r,
        [1, 2],
        gas_params=[x.copy() for x in seed["gas_params"]],
        forcing_config=forcing,
    )
    projection = LinearProjection([seed["Phi"].copy(), seed["Psi"].copy()])
    registry = ParamRegistry(model, projection)
    gas_nitrom = NitromModule(
        data,
        registry,
        fom=fom,
        n_substeps=settings.n_substeps,
        adjoint_method=settings.adjoint_method,
    )
    gas_nitrom.set_unlearnable("B")
    gas_nitrom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
    initial = global_loss(gas_nitrom)
    tick = clock.perf_counter()
    train(
        gas_nitrom,
        n_epochs=settings.nitrom_epochs,
        lr=1.0,
        optimizer_type="lbfgs",
        print_every=5,
        tol=1e-14,
    )
    registry.scatter([getattr(gas_nitrom, name) for name in registry.names])
    models["gas_nitrom"] = checkpoint(
        "gas_nitrom",
        model,
        projection,
        settings,
        signature,
        module=gas_nitrom,
        previous=saved,
        forcing_config=forcing,
        training_seconds=clock.perf_counter() - tick,
    )
    models["gas_nitrom"]["initial_loss"] = initial
    save_checkpoint(directory, models["gas_nitrom"])

    if rank == 0:
        evaluate(directory, models, fom)
        print(f"Models and results: {directory} ({clock.perf_counter() - start:.2f} s)")


if __name__ == "__main__":
    main()
