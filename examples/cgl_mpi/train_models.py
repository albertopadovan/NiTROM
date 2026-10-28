"""Train the CGL NiTROM refinements with one trajectory shard per MPI rank."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cgl"))

import numpy as np
from _utils import (
    checkpoint,
    evaluate,
    load_training,
    prepare,
    resume_checkpoint,
)
from mpi4py import MPI
from settings import arguments
from train_models import (
    REFINEMENTS,
    initialize_models,
    make_nitrom,
    nitrom_data,
    opinf_data,
)

from nitrom.backend import set_backend
from nitrom.checkpointing import load_checkpoint, save_checkpoint
from nitrom.optimization import train


def main(argv=None):
    settings, directory, args = arguments(__doc__, argv, training=True)
    set_backend("numpy")
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    X, derivatives, time, amplitudes = load_training(directory)
    if size > len(X):
        raise ValueError("Use at most eight ranks: each rank needs a trajectory.")
    with np.load(directory / "data" / "testing.npz") as testing:
        C = testing["C"]

    # Basis construction and regularization selection run once on rank zero.
    models, previous = None, None
    if rank == 0:
        prepare(directory)
        previous = {
            kind: resume_checkpoint(directory, kind, settings, args.fresh)
            for kind in REFINEMENTS
        }
        data = opinf_data(X, derivatives, time, C)
        data.pool = SimpleNamespace(comm=MPI.COMM_SELF)
        models = initialize_models(settings, X, time, amplitudes, C, data)
        for kind, saved in models.items():
            save_checkpoint(directory / "models" / f"{kind}.npz", saved)
    # Broadcast only restart choices; each rank reconstructs its own components.
    previous = comm.bcast(
        {kind: saved is not None for kind, saved in previous.items()}
        if rank == 0
        else None,
        root=0,
    )

    # Normalize on the full dataset, THEN shard. Each local loss is a partial sum.
    data = nitrom_data(settings, X, time, C)
    data.X = data.X[rank::size]
    data.weights = data.weights[rank::size]
    data.pool = SimpleNamespace(comm=comm)

    for kind, initialization in REFINEMENTS.items():
        saved = (
            load_checkpoint(directory / "models" / f"{kind}.npz")
            if previous[kind]
            else None
        )
        seed = saved or load_checkpoint(directory / "models" / f"{initialization}.npz")
        module = make_nitrom(settings, data, C, seed)
        initial = comm.allreduce(float(module()), op=MPI.SUM)
        # NiTROM reduces costs and gradients internally; every rank trains in lockstep.
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
        final = comm.allreduce(float(module()), op=MPI.SUM)
        if rank == 0:
            models[kind] = checkpoint(
                kind,
                module.model,
                module.projection,
                settings,
                module=module,
                previous=saved,
                final_loss=final,
            )
            models[kind].metadata["initial_loss"] = initial
            save_checkpoint(directory / "models" / f"{kind}.npz", models[kind])

    if rank == 0:
        evaluate(directory, models, X, time, C, settings)
        print(f"Models and results: {directory} ({size} MPI ranks)")


if __name__ == "__main__":
    main()
