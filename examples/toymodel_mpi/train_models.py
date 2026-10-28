"""Train the toy NiTROM refinements with one trajectory shard per MPI rank."""

import sys
from pathlib import Path
from types import SimpleNamespace

# Reuse the single-process example's physics, data, and model construction.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "toymodel"))

from _utils import (
    checkpoint,
    evaluate,
    load_training,
    prepare,
    resume_checkpoint,
)
from model import ToyModel
from mpi4py import MPI
from settings import arguments
from train_models import REFINEMENTS, initialize_models, make_nitrom

from nitrom.backend import set_backend
from nitrom.checkpointing import load_checkpoint, save_checkpoint
from nitrom.optimization import train


def main(argv=None):
    settings, directory, args = arguments(__doc__, argv, training=True)
    set_backend("numpy")
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    pool, data = load_training(directory)
    if size > len(data.X):
        raise ValueError("Use at most four ranks: each rank needs a trajectory.")
    fom = ToyModel(settings.beta)

    # Build serial baselines on rank zero; COMM_SELF isolates these local solves.
    models, previous = None, None
    if rank == 0:
        prepare(directory)
        previous = {
            kind: resume_checkpoint(directory, kind, settings, args.fresh)
            for kind in REFINEMENTS
        }
        data.pool = SimpleNamespace(comm=MPI.COMM_SELF)
        models = initialize_models(settings, pool, data, fom)
        for kind, saved in models.items():
            save_checkpoint(directory / "models" / f"{kind}.npz", saved)
    # Broadcast only restart choices; each rank reconstructs its own components.
    previous = comm.bcast(
        {kind: saved is not None for kind, saved in previous.items()}
        if rank == 0
        else None,
        root=0,
    )

    # Keep the GLOBAL averaging weights computed before taking the local shard.
    data.X = data.X[rank::size]
    data.dX = data.dX[rank::size]
    data.weights = data.weights[rank::size]
    data.forcing_fns = data.forcing_fns[rank::size]
    data.pool = SimpleNamespace(comm=comm)

    for kind, initialization in REFINEMENTS.items():
        saved = (
            load_checkpoint(directory / "models" / f"{kind}.npz")
            if previous[kind]
            else None
        )
        seed = saved or load_checkpoint(directory / "models" / f"{initialization}.npz")
        module = make_nitrom(settings, data, fom, seed)
        initial = comm.allreduce(float(module()), op=MPI.SUM)
        # NiTROM sums local costs and gradients over data.pool.comm internally.
        # All ranks run the same optimizer and keep identical model parameters.
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

    # Only rank zero writes files and evaluates the held-out responses.
    if rank == 0:
        evaluate(directory, models, fom)
        print(f"Models and results: {directory} ({size} MPI ranks)")


if __name__ == "__main__":
    main()
