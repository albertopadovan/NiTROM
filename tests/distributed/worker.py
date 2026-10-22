"""Exercise separate two-rank groups with different training targets."""

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nitrom.backend import set_backend
from nitrom.bases import compute_data_driven_balancing, compute_pod
from nitrom.latent_space_models import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf, train
from nitrom.projections import LinearProjection
from nitrom.training_data import TrainingData, TrainingPool


def check(mode, directory, world_rank=None):
    backend = set_backend(mode)
    if mode == "numpy":
        from mpi4py import MPI

        world_rank = MPI.COMM_WORLD.Get_rank()
        color = world_rank % 2
        comm = MPI.COMM_WORLD.Split(color, world_rank)
        rank, size = comm.Get_rank(), comm.Get_size()
        barrier = comm.Barrier
    else:
        import torch.distributed as dist

        dist.init_process_group(
            "gloo",
            init_method=(directory / "rendezvous").as_uri(),
            rank=world_rank,
            world_size=4,
            timeout=timedelta(seconds=25),
        )
        world_rank = dist.get_rank()
        groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        color = world_rank % 2
        comm = groups[color]
        rank, size = dist.get_rank(group=comm), dist.get_world_size(group=comm)

        def barrier():
            return dist.barrier(group=comm)

    directory = directory / f"group_{color}"
    coefficient = -0.2 * (color + 1)
    X = np.random.default_rng(20 + color).standard_normal((4, 4, 4))
    if rank == 0:
        directory.mkdir()
        np.save(directory / "time.npy", np.linspace(0, 1, 4))
        for i, trajectory in enumerate(X):
            np.save(directory / f"traj_{i}.npy", trajectory)
            np.save(directory / f"deriv_{i}.npy", coefficient * trajectory)
    barrier()
    pool = TrainingPool(
        4,
        str(directory / "traj_%d.npy"),
        str(directory / "time.npy"),
        comm=comm,
        dtype=backend.float64,
    )
    assert (pool.rank, pool.world_size) == (rank, size)
    assert pool.comm is comm
    data = TrainingData(pool, [0, 1, 2, 3])
    reference_pool = SimpleNamespace(
        X=backend.asarray(X),
        time=pool.time,
        weights=backend.asarray(np.ones(4)),
        backend=backend,
        N=4,
        n_traj=4,
        n_snapshots=4,
        world_size=1,
        rank=0,
        comm=None,
    )
    _, actual_s, _ = compute_pod(pool)
    _, expected_s, _ = compute_pod(reference_pool)
    np.testing.assert_allclose(
        backend.to_numpy(actual_s), backend.to_numpy(expected_s), rtol=1e-12
    )
    actual = compute_data_driven_balancing(pool, rank_M=3)
    expected = compute_data_driven_balancing(reference_pool, rank_M=3)
    np.testing.assert_allclose(
        backend.to_numpy(actual[2]),
        backend.to_numpy(expected[2]),
        atol=1e-10,
        rtol=1e-10,
    )

    phi = backend.asarray(np.eye(4, 2))

    def module():
        model = PolynomialModel(2, [1])
        return OpInfModule(data, model, LinearProjection([phi, backend.copy(phi)]))

    for solver in ("normal", "lstsq"):
        for penalty in (0.0, 0.2):
            fitted = module()
            fitted.reg = {1: penalty}
            solve_opinf(fitted, solver=solver)
            Z = X[:, :2, :].transpose(1, 0, 2).reshape(2, -1)
            gram = Z @ Z.T / (X.shape[0] * X.shape[-1])
            expected = np.linalg.solve(gram + penalty * np.eye(2), coefficient * gram).T
            np.testing.assert_allclose(
                backend.to_numpy(fitted.rom.get_params()[0]), expected, atol=1e-12
            )
    trained = module()
    before = float(trained())
    train(trained, n_epochs=3, optimizer_type="sgd", lr=0.01, print_every=0, tol=0)
    assert float(trained()) < before
    if mode == "numpy":
        results = comm.allgather(trained.rom.get_params()[0])
    else:
        results = [None] * size
        dist.all_gather_object(
            results, backend.to_numpy(trained.rom.get_params()[0]), group=comm
        )
    for result in results[1:]:
        np.testing.assert_array_equal(results[0], result)
    barrier()
    if mode == "torch":
        dist.destroy_process_group()
    else:
        comm.Free()


def torch_worker(rank, directory):
    check("torch", directory, rank)


if __name__ == "__main__":
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode == "torch":
        import torch.multiprocessing as mp

        mp.spawn(torch_worker, args=(directory,), nprocs=4)
    else:
        check(mode, directory)
