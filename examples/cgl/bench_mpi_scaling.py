"""
Time one NiTROM cost and gradient evaluation, averaged over REPS calls.

Call it once per rank count and compare the printed totals:

    E=/Users/albertopadovan/miniconda3/envs/res4py_env/bin
    PYTHONPATH=../../src:. OMP_NUM_THREADS=1 $E/mpiexec -n 1 $E/python -u bench_mpi_scaling.py
    PYTHONPATH=../../src:. OMP_NUM_THREADS=1 $E/mpiexec -n 2 $E/python -u bench_mpi_scaling.py
    PYTHONPATH=../../src:. OMP_NUM_THREADS=1 $E/mpiexec -n 4 $E/python -u bench_mpi_scaling.py

Three pieces are timed separately, since they scale differently:

    forward   module()            -- integrate the ROM, compare outputs
    gradient  module.gradient()   -- backward adjoint sweep
    reduce    all-reduce of the scalar cost and of the gradient list

Each rank owns a strided shard of the trajectories, exactly as train_nitrom.py
does, so the numbers are the real training workload.  Wall time is set by the
slowest rank, so the figures are the max over ranks, averaged over REPS calls
after one warm-up.

Run from this directory (data/ is loaded by relative path).  Keep
OMP_NUM_THREADS=1: otherwise every rank spawns its own BLAS threads, which
contend for the same cores and corrupt the comparison.  There are 16 training
trajectories, so 16 ranks is the maximum.
"""

import time as timer

import numpy as np

import train_nitrom as T
from cgl import CGL
from nitrom.backend import mpi_allreduce_scalar, mpi_allreduce_sum, mpi_rank_size
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

REPS = 10

RANK, WORLD = mpi_rank_size()


def main():
    cgl = CGL()
    nx = cgl.nx
    g = np.exp(-((cgl.x + cgl.xbar)/cgl.s)**2)
    C = np.zeros((2, 2*nx))
    C[0, :nx], C[1, nx:] = g, g

    alphas = np.load("data/train_alpha.npy")
    time = np.load("data/time.npy")[:T.N_SNAP]
    X = np.stack([a*np.load(f"data/train_traj_{j:03d}.npy")[:, :T.N_SNAP]
                  for j, a in enumerate(alphas)])
    Y = np.matmul(C, X)
    weights = np.mean(np.sum(Y**2, axis=1), axis=1)*X.shape[0]*X.shape[2]

    if WORLD > X.shape[0]:
        raise ValueError(
            f"{WORLD} ranks for {X.shape[0]} trajectories; use at most one "
            f"rank per trajectory."
        )
    mine = list(range(RANK, X.shape[0], WORLD))
    data = T.Data(X[mine], time, weights[mine])

    d = np.load("roms_r5.npy", allow_pickle=True).item()[T.BASIS]
    model = PolynomialModel(
        5, [1, 3], dtype=np.float64,
        tensors=[np.ascontiguousarray(d["Ar"]), T.sym_to_tensor(d["Hr"], 5)],
    )
    projection = LinearProjection([np.ascontiguousarray(d["Phi"]),
                                   np.ascontiguousarray(d["Psi"])])
    nit = NitromModule(data, ParamRegistry(model, projection), fom=T.FOM(C),
                       reg=0.0, n_substeps=T.N_SUBSTEPS, adjoint_method="discrete")
    nit.set_unlearnable("Phi", "Psi")

    if RANK == 0:
        print(f"ranks {WORLD} | {len(mine)} traj/rank | {T.N_SNAP} snapshots | "
              f"{T.N_SUBSTEPS} substeps | warming up ...", flush=True)
    t0 = timer.perf_counter()
    nit()
    nit.gradient()                                   # warm up
    if RANK == 0:
        one = timer.perf_counter() - t0
        print(f"  one cost+gradient: {one:.2f} s  ->  {REPS} reps ~ "
              f"{REPS*one:.0f} s", flush=True)

    tf = tg = tr = 0.0
    for k in range(REPS):
        ta = timer.perf_counter()
        cost = float(nit())
        t1 = timer.perf_counter()
        grads = nit.gradient()
        t2 = timer.perf_counter()
        if WORLD > 1:
            cost = mpi_allreduce_scalar(cost)
            grads = [mpi_allreduce_sum(np.asarray(gi)) for gi in grads]
        t3 = timer.perf_counter()
        tf += t1 - ta
        tg += t2 - t1
        tr += t3 - t2
        if RANK == 0:
            print(f"  rep {k + 1:2d}/{REPS}  forward {t1 - ta:6.3f} s  "
                  f"gradient {t2 - t1:6.3f} s", flush=True)

    local = np.array([tf, tg, tr, tf + tg + tr])/REPS
    if WORLD > 1:
        from mpi4py import MPI
        local = MPI.COMM_WORLD.allreduce(local, op=MPI.MAX)

    if RANK == 0:
        print(f"\nranks {WORLD:>3} | {len(mine):>2} traj/rank | {T.N_SNAP} snapshots | "
              f"avg of {REPS} calls, max over ranks")
        print(f"   forward  {local[0]:8.3f} s")
        print(f"   gradient {local[1]:8.3f} s")
        print(f"   reduce   {local[2]:8.4f} s")
        print(f"   total    {local[3]:8.3f} s   (cost = {cost:.6e})", flush=True)


if __name__ == "__main__":
    main()
