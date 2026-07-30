from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dill
import numpy as np

from .backend import (
    comm_rank_size,
    distributed_rank_size,
    get_backend,
    mpi_comm_world,
)


class TrainingPool:
    """Training data pool for trajectory-based optimization.

    Loads trajectory snapshots, optional per-trajectory weights, steady
    forcing fields, and time derivatives from disk, and distributes them
    across ranks when running in a multi-process setting.  Arrays are created
    with the active backend (NumPy or PyTorch).

    Parameters
    ----------
    n_traj : int
        Total number of trajectories to load from disk.
    fname_traj : str
        Format string for trajectory files (e.g., ``'traj_%03d.npy'``).
    fname_time : str
        Format string for the time file.
    dtype : optional
        Data type for all arrays. Defaults to the backend's ``float32``.
    device : str, optional
        Device on which arrays are allocated (ignored by NumPy). Default ``'cpu'``.
    comm : optional
        Communicator defining the process group over which trajectories are
        sharded -- either an ``mpi4py`` communicator (e.g. a split of
        ``COMM_WORLD``) or a ``torch.distributed`` process group; its rank/size
        are read through whichever API it exposes. If ``None`` (default), the
        multiprocessing context is auto-detected -- ``MPI.COMM_WORLD`` under
        ``mpiexec`` (NumPy backend), ``torch.distributed``/``torchrun`` for
        PyTorch, single-process otherwise -- so a parallel job shards correctly
        without the caller having to wire up rank/size by hand.
    **kwargs
        Optional ``fname_weights``, ``fname_forcing``, ``fname_derivs``.
    """

    def __init__(
        self,
        n_traj: int,
        fname_traj: str,
        fname_time: str,
        dtype: Any = None,
        device: str = "cpu",
        comm: Any = None,
        **kwargs: Any,
    ) -> None:
        self.backend = get_backend()
        self.dtype = dtype if dtype is not None else self.backend.float32
        self.device = device
        # Figure out the process group ourselves: an explicit communicator wins;
        # otherwise auto-detect from the active backend (COMM_WORLD under
        # mpiexec, torch.distributed under torchrun, else single-process).  The
        # resolved communicator is stored so collectives (e.g. compute_POD) use
        # the same group the data was sharded over.
        if comm is not None:
            self.comm = comm
            self.rank, self.world_size = comm_rank_size(comm)
        else:
            self.rank, self.world_size = distributed_rank_size()
            # Keep COMM_WORLD around for numpy collectives (None if MPI absent).
            self.comm = mpi_comm_world() if self.backend.is_numpy else None

        self.shift_start_times = kwargs.get("shift_start_times", None)
        if self.shift_start_times is not None:
            self.num_shifts = len(self.shift_start_times)
        else:
            self.num_shifts = int(kwargs.get("num_shifts", 1))

        if self.num_shifts < 1:
            raise ValueError("num_shifts must be >= 1")

        n_virtual = n_traj * self.num_shifts
        if n_virtual <= 0:
            raise ValueError(
                f"n_traj must be a positive integer. Currently, n_traj = {n_traj}."
            )
        self.n_traj = n_virtual
        self.is_distributed = self.world_size > 1

        # Distribute virtual trajectories across ranks
        self.my_n_traj = n_virtual // self.world_size
        self.my_n_traj += 1 if self.rank < n_virtual % self.world_size else 0

        if self.my_n_traj == 0:
            raise ValueError("Every rank needs to own at least one trajectory")

        start_idx = self.rank * (n_virtual // self.world_size) + min(
            self.rank, n_virtual % self.world_size
        )
        self.traj_indices = list(range(start_idx, start_idx + self.my_n_traj))

        # Load data from file (mapping virtual indices to physical indices)
        self.load_trajectories(fname_traj)
        self.load_weights(kwargs)
        self.load_forcing(kwargs)
        self.load_time_derivatives(kwargs)
        self.time = self.backend.asarray(
            np.load(fname_time), dtype=self.dtype, device=self.device
        )

    def load_trajectories(self, fname_traj: str) -> None:
        """Load trajectory snapshots from ``.npy`` files into :attr:`X`."""
        self.fnames_traj = [fname_traj % (k // self.num_shifts) for k in self.traj_indices]
        X = [np.load(f) for f in self.fnames_traj]
        self.X = self.backend.asarray(
            np.stack(X), dtype=self.dtype, device=self.device
        )
        _, self.N, self.n_snapshots = self.X.shape

    def load_weights(self, kwargs: dict[str, Any]) -> None:
        """Load per-trajectory importance weights (default: all ones)."""
        fname_weights: str | None = kwargs.get("fname_weights")
        if fname_weights is not None:
            self.fnames_weights = [fname_weights % (k // self.num_shifts) for k in self.traj_indices]
            weights = [np.load(f) for f in self.fnames_weights]
            self.weights = self.backend.asarray(
                np.stack(weights).reshape(-1), dtype=self.dtype, device=self.device
            )
        else:
            self.weights = self.backend.zeros(
                (self.my_n_traj,), dtype=self.dtype, device=self.device
            ) + 1.0

    def load_forcing(self, kwargs: dict[str, Any]) -> None:
        """Load per-trajectory forcing callables from pickle files."""
        fname_forcing: str | None = kwargs.get("fname_forcing")
        if fname_forcing is not None:
            self.fnames_forcing = [fname_forcing % (k // self.num_shifts) for k in self.traj_indices]
            self.forcing_fns: list[Callable] = []
            for f in self.fnames_forcing:
                with open(f, "rb") as fh:
                    fn = dill.load(fh)
                if not callable(fn):
                    raise TypeError(f"Object loaded from {f} is not callable")
                self.forcing_fns.append(self._wrap_forcing(fn))
        else:
            self.forcing_fns = []

    def _wrap_forcing(self, fn: Callable) -> Callable:
        """Wrap a forcing callable so it returns a backend array of the right dtype."""

        def wrapped(t):
            return self.backend.asarray(
                fn(t), dtype=self.dtype, device=self.device
            )

        return wrapped

    def load_time_derivatives(self, kwargs: dict[str, Any]) -> None:
        """Load precomputed time derivatives (default: zeros)."""
        fname_deriv: str | None = kwargs.get("fname_derivs")
        if fname_deriv is not None:
            self.fnames_deriv = [fname_deriv % (k // self.num_shifts) for k in self.traj_indices]
            dX = [np.load(f) for f in self.fnames_deriv]
            self.dX = self.backend.asarray(
                np.stack(dX), dtype=self.dtype, device=self.device
            )
        else:
            self.dX = self.backend.zeros(
                (self.my_n_traj, self.N, self.n_snapshots),
                dtype=self.dtype,
                device=self.device,
            )


class TrainingData:
    def __init__(
        self,
        pool,
        which_trajs,
        percent_time_length,
        leggauss_deg,
        nsave_rom,
        **kwargs,
    ):
        """
        Training-data view passed to the optimizer.

        pool:                an instance of TrainingPool
        which_trajs:         integer indices selecting a subset of pool's trajectories
        percent_time_length: fraction in (0, 1] of each trajectory's snapshots to use
        leggauss_deg:        number of Gauss-Legendre quadrature points for the gradient
        nsave_rom:           number of ROM snapshots stored between two FOM snapshots
        """
        self.pool = pool
        self.backend = pool.backend
        bkend = self.backend

        shift_start_times = kwargs.get("shift_start_times", getattr(pool, "shift_start_times", None))
        num_shifts = kwargs.get("num_shifts", pool.num_shifts)
        if shift_start_times is not None:
            num_shifts = len(shift_start_times)
        if num_shifts < 1:
            raise ValueError("num_shifts must be >= 1")

        # Automatically expand physical trajectory selections in which_trajs to virtual ones
        physical_n_traj = pool.n_traj // pool.num_shifts
        if all(0 <= t < physical_n_traj for t in which_trajs):
            virtual_trajs = []
            for t in which_trajs:
                for s in range(num_shifts):
                    virtual_trajs.append(t * num_shifts + s)
            which_trajs = virtual_trajs

        self.global_trajs = which_trajs
        self.local_trajs = self._global_to_local_indices(which_trajs)

        # Number of snapshots to keep from percent_time_length
        n_snapshots_total = pool.X.shape[2]
        n_keep = max(1, int(percent_time_length * n_snapshots_total))
        self.time = pool.time[:n_keep]

        if len(self.local_trajs) > 0:
            if shift_start_times is not None:
                pool_time_np = bkend.to_numpy(pool.time)
                start_indices = []
                for t_s in shift_start_times:
                    idx = int(np.argmin(np.abs(pool_time_np - t_s)))
                    if idx + n_keep > n_snapshots_total:
                        raise ValueError(
                            f"Start time {t_s} corresponds to index {idx}, but the window "
                            f"of width {n_keep} exceeds the total snapshots ({n_snapshots_total})."
                        )
                    start_indices.append(idx)
                start_indices = np.array(start_indices, dtype=int)
            else:
                max_start_idx = n_snapshots_total - n_keep
                if max_start_idx < 0:
                    raise ValueError(
                        f"n_keep ({n_keep}) is larger than n_snapshots_total ({n_snapshots_total})"
                    )
                start_indices = np.linspace(0, max_start_idx, num_shifts, dtype=int)
            
            X_list = []
            dX_list = []
            forcing_fns_list = []
            
            for local_idx in self.local_trajs:
                g_idx = self.global_trajs[local_idx]
                shift_idx = int(g_idx % num_shifts)
                start_idx = start_indices[shift_idx]
                
                # Use local_idx:local_idx+1 to preserve 3D shape (1, N, n_keep)
                X_slice = pool.X[local_idx : local_idx + 1, :, start_idx : start_idx + n_keep]
                dX_slice = pool.dX[local_idx : local_idx + 1, :, start_idx : start_idx + n_keep]
                
                X_list.append(X_slice)
                dX_list.append(dX_slice)
                
                if pool.forcing_fns:
                    fn = pool.forcing_fns[local_idx]
                    if fn is not None:
                        shift_time = float(pool.time[start_idx] - pool.time[0])
                        def make_shifted_fn(original_fn, t_shift):
                            return lambda t: original_fn(t + t_shift)
                        forcing_fns_list.append(make_shifted_fn(fn, shift_time))
                    else:
                        forcing_fns_list.append(None)
            
            self.X = bkend.concatenate(X_list, axis=0)
            self.dX = bkend.concatenate(dX_list, axis=0)
            self.forcing_fns = forcing_fns_list
            self.weights = pool.weights[self.local_trajs]
        else:
            shape = (0, pool.N, n_keep)
            self.X = bkend.zeros(shape, device=pool.device, dtype=pool.dtype)
            self.dX = bkend.zeros(shape, device=pool.device, dtype=pool.dtype)
            self.forcing_fns = []
            self.weights = bkend.empty((0,), device=pool.device, dtype=pool.dtype)

        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.nsave_rom = nsave_rom

        # Gauss-Legendre quadrature points and weights
        self.leggauss_deg = leggauss_deg
        tlg, wlg = np.polynomial.legendre.leggauss(self.leggauss_deg)
        self.tlg = bkend.asarray(tlg, device=pool.device, dtype=pool.dtype)
        self.wlg = bkend.asarray(wlg, device=pool.device, dtype=pool.dtype)

        # Scale the weights so the cost measures the average error over
        # snapshots and trajectories.
        self.weights = self.weights * (len(self.global_trajs) * self.n_snapshots)

        # Parse the keyword arguments
        self.which_fix = kwargs.get("which_fix", "fix_none")
        if self.which_fix not in ["fix_tensors", "fix_bases", "fix_none"]:
            raise ValueError("which_fix must be fix_none, fix_tensors or fix_bases")

        self.l2_pen = kwargs.get("stab_promoting_pen")
        self.pen_tf = kwargs.get("stab_promoting_tf")
        self.randic = kwargs.get("stab_promoting_ic")

        if self.l2_pen is not None and self.pen_tf is None:
            raise ValueError(
                "If you provide a value for stab_promoting_pen you also have "
                "to provide a value for stab_promoting_tf"
            )

        if self.l2_pen is not None and self.randic is None:
            raise ValueError(
                "If you provide a value for stab_promoting_pen you also have "
                "to provide a random ic vector of the same size as the ROM"
            )

        if self.randic is not None:
            self.randic = self.randic / bkend.vector_norm(self.randic)
            self.randic = self.randic.reshape(-1)

    def _global_to_local_indices(self, global_indices):
        bkend = self.backend
        global_indices = bkend.asarray(global_indices)
        gpu_indices = bkend.asarray(self.pool.traj_indices)

        # Filter to trajectories owned by this pool, then map global IDs to
        # local positions.
        mask = bkend.isin(global_indices, gpu_indices)
        requested_and_owned = global_indices[mask]

        sort_order = bkend.argsort(gpu_indices)
        sorted_gpu = gpu_indices[sort_order]
        positions = bkend.searchsorted(sorted_gpu, requested_and_owned)
        return sort_order[positions]
