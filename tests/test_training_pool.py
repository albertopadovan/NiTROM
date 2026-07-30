"""TrainingPool multiprocessing auto-detection and trajectory sharding.

The pool takes no ``rank``/``world_size``: it figures out the process group
itself from the active backend's distributed context (``MPI.COMM_WORLD`` under
``mpiexec``, ``torch.distributed`` under ``torchrun``, else single-process), so
a parallel job shards correctly with no extra arguments.  An explicit ``comm``
overrides the auto-detected group.
"""

import numpy as np
import pytest

from nitrom import training_data
from nitrom.backend import set_backend
from nitrom.training_data import TrainingPool

N_TRAJ, N, NT = 4, 5, 7


class _FakeComm:
    """Stand-in for an mpi4py communicator (Get_rank/Get_size API)."""

    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size


class _FakeTorchPG:
    """Stand-in for a torch.distributed process group (rank()/size(), no Get_*)."""

    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")
    yield
    set_backend("torch")


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    for k in range(N_TRAJ):
        np.save(tmp_path / f"traj_{k:03d}.npy", rng.standard_normal((N, NT)))
    np.save(tmp_path / "time.npy", np.linspace(0.0, 1.0, NT))
    return tmp_path


def _pool(data_dir, **kw):
    return TrainingPool(
        n_traj=N_TRAJ,
        fname_traj=str(data_dir / "traj_%03d.npy"),
        fname_time=str(data_dir / "time.npy"),
        dtype=np.float64,
        **kw,
    )


def test_single_process_autodetect(data_dir):
    """No mpiexec: auto-detect yields world_size 1 and loads every trajectory."""
    pool = _pool(data_dir)
    assert (pool.rank, pool.world_size) == (0, 1)
    assert pool.my_n_traj == N_TRAJ
    assert pool.traj_indices == [0, 1, 2, 3]
    assert pool.X.shape == (N_TRAJ, N, NT)


def test_explicit_comm_overrides_autodetect(data_dir, monkeypatch):
    """A passed communicator defines the group and wins over auto-detection."""
    monkeypatch.setattr(training_data, "distributed_rank_size", lambda: (3, 9))
    comm = _FakeComm(1, 2)
    pool = _pool(data_dir, comm=comm)
    assert (pool.rank, pool.world_size) == (1, 2)
    assert pool.traj_indices == [2, 3]
    assert pool.comm is comm  # stored for downstream collectives


def test_torch_process_group_comm(data_dir):
    """A comm exposing the torch process-group API (rank()/size(), no Get_rank)
    is handled too -- not just mpi4py communicators."""
    pool = _pool(data_dir, comm=_FakeTorchPG(1, 2))
    assert (pool.rank, pool.world_size) == (1, 2)
    assert pool.traj_indices == [2, 3]


@pytest.mark.parametrize(
    "rank,size,expected",
    [(0, 2, [0, 1]), (1, 2, [2, 3]), (0, 4, [0]), (3, 4, [3])],
)
def test_autodetected_sharding(data_dir, monkeypatch, rank, size, expected):
    """A detected multi-rank context shards trajectories without any args."""
    monkeypatch.setattr(training_data, "distributed_rank_size", lambda: (rank, size))
    pool = _pool(data_dir)
    assert (pool.rank, pool.world_size) == (rank, size)
    assert pool.traj_indices == expected
    assert pool.my_n_traj == len(expected)


def test_training_data_time_shifting(data_dir):
    """Test that time-shifted windows are correctly extracted, concatenated, and aligned."""
    # times in pool.time: [0.0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333, 1.0]
    shift_start_times = (0.0, 0.3333, 0.6667)
    pool = _pool(data_dir, shift_start_times=shift_start_times)
    td = training_data.TrainingData(
        pool,
        which_trajs=list(range(N_TRAJ)),
        percent_time_length=0.4,
        leggauss_deg=5,
        nsave_rom=15,
    )
    
    # 4 trajectories * 3 shifts = 12 trajectories
    # n_keep = max(1, int(0.4 * 7)) = 2 snapshots
    assert td.X.shape == (12, N, 2)
    assert td.dX.shape == (12, N, 2)
    assert len(td.weights) == 12
    
    # Start times map to closest indices:
    # 0.0 -> index 0 (slice 0:2)
    # 0.3333 -> index 2 (slice 2:4)
    # 0.6667 -> index 4 (slice 4:6)
    # Verification of matching slices:
    np.testing.assert_array_equal(td.X[0], pool.X[0, :, 0:2])
    np.testing.assert_array_equal(td.X[1], pool.X[1, :, 2:4])
    np.testing.assert_array_equal(td.X[2], pool.X[2, :, 4:6])
    np.testing.assert_array_equal(td.X[3], pool.X[3, :, 0:2])


def test_training_pool_sharding_virtual(data_dir, monkeypatch):
    """Test that TrainingPool shards virtual trajectories correctly when world_size > physical n_traj."""
    # Physical N_TRAJ = 4, len(shift_start_times) = 3. Total virtual = 12.
    monkeypatch.setattr(training_data, "distributed_rank_size", lambda: (5, 12))
    pool = _pool(data_dir, shift_start_times=(0.0, 0.33, 0.66))
    assert pool.rank == 5
    assert pool.world_size == 12
    assert pool.traj_indices == [5]
    assert pool.my_n_traj == 1
    # File loaded should be traj_001.npy (physical trajectory 1)
    assert "traj_001.npy" in pool.fnames_traj[0]


