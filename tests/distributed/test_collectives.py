"""Opt-in checks of real MPI and Torch subgroups."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize("mode", ["numpy", "torch"])
def test_split_communicators(mode, tmp_path):
    root = Path(__file__).resolve().parents[2]
    worker = Path(__file__).with_name("worker.py")
    if mode == "numpy":
        pytest.importorskip("mpi4py")
        launcher = shutil.which("mpiexec")
        if launcher is None:
            pytest.skip("mpiexec is unavailable")
        command = [
            launcher,
            "-n",
            "4",
            sys.executable,
            str(worker),
            mode,
            str(tmp_path),
        ]
    else:
        pytest.importorskip("torch")
        command = [
            sys.executable,
            str(worker),
            mode,
            str(tmp_path),
        ]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"Distributed workers timed out:\n{stdout}\n{stderr}")
        assert process.returncode == 0, stdout + stderr
