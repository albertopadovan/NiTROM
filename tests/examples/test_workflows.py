"""Run the documented tutorials from outside the checkout, without LaTeX."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from nitrom import LinearProjection, PolynomialModel, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
KINDS = {
    "toymodel": ("galerkin", "opinf", "gas_opinf", "nitrom", "gas_nitrom"),
    "cgl": ("balanced_opinf", "pod_opinf", "nitrom_lall", "nitrom_pod"),
}


def run(case, script, directory, *options):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(directory.parent / "mpl-cache"),
        "XDG_CACHE_HOME": str(directory.parent / "cache"),
    }
    start = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / case / script),
            "--smoke",
            "--output-dir",
            str(directory),
            *options,
        ],
        cwd=directory.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return time.perf_counter() - start


@pytest.mark.parametrize("case", KINDS)
def test_complete_workflow_and_checkpoint_lifecycle(case, tmp_path):
    directory = tmp_path / case
    elapsed = {"generate": run(case, "generate_data.py", directory)}
    elapsed["train"] = run(case, "train_models.py", directory, "--fresh")
    elapsed["plot"] = run(case, "plot_results.py", directory)
    print(f"{case} smoke seconds: {elapsed}")
    saved = {
        kind: load_checkpoint(directory / "models" / f"{kind}.npz")
        for kind in KINDS[case]
    }
    for checkpoint in saved.values():
        assert checkpoint.metadata["settings"]["profile"] == "smoke"
        assert checkpoint.projection.Phi.shape[1] == checkpoint.model.state_dimension
        assert all(np.isfinite(x).all() for x in checkpoint.model.inner_params())
        assert checkpoint.metadata["final_loss"] is None or np.isfinite(
            checkpoint.metadata["final_loss"]
        )
        assert all(
            np.isfinite(checkpoint.history[key]).all() for key in ("loss", "gradnorm")
        )
    figures = list((directory / "figures").iterdir())
    assert len(figures) == (1 if case == "toymodel" else 5)
    assert all(path.stat().st_size > 1000 for path in figures)
    metrics = json.loads((directory / "results" / "metrics.json").read_text())
    assert set(metrics) == set(KINDS[case])
    with np.load(directory / "results" / "predictions.npz") as result:
        for kind in KINDS[case]:
            assert np.isfinite(result[f"error_{kind}"]).all()
            prediction = result[
                f"prediction_{kind}" if case == "toymodel" else f"impulse_{kind}"
            ]
            assert np.isfinite(prediction).any()
            if case == "toymodel":
                failed = (~np.isfinite(prediction).all(axis=(1, 2))).sum()
                assert metrics[kind]["diverged"] == failed
            else:
                assert np.isfinite(prediction).all()
                assert not any("diverged" in key for key in metrics[kind])

    # Changes to training settings metadata do not block parameter continuation.
    # Only dimensions are checked; users choose --fresh after changing the problem.
    for kind, checkpoint in saved.items():
        checkpoint.metadata["settings"]["nitrom_epochs"] += 1
        save_checkpoint(directory / "models" / f"{kind}.npz", checkpoint)
    # Automatically warm start only the NiTROM refinements, retaining histories.
    run(case, "train_models.py", directory)
    refinements = [
        kind
        for kind in KINDS[case]
        if kind.startswith("nitrom") or kind == "gas_nitrom"
    ]
    for kind in refinements:
        continued = load_checkpoint(directory / "models" / f"{kind}.npz")
        assert continued.metadata["runs"] == 2
        assert len(continued.history["loss"]) > len(saved[kind].history["loss"])
        np.testing.assert_allclose(
            continued.metadata["initial_loss"],
            saved[kind].metadata["final_loss"],
            rtol=1e-9,
            atol=1e-12,
        )

    # A dimension mismatch must fail before overwriting fits.
    path = directory / "models" / f"{refinements[0]}.npz"
    incompatible = load_checkpoint(path)
    r = incompatible.model.state_dimension + 1
    n = incompatible.projection.ambient_space_dimension
    incompatible.model = PolynomialModel(r, [1, 2])
    incompatible.projection = LinearProjection([np.eye(n, r), np.eye(n, r)])
    save_checkpoint(path, incompatible)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / case / "train_models.py"),
            "--smoke",
            "--output-dir",
            str(directory),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Incompatible checkpoint" in result.stderr and "--fresh" in result.stderr

    # --fresh bypasses the incompatible dimensions and reproduces the original fit.
    run(case, "train_models.py", directory, "--fresh")
    for kind in refinements:
        fresh = load_checkpoint(directory / "models" / f"{kind}.npz")
        assert fresh.metadata["runs"] == 1
        for expected, actual in zip(
            saved[kind].model.inner_params(), fresh.model.inner_params(), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", KINDS)
def test_import_safety_and_profile_paths(case, tmp_path):
    code = """
import pathlib
import runpy
import sys
from nitrom.backend import get_backend
folder = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(folder))
previous = get_backend()
for path in folder.glob("*.py"):
    runpy.run_path(str(path), run_name="import_check")
assert get_backend() is previous
from settings import arguments
_, full, _ = arguments("test", [])
_, smoke, _ = arguments("test", ["--smoke"])
assert full == folder / "outputs" / "full"
assert smoke == folder / "outputs" / "smoke"
assert full != smoke
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLES / case)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.distributed
@pytest.mark.parametrize("case", KINDS)
def test_mpi_workflow(case, tmp_path):
    import shutil

    pytest.importorskip("mpi4py")
    launcher = shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("mpiexec is unavailable")
    directory = tmp_path / case
    run(case, "generate_data.py", directory)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    command = [
        launcher,
        "-n",
        "2",
        sys.executable,
        str(EXAMPLES / f"{case}_mpi" / "train_models.py"),
        "--smoke",
        "--output-dir",
        str(directory),
    ]
    result = subprocess.run(
        [*command, "--fresh"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    distributed = {
        kind: load_checkpoint(directory / "models" / f"{kind}.npz")
        for kind in KINDS[case]
    }
    assert (directory / "results" / "metrics.json").is_file()

    # Starting from the same data, serial and MPI optimize the same global loss.
    serial_dir = tmp_path / f"{case}_serial"
    shutil.copytree(directory / "data", serial_dir / "data")
    run(case, "train_models.py", serial_dir, "--fresh")
    for kind, saved in distributed.items():
        serial = load_checkpoint(serial_dir / "models" / f"{kind}.npz")
        if saved.metadata["final_loss"] is not None:
            np.testing.assert_allclose(
                saved.metadata["final_loss"],
                serial.metadata["final_loss"],
                rtol=1e-7,
                atol=1e-10,
            )
        for expected, actual in zip(
            serial.model.inner_params(), saved.model.inner_params(), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    # No resume flag: automatically continue each method's current checkpoint.
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    refinements = [
        kind
        for kind in KINDS[case]
        if kind.startswith("nitrom") or kind == "gas_nitrom"
    ]
    for kind in refinements:
        continued = load_checkpoint(directory / "models" / f"{kind}.npz")
        assert continued.metadata["runs"] == 2
        np.testing.assert_allclose(
            continued.metadata["initial_loss"],
            distributed[kind].metadata["final_loss"],
            rtol=1e-9,
            atol=1e-12,
        )

    # MPI checkpoints also resume directly in the single-process workflow.
    run(case, "train_models.py", directory)
    for kind in refinements:
        assert (
            load_checkpoint(directory / "models" / f"{kind}.npz").metadata["runs"] == 3
        )
