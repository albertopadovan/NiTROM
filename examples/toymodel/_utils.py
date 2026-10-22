"""Local file and checkpoint helpers; training stays in train_models.py."""

import hashlib
import json
import pickle
import shutil

import numpy as np
from settings import configuration

from nitrom.distributed import mpi_allreduce_scalar, mpi_rank_size
from nitrom.latent_space_models import PolynomialModel
from nitrom.projections import LinearProjection


def prepare(directory):
    for name in ("data", "models", "results", "figures"):
        (directory / name).mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def data_signature(directory):
    """Identify the actual training files, not just their paths."""
    digest = hashlib.sha256()
    paths = sorted((directory / "data").glob("train_*"))
    paths += [directory / "data" / "time.npy", directory / "data" / "settings.json"]
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_data_settings(directory, settings):
    path = directory / "data" / "settings.json"
    if not path.exists():
        raise FileNotFoundError(f"Run generate_data.py first: missing {path}")
    saved = json.loads(path.read_text())
    # Training controls can change without regenerating physical trajectories.
    current = configuration(settings)
    data_keys = [
        key
        for key in current
        if key not in ("r", "nitrom_end")
        and not any(
            word in key
            for word in (
                "epochs",
                "adjoint",
                "substeps",
                "opinf_reg",
                "checkpoints",
                "stride",
            )
        )
    ]
    if any(saved.get(key) != current[key] for key in data_keys):
        raise ValueError(
            "Data settings differ; regenerate data with the same profile/settings."
        )


def read_checkpoint(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def resume_checkpoint(directory, kind, settings, signature, fresh=False):
    path = directory / "models" / f"{kind}.pkl"
    if fresh or not path.exists():
        return None
    checkpoint = read_checkpoint(path)
    if (
        checkpoint.get("settings") != configuration(settings)
        or checkpoint.get("data_signature") != signature
        or checkpoint.get("kind") != kind
        or checkpoint.get("r") != settings.r
    ):
        raise ValueError(f"Incompatible checkpoint {path}; use --fresh to restart.")
    return checkpoint


def save_checkpoint(directory, checkpoint):
    if mpi_rank_size()[0] != 0:
        return
    path = directory / "models" / f"{checkpoint['kind']}.pkl"
    if path.exists():
        shutil.copyfile(path, path.with_suffix(".pkl.prev"))
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(checkpoint, stream)
    temporary.replace(path)


def checkpoint(
    kind,
    model,
    projection,
    settings,
    signature,
    module=None,
    previous=None,
    forcing_config=None,
    training_seconds=0.0,
):
    history = previous["history"] if previous else {"loss": [], "gradnorm": []}
    history = {key: list(values) for key, values in history.items()}
    if module is not None:
        history["loss"] += [float(x) for x in getattr(module, "loss_history", [])]
        history["gradnorm"] += [
            float(x) for x in getattr(module, "gradnorm_history", [])
        ]
    return dict(
        kind=kind,
        r=settings.r,
        poly_comp=list(model.poly_comp),
        Phi=np.array(projection.Phi),
        Psi=np.array(projection.Psi),
        tensors=[np.array(x) for x in model.inner_params()],
        gas_params=[np.array(x) for x in model.get_params()]
        if hasattr(model, "model")
        else None,
        forcing_config=forcing_config,
        settings=configuration(settings),
        data_signature=signature,
        runs=previous["runs"] + 1 if previous else 1,
        history=history,
        final_loss=global_loss(module) if module is not None else None,
        training_seconds=training_seconds
        + (previous["training_seconds"] if previous else 0.0),
    )


def restore(checkpoint, *, instability_threshold=np.inf):
    model = PolynomialModel(
        checkpoint["r"],
        checkpoint["poly_comp"],
        tensors=[x.copy() for x in checkpoint["tensors"]],
        forcing_config=checkpoint["forcing_config"],
        instability_threshold=instability_threshold,
    )
    projection = LinearProjection([checkpoint["Phi"].copy(), checkpoint["Psi"].copy()])
    return model, projection


def global_loss(module):
    cost = float(module())
    if mpi_rank_size()[1] > 1:
        cost = mpi_allreduce_scalar(cost, comm=module.comm)
    if not np.isfinite(cost):
        raise FloatingPointError("Training produced a nonfinite loss.")
    return cost


def plot_style(tex):
    from matplotlib import pyplot as plt

    from nitrom.plotting import set_plot_style

    set_plot_style()
    plt.rcParams["text.usetex"] = tex


def output_error(truth, prediction, weights):
    """Keep failed predictions as NaNs; score them as zero output, with a count."""
    failed = ~np.all(np.isfinite(prediction), axis=1)
    scored = np.where(failed[:, None, :], 0.0, prediction)
    curve = (np.sum((truth - scored) ** 2, axis=1) / weights[:, None]).mean(axis=0)
    return curve, int(failed.any(axis=1).sum())


def load_training(directory):
    from nitrom.training_data import TrainingData, TrainingPool

    data = directory / "data"
    pool = TrainingPool(
        4,
        str(data / "train_traj_%03d.npy"),
        str(data / "time.npy"),
        dtype=np.float64,
        fname_weights=str(data / "train_weight_%03d.npy"),
        fname_derivs=str(data / "train_deriv_%03d.npy"),
        fname_forcing=str(data / "train_forcing_%03d.pkl"),
    )
    return pool, TrainingData(pool, list(range(4)))


def evaluate(directory, models, fom):
    """Evaluate once; plotting reads these cached predictions and error curves."""
    from nitrom.time_steppers import solve_ivp

    with np.load(directory / "data" / "testing.npz") as data:
        time, X, amplitudes, weights = (
            data[key] for key in ("time", "X", "amplitudes", "weights")
        )
    truth = fom.compute_output(X)
    arrays = dict(time=time, truth=truth)
    metrics = {}
    for kind, saved in models.items():
        model, projection = restore(saved)
        inputs = [lambda t, b=float(b): np.array([b]) for b in amplitudes]
        with np.errstate(over="ignore", invalid="ignore"):
            Z = solve_ivp(
                model.evaluate_rhs,
                np.zeros((len(amplitudes), saved["r"])),
                0,
                time[-1],
                (time[1] - time[0]) / 5,
                time,
                "rk4",
                external_forcing=inputs,
            )
            valid = np.isfinite(Z).all(axis=1) & (np.linalg.norm(Z, axis=1) < 1e6)
        valid = np.logical_and.accumulate(valid, axis=1)
        decoded = projection.decode(Z.transpose(0, 2, 1).reshape(-1, saved["r"]))
        reconstructed = decoded.reshape(len(amplitudes), len(time), 3).transpose(
            0, 2, 1
        )
        prediction = fom.compute_output(reconstructed)
        prediction = np.where(valid[:, None, :], prediction, np.nan)
        curve, failed = output_error(truth, prediction, weights)
        arrays[f"prediction_{kind}"] = prediction
        arrays[f"error_{kind}"] = curve
        metrics[kind] = dict(
            diverged=failed,
            mean_output_error=float(curve.mean()),
            final_loss=saved["final_loss"],
        )
    np.savez(directory / "results" / "predictions.npz", **arrays)
    write_json(directory / "results" / "metrics.json", metrics)
