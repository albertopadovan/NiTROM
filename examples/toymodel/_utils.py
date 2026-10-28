"""Experiment records, data views, and cached evaluation results."""

import json
from types import SimpleNamespace

import numpy as np
from settings import configuration

from nitrom.checkpointing import Checkpoint, load_checkpoint


def prepare(directory):
    for name in ("data", "models", "results", "figures"):
        (directory / name).mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resume_checkpoint(directory, kind, settings, fresh=False):
    """Select the current run and check its dimensions before overwriting fits."""
    path = directory / "models" / f"{kind}.npz"
    if fresh or not path.exists():
        return None
    saved = load_checkpoint(path)
    if saved.model.state_dimension != settings.r or saved.projection.Phi.shape != (
        3,
        settings.r,
    ):
        raise ValueError(f"Incompatible checkpoint {path}; use --fresh to restart.")
    print(f"Continuing {kind} from {path}")
    return saved


def checkpoint(
    kind, model, projection, settings, module=None, previous=None, final_loss=None
):
    """Attach this experiment's metadata and append its training histories."""
    history = previous.history if previous else {"loss": [], "gradnorm": []}
    history = {key: list(values) for key, values in history.items()}
    if module is not None:
        history["loss"] += [float(x) for x in getattr(module, "loss_history", [])]
        history["gradnorm"] += [
            float(x) for x in getattr(module, "gradnorm_history", [])
        ]
        if final_loss is None:
            final_loss = float(module())
    metadata = dict(
        kind=kind,
        settings=configuration(settings),
        runs=previous.metadata["runs"] + 1 if previous else 1,
        final_loss=final_loss,
    )
    return Checkpoint(model, projection, history=history, metadata=metadata)


def output_error(truth, prediction, weights):
    """Keep failed predictions as NaNs; score them as zero output, with a count."""
    failed = ~np.all(np.isfinite(prediction), axis=1)
    scored = np.where(failed[:, None, :], 0.0, prediction)
    curve = (np.sum((truth - scored) ** 2, axis=1) / weights[:, None]).mean(axis=0)
    return curve, int(failed.any(axis=1).sum())


def load_training(directory):
    """Array-backed views: POD uses raw energy weights; inference averages the loss."""
    with np.load(directory / "data" / "training.npz") as archive:
        X, dX, time, weights, amplitudes = (
            archive[key] for key in ("X", "dX", "time", "weights", "amplitudes")
        )
    pool = SimpleNamespace(X=X, weights=weights, N=3, world_size=1)
    data = SimpleNamespace(
        X=X,
        dX=dX,
        time=time,
        weights=weights * len(X) * len(time),
        forcing_fns=[lambda t, b=float(b): np.array([b]) for b in amplitudes],
    )
    return pool, data


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
        model, projection = saved.model, saved.projection
        inputs = [lambda t, b=float(b): np.array([b]) for b in amplitudes]
        with np.errstate(over="ignore", invalid="ignore"):
            Z = solve_ivp(
                model.evaluate_rhs,
                np.zeros((len(amplitudes), model.state_dimension)),
                0,
                time[-1],
                (time[1] - time[0]) / 5,
                time,
                "rk4",
                external_forcing=inputs,
            )
            valid = np.isfinite(Z).all(axis=1) & (np.linalg.norm(Z, axis=1) < 1e6)
        valid = np.logical_and.accumulate(valid, axis=1)
        reconstructed = np.array([projection.decode(z.T).T for z in Z])
        prediction = fom.compute_output(reconstructed)
        prediction = np.where(valid[:, None, :], prediction, np.nan)
        curve, failed = output_error(truth, prediction, weights)
        arrays[f"prediction_{kind}"] = prediction
        arrays[f"error_{kind}"] = curve
        metrics[kind] = dict(
            diverged=failed,
            mean_output_error=float(curve.mean()),
            final_loss=saved.metadata["final_loss"],
        )
    np.savez(directory / "results" / "predictions.npz", **arrays)
    write_json(directory / "results" / "metrics.json", metrics)
