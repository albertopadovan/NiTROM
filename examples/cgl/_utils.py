"""Experiment records, data views, and cached evaluation results."""

import json

import numpy as np
from settings import configuration

from nitrom.checkpointing import Checkpoint, load_checkpoint
from nitrom.time_steppers import solve_ivp


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
        2 * settings.nx,
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


def basis_pool(X, time, weights=None):
    """In-memory pool for public basis routines; no derivative cache is needed."""
    from types import SimpleNamespace

    return SimpleNamespace(
        X=X, time=time, weights=weights, N=X.shape[1], world_size=1, rank=0, comm=None
    )


def training_view(X, time, weights, derivatives=None):
    """OpInf needs dX; NiTROM needs only states, times, and output-loss weights."""
    from types import SimpleNamespace

    return SimpleNamespace(
        X=X, time=time, weights=weights, dX=derivatives, forcing_fns=[]
    )


def load_training(directory):
    with np.load(directory / "data" / "training.npz") as data:
        return tuple(data[key] for key in ("X", "dX", "time", "amplitudes"))


def selection_cost(model, projection, X, time, C):
    """Unaveraged trajectory objective used to select OpInf regularization."""
    Z = solve_ivp(
        model.evaluate_rhs,
        projection.encode(X[:, :, 0]),
        time[0],
        time[-1],
        (time[1] - time[0]) / 5,
        time,
        "rk4",
    )
    prediction = np.array([C @ projection.decode(z.T).T for z in Z])
    truth = C @ X
    energy = np.mean(np.sum(truth**2, axis=1), axis=1)
    return float(np.sum(np.sum((truth - prediction) ** 2, axis=(1, 2)) / energy))


def evaluate(directory, models, X, time, C, settings):
    """Cache plotted responses and errors; plotting does no solves."""
    with np.load(directory / "data" / "testing.npz") as data:
        saved = {key: data[key] for key in data.files}
    keep = time <= settings.nitrom_end + 1e-10
    t = time[keep]
    impulses = saved["impulses"][:, :, keep]
    ensemble = saved["ensemble"][:, :, keep]
    tf = saved["time_forced"]
    windows = X[:, :, keep]
    energy = np.sum(windows**2, axis=1)
    output_energy = np.sum((C @ windows) ** 2, axis=1)
    arrays = dict(
        time=t,
        time_forced=tf,
        impulse_truth=C @ impulses,
        force_truth=C @ saved["forced"],
        impulse_amplitudes=saved["impulse_amplitudes"],
        impulse_labels=saved["impulse_labels"],
        force_amplitudes=saved["force_amplitudes"],
        force_frequencies=saved["force_frequencies"],
        x=saved["x"],
        xbar=saved["xbar"],
        training_time=t,
        state_energy=energy,
        output_energy=output_energy,
        training_amplitudes=load_training(directory)[3],
    )
    metrics = {}
    for kind, checkpoint in models.items():
        model, projection = checkpoint.model, checkpoint.projection
        initial = np.concatenate((impulses[:, :, 0], ensemble[:, :, 0]))
        Z = solve_ivp(
            model.evaluate_rhs,
            projection.encode(initial),
            t[0],
            t[-1],
            (t[1] - t[0]) / 5,
            t,
            "rk4",
        )
        responses = np.array([C @ projection.decode(z.T).T for z in Z])
        predictions, ensemble_y = responses[: len(impulses)], responses[len(impulses) :]
        direction = projection.encode(saved["direction"])
        inputs = [
            lambda s, a=amplitude, k=frequency, d=direction: (
                a * np.sin(k * saved["omega"] * s) * d
            )
            for amplitude, frequency in zip(
                saved["force_amplitudes"], saved["force_frequencies"], strict=True
            )
        ]
        Z = solve_ivp(
            model.evaluate_rhs,
            np.zeros((len(inputs), model.state_dimension)),
            tf[0],
            tf[-1],
            (tf[1] - tf[0]) / 5,
            tf,
            "rk4",
            external_forcing=inputs,
        )
        forced_y = np.array([C @ projection.decode(z.T).T for z in Z])
        truth = C @ ensemble
        weights = np.mean(np.sum(truth**2, axis=1), axis=1)
        curve = (np.sum((truth - ensemble_y) ** 2, axis=1) / weights[:, None]).mean(
            axis=0
        )
        arrays[f"impulse_{kind}"] = predictions
        arrays[f"forced_{kind}"] = forced_y
        arrays[f"ensemble_{kind}"] = ensemble_y
        arrays[f"error_{kind}"] = curve
        arrays[f"Phi_{kind}"], arrays[f"Psi_{kind}"] = (
            projection.Phi,
            projection.Psi,
        )
        impulse_errors = []
        for target, prediction in zip(C @ impulses, predictions, strict=True):
            impulse_errors.append(
                float(np.linalg.norm(target - prediction) / np.linalg.norm(target))
            )
        metrics[kind] = dict(
            impulse_relative_errors=impulse_errors,
            mean_output_error=float(curve.mean()),
            final_loss=checkpoint.metadata["final_loss"],
        )
    np.savez(directory / "results" / "predictions.npz", **arrays)
    write_json(directory / "results" / "metrics.json", metrics)


def orthonormalize_psi(Psi, Ar, Hr):
    """QR changes z = Psi.T @ q, so transform the operators into the new coordinates."""
    Psi_o, Rq = np.linalg.qr(Psi)
    T = np.linalg.inv(Rq).T
    Ti = Rq.T
    Ar_new = T @ Ar @ Ti
    # optimize=True: the default walks all r^8 index combinations, which is
    # harmless at r = 5 but becomes expensive as the rank grows.
    Hr_new = np.einsum("ai,ijkl,jb,kc,ld->abcd", T, Hr, Ti, Ti, Ti, optimize=True)
    return Psi_o, Ar_new, Hr_new
