"""Centered data views, experiment records, and cached rollout diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
from model import input_functions
from settings import configuration

from nitrom.checkpointing import Checkpoint, load_checkpoint
from nitrom.latent_space_models import AtrPolynomialModel
from nitrom.time_steppers import solve_ivp


def prepare(directory):
    for name in ("data", "models", "results", "figures"):
        (directory / name).mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_training(directory):
    with np.load(directory / "data" / "training.npz") as data:
        return tuple(data[key] for key in ("X", "dX", "time", "amplitudes", "mean"))


def training_view(X, dX, time, amplitudes):
    """Average normalized squared error over trajectories and saved times."""
    energy = np.mean(np.sum(X**2, axis=1), axis=1)
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError(
            "Training trajectories need finite, positive fluctuation energy."
        )
    return SimpleNamespace(
        X=X,
        dX=dX,
        time=time,
        weights=energy * len(X) * len(time),
        forcing_fns=input_functions(amplitudes),
    )


def basis_pool(X):
    return SimpleNamespace(X=X, N=X.shape[1], world_size=1)


def resume_checkpoint(directory, kind, settings, mean, fresh=False):
    path = directory / "models" / f"{kind}.npz"
    if fresh or not path.exists():
        return None
    saved = load_checkpoint(path)
    if (
        saved.model.state_dimension != settings.r
        or saved.projection.Phi.shape != (settings.n, settings.r)
        or not np.array_equal(saved.metadata.get("mean"), mean)
    ):
        raise ValueError(f"Incompatible checkpoint {path}; use --fresh to restart.")
    print(f"Continuing {kind} from {path}")
    return saved


def checkpoint(kind, model, projection, settings, mean, module, previous=None):
    history = previous.history if previous else {"loss": [], "gradnorm": []}
    history = {key: list(values) for key, values in history.items()}
    for key, attribute in (("loss", "loss_history"), ("gradnorm", "gradnorm_history")):
        history[key] += [float(x) for x in getattr(module, attribute, [])]
    return Checkpoint(
        model,
        projection,
        history=history,
        metadata=dict(
            kind=kind,
            settings=configuration(settings),
            mean=mean.copy(),
            runs=previous.metadata["runs"] + 1 if previous else 1,
            final_loss=float(module()),
        ),
    )


def rollout(saved, initial, mean, time, inputs, dt):
    """After the first failed sample, keep the trajectory explicitly unavailable."""
    model, projection = saved.model, saved.projection
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Z = solve_ivp(
            model.evaluate_rhs,
            projection.encode(initial - mean),
            time[0],
            time[-1],
            dt,
            time,
            "rk4",
            external_forcing=inputs,
        )
        valid = np.isfinite(Z).all(axis=1) & (np.linalg.norm(Z, axis=1) < 1e6)
        valid = np.logical_and.accumulate(valid, axis=1)
        Z = np.where(valid[:, None, :], Z, np.nan)
        prediction = (
            np.array([projection.decode(z.T).T for z in Z]) + mean[None, :, None]
        )
        valid &= np.isfinite(prediction).all(axis=1)
        valid = np.logical_and.accumulate(valid, axis=1)
    return np.where(valid[:, None, :], Z, np.nan), np.where(
        valid[:, None, :], prediction, np.nan
    )


def relative_error(truth, prediction, mean):
    energy = np.mean(np.sum((truth - mean[None, :, None]) ** 2, axis=1), axis=1)
    return np.sqrt(np.sum((truth - prediction) ** 2, axis=1) / energy[:, None])


def state_statistics(X, edges, max_lag):
    """Pooled site marginal and site-averaged temporal autocorrelation."""
    if not np.isfinite(X).all():
        return None
    mean = X.mean(axis=1)
    centered = X - mean[:, None]
    variance = np.mean(centered**2, axis=1)
    counts, _ = np.histogram(X, bins=edges)
    density = counts / (X.size * np.diff(edges))
    denominator = np.mean(centered**2)
    acf = np.full(max_lag + 1, np.nan)
    if denominator > 0:
        acf[0] = 1.0
        for lag in range(1, max_lag + 1):
            acf[lag] = np.mean(centered[:, lag:] * centered[:, :-lag]) / denominator
    return dict(
        mean=mean,
        variance=variance,
        density=density,
        autocorrelation=acf,
        histogram_outside_fraction=float(1.0 - counts.sum() / X.size),
    )


def atr_diagnostics(model, Z, time, inputs, input_bound):
    """A common trapping ellipsoid valid for every scalar input |u| <= bound."""
    Qi, Ri = np.linalg.inv(model.Q), np.linalg.inv(model.R)
    Q, R = Qi @ Qi.T, Ri @ Ri.T
    eig_Q, eig_R = np.linalg.eigvalsh(Q), np.linalg.eigvalsh(R)
    effective = max(
        np.linalg.norm(model.Bhat + sign * input_bound * model.B[:, 0])
        for sign in (-1, 1)
    )
    rho = effective / (eig_Q[0] * eig_R[0])
    level = 0.5 * eig_Q[-1] * rho**2
    shifted = Z - model.m[None, :, None]
    y = Q @ shifted
    V = 0.5 * np.sum(shifted * y, axis=1)
    u = np.array([[fn(t)[0] for t in time] for fn in inputs])
    dissipation = np.sum(y * (R @ y), axis=1)
    supply = np.sum(
        y * (model.Bhat[None, :, None] + model.B[None, :, 0, None] * u[:, None, :]),
        axis=1,
    )
    derivative = -dissipation + supply
    valid = np.isfinite(V) & np.isfinite(derivative)
    outside = valid & (np.linalg.norm(shifted, axis=1) > rho * (1 + 1e-10))
    violations = outside & (
        derivative > 1e-9 * (1 + np.abs(dissipation) + np.abs(supply))
    )
    metadata = dict(
        input_bound=input_bound,
        effective_forcing_bound=float(effective),
        decrease_radius=float(rho),
        containing_radius=float(rho * np.sqrt(eig_Q[-1] / eig_Q[0])),
        ellipsoid_level=float(level),
        ellipsoid_metric=Q.tolist(),
        shift=model.m.tolist(),
        valid_samples=int(valid.sum()),
        outside_samples=int(outside.sum()),
        derivative_violations=int(violations.sum()),
        ellipsoid_occupancy=float(((level >= V) & valid).sum() / valid.sum())
        if valid.any()
        else None,
        max_derivative_outside=float(derivative[outside].max())
        if outside.any()
        else None,
    )
    return metadata, V, derivative


def evaluate(directory, models, settings, mean):
    """Cache physical predictions and statistics; plotting performs no solves."""
    with np.load(directory / "data" / "testing.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    time, long_time = data["time"], data["long_time"]
    truth, long_truth = data["X"], data["long_X"]
    inputs = input_functions(
        data["amplitudes"], sinusoidal=data["sinusoidal"], period=float(data["period"])
    )
    if any(
        abs(fn(t)[0]) > settings.input_bound + 1e-12 for fn in inputs for t in long_time
    ):
        raise ValueError("Test inputs exceed the bound used for ATR diagnostics.")
    keep = long_time >= settings.statistics_start - 1e-10
    constants = np.flatnonzero(~data["sinusoidal"])
    stationary_truth = long_truth[constants][:, :, keep]
    lower, upper = stationary_truth.min(), stationary_truth.max()
    padding = max(0.1 * (upper - lower), 1e-6)
    edges = np.linspace(lower - padding, upper + padding, 65)
    max_lag = min(round(2.0 / settings.sample_dt), stationary_truth.shape[-1] - 1)
    reference_stats = [state_statistics(X, edges, max_lag) for X in stationary_truth]
    statistic_names = ("mean", "variance", "density", "autocorrelation")
    arrays = dict(
        time=time,
        long_time=long_time,
        truth=truth,
        long_truth=long_truth,
        labels=data["labels"],
        constant_indices=constants,
        density_edges=edges,
        lags=np.arange(max_lag + 1) * settings.sample_dt,
        statistics_start=settings.statistics_start,
        mean=mean,
    )
    for name in statistic_names:
        arrays[f"{name}_truth"] = np.array([s[name] for s in reference_stats])
    metrics = {}
    for kind, saved in models.items():
        _, prediction = rollout(
            saved,
            truth[:, :, 0],
            mean,
            time,
            inputs,
            settings.sample_dt / settings.n_substeps,
        )
        Z, long_prediction = rollout(
            saved,
            long_truth[:, :, 0],
            mean,
            long_time,
            inputs,
            settings.sample_dt / settings.n_substeps,
        )
        centered = truth - mean[None, :, None]
        projected = (
            np.array(
                [
                    saved.projection.decode(saved.projection.encode(X.T)).T
                    for X in centered
                ]
            )
            + mean[None, :, None]
        )
        error = relative_error(truth, prediction, mean)
        projection_error = relative_error(truth, projected, mean)
        arrays[f"prediction_{kind}"] = prediction
        arrays[f"long_prediction_{kind}"] = long_prediction
        arrays[f"error_{kind}"] = error
        arrays[f"projection_error_{kind}"] = projection_error
        short_failed = ~np.isfinite(prediction).all(axis=(1, 2))
        long_failed = ~np.isfinite(long_prediction).all(axis=(1, 2))
        method = dict(
            prediction_failures=int(short_failed.sum()),
            long_failures=int(long_failed.sum()),
            final_loss=saved.metadata["final_loss"],
            relative_rmse=[
                float(np.sqrt(np.mean(e**2))) if np.isfinite(e).all() else None
                for e in error
            ],
            projection_relative_rmse=np.sqrt(
                np.mean(projection_error**2, axis=1)
            ).tolist(),
            short_failed=short_failed.tolist(),
            long_failed=long_failed.tolist(),
        )
        stats = [
            state_statistics(X, edges, max_lag)
            for X in long_prediction[constants][:, :, keep]
        ]
        method["constant_statistics"] = []
        for i, stat in enumerate(stats):
            if stat is None:
                method["constant_statistics"].append(None)
                continue
            ref = reference_stats[i]
            acf_error = np.linalg.norm(stat["autocorrelation"] - ref["autocorrelation"])
            method["constant_statistics"].append(
                dict(
                    mean_rmse=float(
                        np.sqrt(np.mean((stat["mean"] - ref["mean"]) ** 2))
                    ),
                    variance_rmse=float(
                        np.sqrt(np.mean((stat["variance"] - ref["variance"]) ** 2))
                    ),
                    density_l1=float(
                        np.sum(
                            np.abs(stat["density"] - ref["density"]) * np.diff(edges)
                        )
                    ),
                    autocorrelation_l2=float(acf_error)
                    if np.isfinite(acf_error)
                    else None,
                    histogram_outside_fraction=stat["histogram_outside_fraction"],
                )
            )
        for name in statistic_names:
            arrays[f"{name}_{kind}"] = np.array(
                [
                    stat[name]
                    if stat is not None
                    else np.full_like(reference_stats[i][name], np.nan)
                    for i, stat in enumerate(stats)
                ]
            )
        if isinstance(saved.model, AtrPolynomialModel):
            bound, V, derivative = atr_diagnostics(
                saved.model, Z, long_time, inputs, settings.input_bound
            )
            method["atr"] = bound
            arrays[f"lyapunov_{kind}"], arrays[f"lyapunov_derivative_{kind}"] = (
                V,
                derivative,
            )
        metrics[kind] = method
    np.savez(directory / "results" / "predictions.npz", **arrays)
    write_json(directory / "results" / "metrics.json", metrics)
