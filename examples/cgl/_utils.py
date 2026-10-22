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


def basis_pool(X, time, weights=None):
    """In-memory pool for public basis routines; no derivative cache is needed."""
    from types import SimpleNamespace

    return SimpleNamespace(
        X=X, time=time, weights=weights, N=X.shape[1], world_size=1, rank=0, comm=None
    )


def training_view(X, time, weights, derivatives=None):
    """Array-backed training view, sharded by trajectory under MPI.

    OpInf reads dX; NiTROM needs only X, time, weights, and forcing_fns.
    We supply weights explicitly to preserve the benchmark's normalization.
    """
    from types import SimpleNamespace

    rank, size = mpi_rank_size()
    if size > len(X):
        raise ValueError("Use at most one MPI rank per training trajectory.")
    return SimpleNamespace(
        X=X[rank::size],
        time=time,
        weights=weights[rank::size],
        dX=derivatives[rank::size] if derivatives is not None else None,
        forcing_fns=[],
    )


def load_training(directory):
    data = directory / "data"
    amplitudes = np.load(data / "train_amplitudes.npy")
    X = np.stack(
        [np.load(data / f"train_traj_{j:03d}.npy") for j in range(len(amplitudes))]
    )
    derivatives = np.stack(
        [np.load(data / f"train_deriv_{j:03d}.npy") for j in range(len(amplitudes))]
    )
    return X, derivatives, np.load(data / "time.npy"), amplitudes


def rollout(model, projection, time, x0, forcing=None):
    """DOP853 evaluation with an explicit blow-up event and NaNs after failure."""
    from scipy.integrate import solve_ivp

    z0 = projection.encode(x0)
    limit = 1e6 * max(np.linalg.norm(z0), 1.0)

    def event(t, z):
        return np.linalg.norm(z) - limit

    event.terminal = True

    def rhs(t, z):
        value = model.evaluate_rhs(t, z)
        return value if forcing is None else value + projection.encode(forcing(t))

    Z = np.full((model.state_dimension, len(time)), np.nan)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = solve_ivp(
                rhs,
                (time[0], time[-1]),
                z0,
                t_eval=time,
                method="DOP853",
                rtol=1e-9,
                atol=1e-12,
                events=event,
            )
        Z[:, : result.y.shape[1]] = result.y
    except (ValueError, FloatingPointError, OverflowError):
        Z[:, 0] = z0
    return projection.decode(Z.T).T


def selection_cost(model, projection, X, time, C):
    """Unaveraged trajectory objective used to select OpInf regularization."""
    cost = 0.0
    for trajectory in X:
        prediction = C @ rollout(model, projection, time, trajectory[:, 0])
        if not np.isfinite(prediction).all():
            return np.inf
        truth = C @ trajectory
        cost += np.sum((truth - prediction) ** 2) / np.mean(np.sum(truth**2, axis=0))
    return float(cost)


def evaluate(directory, models, X, time, C, settings):
    """Cache every plotted response and its failures; plotting does no solves."""
    with np.load(directory / "data" / "testing.npz") as data:
        saved = {key: data[key] for key in data.files}
    keep = time <= settings.nitrom_end + 1e-10
    t = time[keep]
    impulses = saved["impulses"][:, :, keep]
    ensemble = saved["ensemble"][:, :, keep]
    tf, forced = saved["time_forced"], saved["forced"]
    windows, length = checkpoint_windows(
        X[:, :, keep], settings.nitrom_checkpoints, settings.nitrom_stride
    )
    energy = np.sum(windows**2, axis=1)
    output_energy = np.sum((C @ windows) ** 2, axis=1)
    arrays = dict(
        time=t,
        time_forced=tf,
        impulse_truth=C @ impulses,
        force_truth=C @ forced,
        impulse_amplitudes=saved["impulse_amplitudes"],
        impulse_labels=saved["impulse_labels"],
        force_amplitudes=saved["force_amplitudes"],
        force_frequencies=saved["force_frequencies"],
        x=saved["x"],
        xbar=saved["xbar"],
        training_time=t[:length],
        state_energy=energy,
        output_energy=output_energy,
        training_amplitudes=np.repeat(
            np.load(directory / "data" / "train_amplitudes.npy"), len(windows) // len(X)
        ),
    )
    metrics = {}
    for kind, checkpoint in models.items():
        model, projection = restore(checkpoint)
        predictions = np.array(
            [C @ rollout(model, projection, t, q[:, 0]) for q in impulses]
        )
        ensemble_y = np.array(
            [C @ rollout(model, projection, t, q[:, 0]) for q in ensemble]
        )
        forced_y = []
        for amplitude, frequency in zip(
            saved["force_amplitudes"], saved["force_frequencies"], strict=True
        ):

            def forcing(s, a=amplitude, k=frequency):
                return a * np.sin(k * saved["omega"] * s) * saved["direction"]

            forced_y.append(
                C @ rollout(model, projection, tf, np.zeros(X.shape[1]), forcing)
            )
        forced_y = np.array(forced_y)
        truth = C @ ensemble
        weights = np.mean(np.sum(truth**2, axis=1), axis=1)
        curve, failed = output_error(truth, ensemble_y, weights)
        arrays[f"impulse_{kind}"] = predictions
        arrays[f"forced_{kind}"] = forced_y
        arrays[f"ensemble_{kind}"] = ensemble_y
        arrays[f"error_{kind}"] = curve
        arrays[f"Phi_{kind}"], arrays[f"Psi_{kind}"] = (
            checkpoint["Phi"],
            checkpoint["Psi"],
        )
        impulse_errors = []
        for target, prediction in zip(C @ impulses, predictions, strict=True):
            impulse_errors.append(
                float(np.linalg.norm(target - prediction) / np.linalg.norm(target))
                if np.isfinite(prediction).all()
                else None
            )
        metrics[kind] = dict(
            ensemble_diverged=failed,
            impulse_relative_errors=impulse_errors,
            impulse_diverged=int((~np.isfinite(predictions).all(axis=(1, 2))).sum()),
            forced_diverged=int((~np.isfinite(forced_y).all(axis=(1, 2))).sum()),
            mean_output_error=float(curve.mean()),
            final_loss=checkpoint["final_loss"],
        )
    np.savez(directory / "results" / "predictions.npz", **arrays)
    write_json(directory / "results" / "metrics.json", metrics)


def orthonormalize_psi(Phi, Psi, Ar, Hr):
    r"""
    Equivalent model whose test basis has orthonormal columns.

    riemannian_optimize projects the starting point onto its manifold with a QR,
    so a Stiefel-constrained Psi is orthonormalized before the first step.  That
    is not a harmless re-scaling: z = Psi^T q, so it changes the latent
    coordinates and would invalidate A_r and H_r unless they move with it.

    Writing the QR as Psi = Psi_o R, the new coordinates are z' = T z with
    T = R^-T, so A' = T A T^-1 and H'(z',z',z') = T H(T^-1 z', ...).  Phi is left
    alone: the decode Phi (Psi^T Phi)^-1 is invariant under Phi -> Phi Q, so
    Grassmann's own QR costs nothing and needs no correction.

    :returns: ``(Psi_o, Ar_new, Hr_new)`` -- the same input-output model
    """
    Psi_o, Rq = np.linalg.qr(Psi)
    T = np.linalg.inv(Rq).T
    Ti = np.linalg.inv(T)
    Ar_new = T @ Ar @ Ti
    # optimize=True: the default walks all r^8 index combinations, which is
    # harmless at r = 5 but becomes expensive as the rank grows.
    Hr_new = np.einsum("ai,ijkl,jb,kc,ld->abcd", T, Hr, Ti, Ti, Ti, optimize=True)
    return Psi_o, Ar_new, Hr_new


def checkpoint_windows(X, m, q):
    """
    Sliding equal-length windows starting at snapshots 0, q, ..., (m-1)q.

    The CGL is autonomous, so the state at snapshot k*q is a legitimate initial
    condition at t = 0 and the window that follows it is a trajectory in its own
    right.  All windows are given the same length L = n_snap - (m-1)q, so a
    single time grid serves them all; checkpoints that would run past the end
    are dropped.  This is the same construction compute_data_driven_balancing
    uses to enlarge the span of X_0, at zero extra simulation cost.

    :returns: ``(windows, L)`` with windows of shape ``(n_traj*m, n, L)``,
        ordered trajectory-major so that MPI's strided shard mixes parents.
    """
    n_snap = X.shape[2]
    starts = [k * q for k in range(m) if k * q < n_snap]
    L = n_snap - starts[-1]
    W = np.stack([Xj[:, s : s + L] for Xj in X for s in starts])
    return W, L
