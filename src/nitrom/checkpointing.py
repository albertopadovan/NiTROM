"""Portable model/projection checkpoints, independent of experiment layout.

Archives contain numerical arrays and a versioned JSON manifest; they do not
pickle live objects, data, forcing callables, or optimizer state. Built-in
components are reconstructed with their constructor configuration and named
parameters. Register an adapter to support a user-defined component type.
"""

import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .backend import get_backend
from .latent_space_models import (
    AtrPolynomialModel,
    GasPolynomialModel,
    PolynomialModel,
)
from .projections import LinearProjection, PolynomialProjection
from .roms import ParamRegistry

__all__ = [
    "Checkpoint",
    "load_checkpoint",
    "register_checkpoint_type",
    "save_checkpoint",
]

_FORMAT_VERSION = 1
_TYPES: dict[str, tuple[type, Callable, Callable]] = {}


@dataclass
class Checkpoint:
    """Components and optional experiment records returned by load_checkpoint.

    ``history`` and ``metadata`` may contain string-keyed dictionaries, lists,
    tuples, real scalars, and numerical arrays. They remain ordinary Python and
    NumPy values on load, independently of the components' requested backend.
    An in-memory record holds references; :func:`save_checkpoint` snapshots the
    values when called.
    """

    model: Any
    projection: Any = None
    history: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def register_checkpoint_type(name: str, cls: type, *, dump: Callable, load: Callable):
    """Register an exact component type under a stable archive identifier.

    ``dump(component)`` returns state using the value types supported by
    :class:`Checkpoint`. ``load(state, *, backend, device, dtype)`` constructs
    the component, rebuilding any caches. Its ``dtype`` is the caller's optional
    override; without one, the adapter should preserve the saved dtype. Arrays
    in ``state`` arrive as NumPy arrays. Register the adapter in every process
    that saves or loads the custom type. Existing names/types cannot be replaced.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Checkpoint type name must be a nonempty string.")
    if not isinstance(cls, type) or not callable(dump) or not callable(load):
        raise TypeError("Expected a component class and callable dump/load adapters.")
    if name in _TYPES or any(entry[0] is cls for entry in _TYPES.values()):
        raise ValueError(f"Checkpoint type or name is already registered: {name!r}.")
    _TYPES[name] = (cls, dump, load)


def _pack(value, arrays):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"kind": "float", "value": str(value)}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Checkpoint dictionaries require string keys.")
        return {
            "kind": "dict",
            "items": {k: _pack(v, arrays) for k, v in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "items": [_pack(v, arrays) for v in value],
        }
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        key = f"array_{len(arrays)}"
        arrays[key] = value.copy()
        return {"kind": "array", "key": key}
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}.")


def _unpack(value, archive):
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "array":
        return archive[value["key"]].copy()
    if kind == "float":
        return float(value["value"])
    if kind == "dict":
        return {k: _unpack(v, archive) for k, v in value["items"].items()}
    if kind in ("list", "tuple"):
        items = [_unpack(v, archive) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    raise ValueError(f"Unknown checkpoint value encoding: {kind!r}.")


def _dump_component(component):
    if component is None:
        return None
    for name, (cls, dump, _) in _TYPES.items():
        if type(component) is cls:
            return {"type": name, "state": dump(component)}
    raise TypeError(
        f"Unregistered checkpoint type {type(component).__name__}; "
        "use register_checkpoint_type first."
    )


def _validate_pair(model, projection, *, link=False):
    if projection is None:
        return
    if model.state_dimension != projection.latent_space_dimension:
        raise ValueError("Model and projection latent dimensions do not match.")
    registry = ParamRegistry(model, projection)
    registry.assert_shared_consistent()
    if link and registry.shared_names:
        registry.scatter(registry.values())


def save_checkpoint(path, model, projection=None, *, history=None, metadata=None):
    """Atomically save a model and optional projection to an NPZ archive.

    ``model`` may also be a :class:`Checkpoint`, in which case the other
    arguments must be omitted. ``path`` is used exactly as supplied; no suffix
    is appended. Parent directories are created if needed. Component backends
    and latent dimensions must agree, as must shared parameter values.

    The checkpoint restores parameters and records, not optimizer state or
    a training module. Supply data and forcing callables again when training.
    """
    if isinstance(model, Checkpoint):
        if projection is not None or history is not None or metadata is not None:
            raise TypeError(
                "Pass a Checkpoint alone, or separate components and records."
            )
        record = model
    else:
        record = Checkpoint(
            model,
            projection,
            history={} if history is None else history,
            metadata={} if metadata is None else metadata,
        )
    if record.model is None:
        raise ValueError("A checkpoint requires a model.")
    if not isinstance(record.history, dict) or not isinstance(record.metadata, dict):
        raise TypeError("Checkpoint history and metadata must be dictionaries.")
    _validate_pair(record.model, record.projection)
    arrays = {}
    payload = _pack(
        dict(
            model=_dump_component(record.model),
            projection=_dump_component(record.projection),
            history=record.history,
            metadata=record.metadata,
        ),
        arrays,
    )
    manifest = json.dumps(
        dict(format="nitrom.checkpoint", version=_FORMAT_VERSION, payload=payload),
        allow_nan=False,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, manifest=np.array(manifest), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve_backend(backend):
    if backend is None:
        return get_backend()
    if isinstance(backend, str):
        if backend == "numpy":
            from .backend._numpy import NumpyBackend

            return NumpyBackend()
        if backend == "torch":
            from .backend._torch import TorchBackend

            return TorchBackend()
        raise ValueError(f"Unknown checkpoint backend: {backend!r}.")
    return backend


def load_checkpoint(path, *, backend=None, device=None, dtype=None) -> Checkpoint:
    """Reconstruct a saved model/projection and return a :class:`Checkpoint`.

    ``backend`` is an adapter or ``"numpy"``/``"torch"``; by default the active
    construction backend is used. This does not change the global backend.
    ``device`` defaults to CPU. ``dtype`` optionally overrides saved component
    dtypes and may be a backend dtype or a name such as ``"float64"``.
    Custom types require their registered adapters to be available on load.
    """
    bkend = _resolve_backend(backend)
    if isinstance(dtype, str):
        dtype = getattr(bkend, dtype)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        if manifest.get("format") != "nitrom.checkpoint":
            raise ValueError("Not a NiTROM checkpoint archive.")
        if manifest.get("version") != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {manifest.get('version')!r}."
            )
        payload = _unpack(manifest["payload"], archive)

    def restore(spec):
        if spec is None:
            return None
        name = spec["type"]
        if name not in _TYPES:
            raise ValueError(f"Unregistered checkpoint type: {name!r}.")
        cls, _, loader = _TYPES[name]
        obj = loader(spec["state"], backend=bkend, device=device or "cpu", dtype=dtype)
        if type(obj) is not cls:
            raise TypeError(
                f"Checkpoint adapter {name!r} did not return {cls.__name__}."
            )
        return obj

    model, projection = restore(payload["model"]), restore(payload["projection"])
    if model is None:
        raise ValueError("A checkpoint requires a model.")
    _validate_pair(model, projection, link=True)
    return Checkpoint(model, projection, payload["history"], payload["metadata"])


def _component_state(component):
    params = component.get_params()
    dtype = params[0].dtype if params else component.dtype
    return dict(
        dtype=str(dtype).removeprefix("torch."),
        parameters=dict(zip(component.param_names, params, strict=True)),
    )


def _dump_model(model):
    state = _component_state(model)
    inner = model.model if isinstance(model, GasPolynomialModel) else model
    forcing = None
    if model.forcing_exists:
        forcing = dict(forcing_exists=True, m=model.B.shape[1])
        # The public B-is-fixed convention is represented by the inner model's
        # bkend attribute. Store the current operator, not a stale constructor B.
        if hasattr(inner, "bkend"):
            forcing["B"] = model.B
    state.update(
        r=model.state_dimension,
        poly_comp=list(model.poly_comp),
        instability_threshold=inner.thresh,
        forcing_config=forcing,
    )
    return state


def _parameters(state, shapes, backend, device, dtype):
    params = state["parameters"]
    if set(params) != set(shapes):
        raise ValueError(
            "Checkpoint parameter names do not match component configuration."
        )
    values = []
    for name, shape in shapes.items():
        value = params[name]
        if value.shape != shape:
            raise ValueError(
                f"Checkpoint parameter {name!r} has shape {value.shape}, expected {shape}."
            )
        values.append(backend.asarray(value, dtype=dtype, device=device))
    return values


def _load_model(cls, state, *, backend, device, dtype):
    dtype = dtype if dtype is not None else getattr(backend, state["dtype"])
    r, degrees = state["r"], state["poly_comp"]
    forcing = state["forcing_config"]
    if forcing is not None and "B" in forcing:
        forcing = dict(
            forcing, B=backend.asarray(forcing["B"], dtype=dtype, device=device)
        )
    if cls is PolynomialModel:
        shapes = {f"A_{k}": (r,) * (k + 1) for k in degrees}
        param_key = "tensors"
    else:
        shapes = {name: shape for name, shape, _ in cls._gas_param_specs(r, degrees)}
        param_key = "atr_params" if cls is AtrPolynomialModel else "gas_params"
    if forcing is not None and forcing["forcing_exists"]:
        shapes["B"] = (r, forcing["m"])
    params = _parameters(state, shapes, backend, device, dtype)
    return cls(
        r,
        degrees,
        backend=backend,
        device=device,
        dtype=dtype,
        instability_threshold=state["instability_threshold"],
        forcing_config=forcing,
        **{param_key: params},
    )


def _dump_projection(projection):
    state = _component_state(projection)
    state.update(
        n=projection.ambient_space_dimension, r=projection.latent_space_dimension
    )
    if isinstance(projection, LinearProjection):
        state["biorthogonalize"] = projection.biorthogonalize
    else:
        state["nonlin_poly_comp"] = list(projection.nonlin_poly_comp)
    return state


def _load_projection(cls, state, *, backend, device, dtype):
    dtype = dtype if dtype is not None else getattr(backend, state["dtype"])
    n, r = state["n"], state["r"]
    shapes = dict(Phi=(n, r), Psi=(n, r))
    if cls is PolynomialProjection:
        shapes.update({f"A{k}": (n,) + (r,) * k for k in state["nonlin_poly_comp"]})
    params = _parameters(state, shapes, backend, device, dtype)
    if cls is LinearProjection:
        return cls(params, biorthogonalize=state["biorthogonalize"], backend=backend)
    return cls(state["nonlin_poly_comp"], params, backend=backend)


def _register_builtin(cls, dump, loader):
    def load(state, **kwargs):
        return loader(cls, state, **kwargs)

    register_checkpoint_type(f"nitrom.{cls.__name__}", cls, dump=dump, load=load)


for _cls in (PolynomialModel, GasPolynomialModel, AtrPolynomialModel):
    _register_builtin(_cls, _dump_model, _load_model)
for _cls in (LinearProjection, PolynomialProjection):
    _register_builtin(_cls, _dump_projection, _load_projection)
