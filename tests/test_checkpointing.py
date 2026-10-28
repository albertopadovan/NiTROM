"""Checkpoints preserve dynamics, coordinate conventions, and component types."""

import json

import numpy as np
import pytest

from nitrom import (
    AtrPolynomialModel,
    Checkpoint,
    GasPolynomialModel,
    LinearProjection,
    PolynomialModel,
    PolynomialProjection,
    get_backend,
    load_checkpoint,
    register_checkpoint_type,
    save_checkpoint,
    set_backend,
)


def make_model(cls, backend, forcing):
    def array(x):
        return backend.asarray(x, dtype=backend.float64)

    B = array([[0.2], [-0.3]])
    config = None if forcing == "none" else dict(forcing_exists=True, m=1)
    if forcing == "fixed":
        config["B"] = B
    if cls is PolynomialModel:
        degrees = [0, 1, 2, 3]
        params = [array(np.full((2,) * (k + 1), 0.01 / (k + 1))) for k in degrees]
        key = "tensors"
    else:
        degrees = [1, 2]
        params = [
            array([[0, 0.1], [-0.2, 0]]),
            array(np.eye(2)),
            array([[1.0, 0.1], [0.1, 1.1]]),
            array(np.arange(8).reshape(2, 2, 2) * 0.01),
        ]
        if cls is AtrPolynomialModel:
            params += [array([0.1, -0.2]), array([0.3, -0.1])]
        key = "atr_params" if cls is AtrPolynomialModel else "gas_params"
    if config is not None:
        params.append(B)
    return cls(
        2,
        degrees,
        backend=backend,
        dtype=backend.float64,
        forcing_config=config,
        instability_threshold=37,
        **{key: params},
    )


@pytest.mark.parametrize(
    "cls", [PolynomialModel, GasPolynomialModel, AtrPolynomialModel]
)
@pytest.mark.parametrize("forcing", ["none", "learned", "fixed"])
def test_model_roundtrip_preserves_dynamics_and_free_parameters(
    tmp_path, backend, cls, forcing
):
    model = make_model(cls, backend, forcing)
    path = tmp_path / "nested" / "model.npz"
    save_checkpoint(
        path, model, history={"loss": [1.0, 0.5]}, metadata={"basis": np.eye(2)}
    )
    restored = load_checkpoint(path)
    assert type(restored.model) is cls
    assert restored.projection is None
    assert restored.history == {"loss": [1.0, 0.5]}
    np.testing.assert_array_equal(restored.metadata["basis"], np.eye(2))
    assert restored.model.param_names == model.param_names
    for actual, expected in zip(
        restored.model.get_params(), model.get_params(), strict=True
    ):
        np.testing.assert_array_equal(
            backend.to_numpy(actual), backend.to_numpy(expected)
        )
    z = backend.asarray([[0.2, -0.1], [-0.3, 0.4]], dtype=backend.float64)
    inputs = [lambda t: backend.asarray([t], dtype=backend.float64)] * 2
    kwargs = dict(external_forcing=inputs) if forcing != "none" else {}
    np.testing.assert_allclose(
        backend.to_numpy(restored.model.evaluate_rhs(0.2, z, **kwargs)),
        backend.to_numpy(model.evaluate_rhs(0.2, z, **kwargs)),
        rtol=1e-13,
        atol=1e-14,
    )
    inner = restored.model.model if cls is not PolynomialModel else restored.model
    assert inner.thresh == 37
    assert hasattr(inner, "bkend") == (forcing == "fixed")


@pytest.mark.parametrize("kind", ["encoder", "decoder", "polynomial"])
def test_projection_roundtrip_preserves_maps_and_precision(tmp_path, backend, kind):
    def array(x):
        return backend.asarray(x, dtype=backend.float32)

    Phi = array([[1, 0], [0, 1], [0.1, -0.2]])
    Psi = array([[0.9, 0.1], [0, 1.1], [0.2, -0.1]])
    if kind == "polynomial":
        projection = PolynomialProjection(
            [2, 3],
            [
                Phi,
                Psi,
                array(np.arange(12).reshape(3, 2, 2) * 0.01),
                array(np.arange(24).reshape(3, 2, 2, 2) * 0.001),
            ],
            backend=backend,
        )
    else:
        projection = LinearProjection([Phi, Psi], biorthogonalize=kind, backend=backend)
    model = PolynomialModel(
        2, [1], backend=backend, dtype=backend.float32, instability_threshold=np.inf
    )
    path = tmp_path / "projection.npz"
    save_checkpoint(path, model, projection)
    loaded = load_checkpoint(path)
    assert type(loaded.projection) is type(projection)
    assert loaded.projection.Phi.dtype == backend.float32
    assert loaded.model.thresh == np.inf
    q = array([[0.3, -0.2, 0.1], [0.1, 0.4, -0.1]])
    z = projection.encode(q)
    np.testing.assert_allclose(
        backend.to_numpy(loaded.projection.encode(q)), backend.to_numpy(z)
    )
    np.testing.assert_allclose(
        backend.to_numpy(loaded.projection.decode(z)),
        backend.to_numpy(projection.decode(z)),
    )


@pytest.mark.parametrize("source,target", [("numpy", "torch"), ("torch", "numpy")])
def test_backend_conversion_is_explicit_and_does_not_change_default(
    tmp_path, source, target
):
    pytest.importorskip("torch")
    bkend = set_backend(source)
    model = make_model(GasPolynomialModel, bkend, "fixed")
    phi = bkend.asarray(np.eye(3, 2), dtype=bkend.float64)
    projection = LinearProjection([phi, phi], backend=bkend)
    path = tmp_path / "portable.npz"
    save_checkpoint(path, model, projection)
    previous = get_backend()
    loaded = load_checkpoint(path, backend=target, dtype="float32", device="cpu")
    assert get_backend() is previous
    assert loaded.model.backend.name == target
    assert loaded.projection.backend.name == target
    for value in [*loaded.model.get_params(), *loaded.projection.get_params()]:
        assert value.dtype == loaded.model.backend.float32


def test_metadata_encoding_and_snapshot_ownership(tmp_path, numpy_backend):
    model = PolynomialModel(1, [1], tensors=[np.array([[-1.0]])])
    metadata = dict(
        nested={"kind": "array", "key": "user value"},
        values=(None, True, np.int64(4), float("inf")),
        matrix=np.arange(4).reshape(2, 2),
    )
    record = Checkpoint(model, metadata=metadata)
    path = tmp_path / "checkpoint.custom_suffix"
    save_checkpoint(path, record)
    model.A_1[:] = 5
    metadata["matrix"][:] = 100
    loaded = load_checkpoint(path)
    assert loaded.model.A_1[0, 0] == -1
    assert loaded.metadata["nested"] == metadata["nested"]
    assert loaded.metadata["values"] == (None, True, 4, float("inf"))
    np.testing.assert_array_equal(loaded.metadata["matrix"], np.arange(4).reshape(2, 2))


def test_custom_type_registration_and_shared_parameters(tmp_path, numpy_backend):
    class SharedModel:
        def __init__(self, A, Phi, backend):
            self.A, self.Phi, self.backend = A, Phi, backend
            self.state_dimension = A.shape[0]
            self.param_names = ["A", "Phi"]

        def get_params(self):
            return [self.A, self.Phi]

        def update_params(self, values):
            self.A, self.Phi = values

    def dump(model):
        return dict(A=model.A, Phi=model.Phi)

    def load(state, *, backend, device, dtype):
        return SharedModel(
            *(
                backend.asarray(state[key], dtype=dtype, device=device)
                for key in ("A", "Phi")
            ),
            backend,
        )

    phi = np.eye(3, 2)
    model = SharedModel(np.eye(2), phi.copy(), numpy_backend)
    projection = LinearProjection([phi.copy(), phi.copy()])
    path = tmp_path / "custom.npz"
    with pytest.raises(TypeError, match="Unregistered"):
        save_checkpoint(path, model, projection)
    register_checkpoint_type("test.SharedModel", SharedModel, dump=dump, load=load)
    with pytest.raises(ValueError, match="already registered"):
        register_checkpoint_type("test.SharedModel", SharedModel, dump=dump, load=load)
    save_checkpoint(path, model, projection)
    restored = load_checkpoint(path)
    assert type(restored.model) is SharedModel
    assert restored.model.Phi is restored.projection.Phi
    model.Phi[:] = 2
    with pytest.raises(ValueError, match="has drifted"):
        save_checkpoint(path, model, projection)


def test_failed_writes_preserve_existing_checkpoint(
    tmp_path, numpy_backend, monkeypatch
):
    model = PolynomialModel(1, [1])
    path = tmp_path / "model.npz"
    save_checkpoint(path, model)
    before = path.read_bytes()
    with pytest.raises(TypeError, match="Unsupported"):
        save_checkpoint(path, model, metadata={"forcing": lambda t: t})
    assert path.read_bytes() == before

    def fail(stream, **arrays):
        stream.write(b"incomplete archive")
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(path, model)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_archive_version_and_parameter_shape(tmp_path, numpy_backend):
    model = PolynomialModel(2, [1])
    path = tmp_path / "model.npz"
    save_checkpoint(path, model)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    manifest = json.loads(str(values["manifest"].item()))
    manifest["version"] = 999
    np.savez(path, **dict(values, manifest=np.array(json.dumps(manifest))))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path)
    manifest["version"] = 1
    values["array_0"] = np.zeros((3, 2))
    np.savez(path, **dict(values, manifest=np.array(json.dumps(manifest))))
    with pytest.raises(ValueError, match="shape"):
        load_checkpoint(path)


def test_incompatible_components_rejected(tmp_path, numpy_backend):
    model = PolynomialModel(2, [1])
    projection = LinearProjection([np.eye(3, 1), np.eye(3, 1)])
    with pytest.raises(ValueError, match="dimensions"):
        save_checkpoint(tmp_path / "bad.npz", model, projection)
