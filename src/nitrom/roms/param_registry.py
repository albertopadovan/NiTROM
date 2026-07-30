from dataclasses import dataclass
from typing import Any

from ..backend import get_backend
from ..latent_space_models.model import Model
from ..projections.projection import Projection


@dataclass(frozen=True)
class RegisteredParam:
    r"""
    A single entry in a :class:`ParamRegistry`.

    :param name: parameter name (matches the corresponding entry in
        :attr:`Model.param_names` and/or :attr:`Projection.param_names`)
    :type name: str
    :param shared: ``True`` if the same logical parameter lives in *both*
        the latent-space model and the projection (e.g. an encoder
        :math:`\Psi` that also appears in the latent dynamics), ``False``
        otherwise
    :type shared: bool
    :param sources: which components hold this parameter; a subset of
        ``("model", "projection")``.  Length two iff ``shared`` is ``True``.
    :type sources: tuple[str, ...]
    """

    name: str
    shared: bool
    sources: tuple[str, ...]


class ParamRegistry:
    r"""
    Registry of the learnable parameters of a ROM, spanning the
    latent-space :class:`Model` and the :class:`Projection`.

    Each parameter is recorded once with a :attr:`RegisteredParam.shared`
    flag: ``True`` when the *same* logical parameter appears in both the
    model and the projection (so a single optimization variable feeds both
    sites and their gradients must be summed), ``False`` when it belongs to
    only one component.

    Sharing is keyed by parameter name and detected automatically: any name
    appearing in both :attr:`Model.param_names` and
    :attr:`Projection.param_names` is treated as shared.

    :param model: latent-space dynamics model
    :type model: Model
    :param projection: projection between ambient and latent spaces
    :type projection: Projection
    """

    def __init__(self, model: Model, projection: Projection):
        self._model = model
        self._projection = projection

        model_names = list(model.param_names)
        proj_names = list(projection.param_names)
        self._shared_names = [n for n in model_names if n in proj_names]

        # Build the ordered list of entries: model parameters first
        # (shared ones recorded at their model position), then the
        # projection-only parameters.
        entries: list[RegisteredParam] = []
        for name in model_names:
            if name in self._shared_names:
                entries.append(
                    RegisteredParam(name, True, ("model", "projection"))
                )
            else:
                entries.append(RegisteredParam(name, False, ("model",)))
        for name in proj_names:
            if name not in self._shared_names:
                entries.append(RegisteredParam(name, False, ("projection",)))

        self._entries = entries
        self._by_name = {entry.name: entry for entry in entries}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def model(self) -> Model:
        """Latent-space dynamics model."""
        return self._model

    @property
    def projection(self) -> Projection:
        """Projection between ambient and latent spaces."""
        return self._projection

    @property
    def entries(self) -> list[RegisteredParam]:
        """All registered parameters, in optimization order."""
        return list(self._entries)

    @property
    def names(self) -> list[str]:
        """Names of all registered parameters, in optimization order."""
        return [entry.name for entry in self._entries]

    @property
    def shared_names(self) -> list[str]:
        """Names of the parameters shared between model and projection."""
        return [entry.name for entry in self._entries if entry.shared]

    @property
    def independent_names(self) -> list[str]:
        """Names of the parameters belonging to a single component."""
        return [entry.name for entry in self._entries if not entry.shared]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, name: str) -> RegisteredParam:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def is_shared(self, name: str) -> bool:
        """Return whether parameter ``name`` is shared."""
        return self._by_name[name].shared

    def sources(self, name: str) -> tuple[str, ...]:
        """Return which components hold parameter ``name``."""
        return self._by_name[name].sources

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------
    def value(self, name: str) -> Any:
        r"""
        Return the current tensor for parameter ``name``.

        For a shared parameter the value is read from the model's copy;
        use :meth:`assert_shared_consistent` to verify the two copies agree.

        :param name: parameter name
        :type name: str
        :rtype: Any
        """
        entry = self._by_name[name]
        component = "model" if "model" in entry.sources else "projection"
        return self._component_values(component)[name]

    def values(self) -> list[Any]:
        """Current tensors for all registered parameters, in order."""
        return [self.value(name) for name in self.names]

    def _component_values(self, component: str) -> dict[str, Any]:
        """Map ``name -> tensor`` for the given component."""
        if component == "model":
            return dict(zip(self._model.param_names, self._model.get_params()))
        return dict(
            zip(self._projection.param_names, self._projection.get_params())
        )

    def assert_shared_consistent(self, atol: float = 0.0, rtol: float = 1e-12) -> None:
        r"""
        Verify that every shared parameter holds the same value in the model
        and the projection, raising if any pair has drifted apart.

        :param atol: absolute tolerance passed to :func:`torch.allclose`
        :type atol: float
        :param rtol: relative tolerance passed to :func:`torch.allclose`
        :type rtol: float
        :raises ValueError: if a shared parameter's two copies disagree
        """
        model_vals = self._component_values("model")
        proj_vals = self._component_values("projection")
        for name in self.shared_names:
            if not get_backend().allclose(
                model_vals[name], proj_vals[name], atol=atol, rtol=rtol
            ):
                raise ValueError(
                    f"Shared parameter '{name}' has drifted: the model and "
                    f"projection copies are not equal."
                )

    # ------------------------------------------------------------------
    # Enforcing equality of shared parameters
    # ------------------------------------------------------------------
    def scatter(self, values: list[Any]) -> None:
        r"""
        Write parameter values into both components, routing each value
        through its component's update method.

        This is the single write path that *enforces* equality of shared
        parameters: a shared value is written as the **same tensor object**
        into both the model and the projection, so the two copies cannot
        drift.  Updating the ROM only ever through this method keeps shared
        parameters identical by construction.

        :param values: tensors aligned with :attr:`names` (i.e. one value per
            registered parameter, shared parameters appearing once)
        :type values: list[Any]
        """
        if len(values) != len(self._entries):
            raise ValueError(
                f"Expected {len(self._entries)} values (one per registered "
                f"parameter), got {len(values)}."
            )
        vals = dict(zip(self.names, values))
        self._model.update_params([vals[n] for n in self._model.param_names])
        self._projection.update([vals[n] for n in self._projection.param_names])

    def sync(self, canonical: str = "model") -> None:
        r"""
        Force every shared parameter's two copies to agree *right now* by
        taking the value from ``canonical`` and re-scattering.

        Use this once after independent initialization (when the model and
        projection may hold different copies of a shared parameter); from
        then on, update exclusively through :meth:`scatter` to keep them in
        sync.

        :param canonical: which component's copy to treat as authoritative
            for shared parameters, ``"model"`` or ``"projection"``
        :type canonical: str
        """
        if canonical not in ("model", "projection"):
            raise ValueError(
                f"canonical must be 'model' or 'projection', got {canonical!r}."
            )
        model_vals = self._component_values("model")
        proj_vals = self._component_values("projection")
        canon = model_vals if canonical == "model" else proj_vals
        values = []
        for entry in self._entries:
            if entry.shared:
                values.append(canon[entry.name])
            elif "model" in entry.sources:
                values.append(model_vals[entry.name])
            else:
                values.append(proj_vals[entry.name])
        self.scatter(values)

    def __repr__(self) -> str:
        items = ", ".join(
            f"{e.name}{'*' if e.shared else ''}" for e in self._entries
        )
        return f"ParamRegistry([{items}])  (* = shared)"
