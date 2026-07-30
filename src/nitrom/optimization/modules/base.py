import abc
from collections import OrderedDict
from typing import Any

from ...backend import get_backend

#: Manifolds a parameter may be optimized on (see
#: :meth:`InferenceModule.set_manifold_types`).
MANIFOLD_TYPES = frozenset({"euclidean", "grassmann", "stiefel"})


class BackendModule:
    r"""Minimal ``nn.Module``-like parameter container for either backend.

    Provides just the parameter-registration surface NiTROM uses
    (``register_parameter``, :meth:`parameters`, :meth:`named_parameters`, and
    ``nn.Parameter`` auto-registration on attribute assignment).  Under the
    **torch** backend, registered parameters are wrapped in
    :class:`torch.nn.Parameter`, so ``torch.optim``, autograd, and ``.grad``
    behave exactly as with a real ``nn.Module``.  Under the **numpy** backend,
    parameters are stored as plain arrays.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_params", OrderedDict())

    def register_parameter(self, name: str, value: Any) -> None:
        """Register ``value`` as a trainable parameter named ``name``."""
        bkend = get_backend()
        if bkend.is_torch:
            import torch
            if not isinstance(value, torch.nn.Parameter):
                value = torch.nn.Parameter(value)
        self._params[name] = value
        object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        # Mirror nn.Module: assigning a torch.nn.Parameter auto-registers it
        # (used by lightweight test modules).  Numpy params must be registered
        # explicitly via register_parameter (plain arrays carry no marker).
        if get_backend().is_torch:
            import torch
            if isinstance(value, torch.nn.Parameter):
                self.register_parameter(name, value)
                return
        object.__setattr__(self, name, value)

    def parameters(self) -> list:
        """Registered parameters, in registration order."""
        return list(self._params.values())

    def named_parameters(self) -> list:
        """``(name, parameter)`` pairs, in registration order."""
        return list(self._params.items())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class InferenceModule(BackendModule, abc.ABC):
    r"""
    Abstract base class for inference modules trained with analytic gradients.

    An inference module exposes a scalar cost via :meth:`forward` and the
    *analytic* gradient of that cost via :meth:`gradient`.  Keeping both on the
    same object lets a single training loop (see
    :func:`nitrom.optimization.train`) drive any concrete module -- operator
    inference, polynomial-manifold inference, NiTROM, ... -- without relying on
    autograd, on either array backend.

    **Invariant.** :meth:`gradient` must return one tensor per trainable
    parameter, in the *same order* as :meth:`parameters`.  The training loop
    zips the two together to assign gradients, so the ordering contract must
    hold for every subclass.

    **Learnability.** Every registered parameter is learnable by default.
    Callers may freeze parameters with :meth:`set_unlearnable` (and restore
    them with :meth:`set_learnable`); a subclass enforces this by passing its
    computed gradients through :meth:`_apply_learnability`, which zeroes the
    gradient of any frozen parameter so the optimizer leaves it fixed.

    **Manifolds.** A parameter may be optimized on a matrix manifold
    (``"grassmann"`` or ``"stiefel"``) rather than Euclidean space; configure
    this per parameter with :meth:`set_manifold_types`.  The training loop reads
    :meth:`get_manifold_types` to retract iterates and tangent-project the
    gradients.
    """

    @property
    def is_learnable(self) -> dict[str, bool]:
        """
        Per-parameter learnability flags, keyed by parameter name in
        :meth:`parameters` order.  Lazily initialized to ``True`` for every
        registered parameter on first access.
        """
        if getattr(self, "_is_learnable", None) is None:
            self._is_learnable = {
                name: True for name, _ in self.named_parameters()
            }
        return self._is_learnable

    def set_unlearnable(self, *names: str) -> None:
        """
        Mark one or more parameters as non-learnable.

        All parameters are learnable by default.  A non-learnable parameter
        keeps its current value during training: its gradient is zeroed in
        :meth:`_apply_learnability`, so the optimizer never updates it.  A
        typical use is to freeze the input operator, ``set_unlearnable("B")``.

        :param names: parameter names, each a registered parameter
        :raises KeyError: if a name is not a registered parameter
        """
        self._set_learnable(names, False)

    def set_learnable(self, *names: str) -> None:
        """
        Mark one or more parameters as learnable (the default state).

        Reverses :meth:`set_unlearnable`, so the optimizer updates these
        parameters again.

        :param names: parameter names, each a registered parameter
        :raises KeyError: if a name is not a registered parameter
        """
        self._set_learnable(names, True)

    def _set_learnable(self, names: tuple[str, ...], value: bool) -> None:
        for name in names:
            if name not in self.is_learnable:
                raise KeyError(
                    f"Unknown parameter '{name}'; expected one of "
                    f"{list(self.is_learnable)}."
                )
            self.is_learnable[name] = value

    def _apply_learnability(self, grads: list) -> list:
        """
        Zero the gradient of any non-learnable parameter, in place.

        ``grads`` must be in :meth:`parameters` order (the module contract),
        which matches the key order of :attr:`is_learnable`.

        :param grads: gradients, one per parameter, in parameter order
        :returns: the same list with non-learnable entries zeroed
        """
        bkend = get_backend()
        for i, learnable in enumerate(self.is_learnable.values()):
            if not learnable:
                grads[i] = bkend.zeros_like(grads[i])
        return grads

    @property
    def manifold_types(self) -> list[str]:
        """
        Per-parameter manifold type, in :meth:`parameters` order.  Lazily
        initialized to ``"euclidean"`` for every registered parameter; set it
        with :meth:`set_manifold_types`.
        """
        if getattr(self, "_manifold_types", None) is None:
            self._manifold_types = ["euclidean"] * len(self.parameters())
        return self._manifold_types

    def set_manifold_types(self, names: list[str], types: list[str]) -> None:
        """
        Assign the manifold each named parameter is optimized on.

        :param names: parameter names
        :param types: matching manifold types, each one of ``"euclidean"``,
            ``"grassmann"``, or ``"stiefel"``
        :raises ValueError: if ``names`` and ``types`` differ in length, or a
            type is not a recognized manifold
        :raises KeyError: if a name is not a registered parameter
        """
        if len(names) != len(types):
            raise ValueError(
                f"names and types must have equal length, got "
                f"{len(names)} and {len(types)}."
            )
        index = {name: i for i, (name, _) in enumerate(self.named_parameters())}
        mtypes = self.manifold_types
        for name, mtype in zip(names, types, strict=True):
            if name not in index:
                raise KeyError(
                    f"Unknown parameter '{name}'; expected one of {list(index)}."
                )
            m = mtype.lower()
            if m not in MANIFOLD_TYPES:
                raise ValueError(
                    f"manifold type for '{name}' must be one of "
                    f"{sorted(MANIFOLD_TYPES)}, got '{mtype}'."
                )
            mtypes[index[name]] = m

    def get_manifold_types(self) -> list[str]:
        """Return the per-parameter manifold types, in :meth:`parameters` order."""
        return self.manifold_types

    @abc.abstractmethod
    def forward(self) -> Any:
        """
        Evaluate the scalar cost.

        :returns: scalar loss
        """
        ...

    @abc.abstractmethod
    def gradient(self) -> list:
        """
        Compute the analytic gradient of the cost with respect to the
        trainable parameters, in the same order as :meth:`parameters`.

        :returns: list of gradient arrays, one per trainable parameter
        """
        ...
