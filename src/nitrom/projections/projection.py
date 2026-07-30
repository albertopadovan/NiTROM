import abc
from typing import Any

from ..backend import get_backend


class Projection(metaclass=abc.ABCMeta):
    """Abstract base class for projections between full and reduced spaces.

    Binds to the active array backend (NumPy or PyTorch) at construction via
    :func:`nitrom.backend.get_backend`; all array operations go through
    :attr:`backend`.
    """

    def __init__(
        self,
        n: int,
        r: int,
        param_names: list[str],
        device: str = "cpu",
        dtype: Any = None,
    ):
        self._backend = get_backend()
        self._n = n
        self._r = r
        self._device = device
        self._dtype = dtype if dtype is not None else self._backend.float64
        self._param_names = param_names

    @property
    def backend(self):
        """The array backend this projection is bound to."""
        return self._backend

    @property
    def ambient_space_dimension(self) -> int:
        """Ambient space dimension."""
        return self._n

    @property
    def latent_space_dimension(self) -> int:
        """Latent space dimension."""
        return self._r

    @property
    def device(self) -> str:
        """Device on which arrays are allocated."""
        return self._device

    @property
    def dtype(self) -> Any:
        """Data type of arrays."""
        return self._dtype

    @property
    def param_names(self) -> list[str]:
        """Names of the projection parameters."""
        return self._param_names

    @abc.abstractmethod
    def get_params(self) -> list[Any]:
        """
        Return the current parameter arrays as a list, in the same
        order as :attr:`param_names`.

        :rtype: list
        """
        ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the projection parameters.
        Subclasses define the specific arguments required.
        """
        ...

    @abc.abstractmethod
    def encode(self, q: Any) -> Any:
        """
        Project from full space to reduced space.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :rtype: backend array
        """
        ...

    @abc.abstractmethod
    def decode(self, z: Any) -> Any:
        """
        Reconstruct from reduced space to full space.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        ...

    @abc.abstractmethod
    def vjp_encode(self, q: Any, v: Any, *args, **kwargs) -> tuple:
        """
        VJP of the encoder with respect to the projection parameters.
        Subclasses define the specific parameters and return order.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :param v: upstream adjoint seed of shape matching the encoder output
        :rtype: tuple
        """
        ...

    @abc.abstractmethod
    def vjp_decode(self, z: Any, v: Any, *args, **kwargs) -> tuple:
        """
        VJP of the decoder with respect to the projection parameters.
        Subclasses define the specific parameters and return order.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: upstream adjoint seed of shape matching the decoder output
        :rtype: tuple
        """
        ...

    @abc.abstractmethod
    def vjp_decode_state(self, z: Any, v: Any, *args, **kwargs) -> Any:
        r"""
        VJP of the decoder with respect to the latent **state** :math:`z`
        (not the parameters):

        .. math::

            \left(\frac{\partial\,\text{decode}(z)}{\partial z}\right)^\top v.

        This is the decoder Jacobian-transpose applied to a full-space
        cotangent, used to map an ambient-space adjoint seed back into the
        latent space (e.g. the measurement source of an adjoint solve).

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :returns: latent-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: backend array
        """
        ...
