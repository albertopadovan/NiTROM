import abc
from typing import Any

from ..backend import get_backend


class Model(metaclass=abc.ABCMeta):
    r"""Abstract base class for NiTROM latent-space dynamics models.

    A model binds to the active array backend (NumPy or PyTorch) at
    construction time via :func:`nitrom.backend.get_backend`; all array
    operations go through :attr:`backend` so the same code runs on either.

    :param r: reduced state dimension
    :type r: int
    :param param_names: names of the model parameters
    :type param_names: list[str]
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    """

    def __init__(
        self,
        r: int,
        param_names: list[str],
        device: str = "cpu",
        dtype: Any = None,
    ):
        self._backend = get_backend()
        self._r = r
        self._device = device
        self._dtype = dtype if dtype is not None else self._backend.float64
        self._param_names = param_names

    @property
    def backend(self):
        """The array backend this model is bound to."""
        return self._backend

    @property
    def state_dimension(self) -> int:
        """Reduced state dimension."""
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
        """Names of the model parameters."""
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
    def update_params(self, *args, **kwargs) -> None:
        """
        Update the model parameters.
        Subclasses define the specific arguments required.
        """
        ...

    @abc.abstractmethod
    def evaluate_rhs(self, t: float, z: Any, **kwargs) -> Any:
        r"""
        Evaluate the right-hand side of the ROM.
        t:  time instance
        z:  state vector (n,) or (m, n)
        """
        ...

    @abc.abstractmethod
    def evaluate_adjoint_rhs(self, t: float, z: Any, Z: Any, **kwargs) -> Any:
        r"""
        Evaluate the adjoint right-hand side.
        t:  time instance
        z:  adjoint state vector (n,) or (m, n)
        Z:  base flow at which to evaluate the Jacobian, same shape as z
        """
        ...

    @abc.abstractmethod
    def vjp_evaluate_rhs(self, z: Any, v: Any, *args, **kwargs) -> list:
        r"""
        VJP of the RHS with respect to the model parameters.
        Subclasses define the specific return contents.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :rtype: list
        """
        ...

    def inner_params(self) -> list[Any]:
        """
        Return the inner parameter tensors for which `vjp_evaluate_rhs` 
        directly computes gradients. By default, this is `self.get_params()`.
        
        :rtype: list
        """
        return self.get_params()

    def project_inner_gradients(self, inner_grads: list[Any]) -> list[Any]:
        """
        Project accumulated inner gradients to the actual model parameters.
        By default, this is the identity mapping.
        
        :param inner_grads: gradients with respect to `inner_params()`
        :rtype: list
        """
        return inner_grads

    def inner_vjp_evaluate_rhs(self, z: Any, v: Any, *args, **kwargs) -> list:
        """
        VJP of the RHS with respect to the inner parameter tensors.
        For models without a two-level parameterization, this is identical 
        to `vjp_evaluate_rhs`.
        """
        return self.vjp_evaluate_rhs(z, v, *args, **kwargs)
