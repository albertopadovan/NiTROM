"""Non-intrusive trajectory-based reduced-order modelling."""

from .backend import get_backend, set_backend
from .bases import compute_data_driven_balancing, compute_pod
from .checkpointing import (
    Checkpoint,
    load_checkpoint,
    register_checkpoint_type,
    save_checkpoint,
)
from .diagnostics import (
    check_model_properties,
    check_projection,
    compare_gradients,
    register_model_check,
)
from .distributed import cleanup_distributed, setup_distributed
from .latent_space_models import (
    AtrPolynomialModel,
    GasPolynomialModel,
    Model,
    PolynomialModel,
)
from .optimization import (
    InferenceModule,
    NitromModule,
    OpInfModule,
    PolyManifoldInfModule,
    solve_opinf,
    train,
)
from .projections import LinearProjection, PolynomialProjection, Projection
from .roms.param_registry import ParamRegistry
from .time_steppers import (
    FixedStepSolution,
    solve_adjoint_ivp_discrete,
    solve_fixed_ivp,
    solve_ivp,
    step,
)
from .training_data import TrainingData, TrainingPool

__all__ = [
    "AtrPolynomialModel",
    "Checkpoint",
    "FixedStepSolution",
    "GasPolynomialModel",
    "InferenceModule",
    "LinearProjection",
    "Model",
    "NitromModule",
    "OpInfModule",
    "ParamRegistry",
    "PolyManifoldInfModule",
    "PolynomialModel",
    "PolynomialProjection",
    "Projection",
    "TrainingData",
    "TrainingPool",
    "check_model_properties",
    "check_projection",
    "cleanup_distributed",
    "compare_gradients",
    "compute_data_driven_balancing",
    "compute_pod",
    "get_backend",
    "load_checkpoint",
    "register_checkpoint_type",
    "register_model_check",
    "save_checkpoint",
    "set_backend",
    "setup_distributed",
    "solve_adjoint_ivp_discrete",
    "solve_fixed_ivp",
    "solve_ivp",
    "solve_opinf",
    "step",
    "train",
]
