"""Explicit, on-demand checks of ROM setup and structure."""

from .gradients import compare_gradients
from .models import check_model_properties, register_model_check
from .projections import check_projection

__all__ = [
    "check_model_properties",
    "check_projection",
    "compare_gradients",
    "register_model_check",
]
