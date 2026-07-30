"""Shared finite-difference gradient-check utilities for the inference modules.

Each module exposes a scalar cost via ``module()`` and its analytic gradient via
``module.gradient()`` (in :meth:`parameters` order).  These helpers compare the
analytic gradient against a central finite difference of the cost, which works
uniformly whether or not the module's forward pass is autograd-friendly (the
NiTROM adjoint, for instance, is hand-derived).
"""

import numpy as np
import torch

DTYPE = torch.float64


def random_basis(N: int, r: int, seed: int) -> torch.Tensor:
    """A random ``(N, r)`` matrix with orthonormal columns."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(N, r, generator=g, dtype=DTYPE))
    return Q.contiguous()


class Data:
    """Minimal stand-in for :class:`TrainingData` (X, dX, weights, time, forcing)."""

    def __init__(self, X, dX=None, weights=None, time=None, forcing_fns=None):
        self.X = X
        self.dX = dX
        ntraj, _, nt = X.shape
        self.weights = (
            weights if weights is not None else torch.ones(ntraj, dtype=DTYPE)
        )
        self.time = (
            time if time is not None else torch.linspace(0.0, 1.0, nt, dtype=DTYPE)
        )
        self.forcing_fns = forcing_fns if forcing_fns is not None else []


class LinearOutputFOM:
    """Full-order model with a linear output ``y = C x`` (for NiTROM)."""

    def __init__(self, C: torch.Tensor):
        self.C = C  # (no, N)

    def compute_output(self, q: torch.Tensor) -> torch.Tensor:
        # (no, N) @ (..., N, nt) -> (..., no, nt); also handles (N,) -> (no,).
        return torch.matmul(self.C, q)

    def compute_output_derivative(self, q: torch.Tensor) -> torch.Tensor:
        return self.C  # constant Jacobian


def finite_diff_grad(module, eps: float = 1e-6) -> list[torch.Tensor]:
    """Central finite-difference gradient of ``module()`` w.r.t. each parameter."""
    fd = []
    with torch.no_grad():
        for p in module.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()  # same values; enables flat view
            g = torch.zeros_like(p)
            flat, gflat = p.view(-1), g.view(-1)
            for i in range(flat.numel()):
                orig = flat[i].item()
                flat[i] = orig + eps
                jp = float(module())
                flat[i] = orig - eps
                jm = float(module())
                flat[i] = orig
                gflat[i] = (jp - jm) / (2.0 * eps)
            fd.append(g)
    return fd


def assert_grad_close(analytic, numeric, rtol: float = 1e-5, atol: float = 1e-7):
    """Assert each analytic gradient matches its finite-difference counterpart."""
    assert len(analytic) == len(numeric), (
        f"{len(analytic)} analytic vs {len(numeric)} numeric gradients"
    )
    for i, (a, n) in enumerate(zip(analytic, numeric, strict=True)):
        assert a.shape == n.shape, f"param {i}: shape {a.shape} vs {n.shape}"
        np.testing.assert_allclose(
            a.detach().cpu().numpy(),
            n.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"param {i} gradient mismatch",
        )
