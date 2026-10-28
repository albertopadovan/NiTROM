"""Lorenz-96 with periodic indexing and uniform scalar input 8 + u(t)."""

import numpy as np


class Lorenz96:
    output_is_state = True

    def __init__(self, n=40, F=8.0):
        if n < 4:
            raise ValueError("Lorenz-96 requires at least four sites.")
        self.n, self.F = n, F

    def rhs(self, t, x, forcing=0.0):
        """State axis is last: (n,) or (..., n); forcing broadcasts to x."""
        return (
            np.roll(x, 1, axis=-1) * (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1))
            - x
            + self.F
            + forcing
        )

    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):
        return np.eye(self.n)


def input_functions(amplitudes, *, sinusoidal=None, period=5.0):
    """Reconstruct scalar inputs from arrays rather than serializing callables."""
    if sinusoidal is None:
        sinusoidal = np.zeros(len(amplitudes), dtype=bool)
    return [
        (lambda t, a=float(a): np.array([a * np.sin(2 * np.pi * t / period)]))
        if sine
        else (lambda t, a=float(a): np.array([a]))
        for a, sine in zip(amplitudes, sinusoidal, strict=True)
    ]
