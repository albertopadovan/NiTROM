"""Three-state forced quadratic ODE reduced to two latent coordinates."""

import numpy as np


class ToyModel:
    def __init__(self, beta=20.0):
        self.beta = beta
        self.A = np.diag([-1.0, -2.0, -5.0])
        self.H = np.zeros((3, 3, 3))
        self.H[:, :, -1] = np.diag([beta, beta, 0.0])
        self.B = np.ones((3, 1))
        self.C = np.ones((1, 3))

    def rhs(self, t, x, forcing):
        """States have shape (..., 3); forcing is the full-space field B u."""
        return x @ self.A.T + np.einsum("ijk,...j,...k->...i", self.H, x, x) + forcing

    def compute_output(self, q):
        return self.C @ q

    def compute_output_derivative(self, q):
        return self.C

    def steady_weights(self, amplitudes):
        b = np.asarray(amplitudes)
        ratio = self.beta / 5 * b
        steady = np.stack([-b / (-1 + ratio), -b / (-2 + ratio), b / 5], axis=-1)
        return np.sum((steady @ self.C.T) ** 2, axis=-1)

    def project(self, Phi, Psi):
        decoder = Phi @ np.linalg.inv(Psi.T @ Phi)
        A = Psi.T @ self.A @ decoder
        H = np.einsum("ai,ijk,jb,kc->abc", Psi.T, self.H, decoder, decoder)
        return [A, H, Psi.T @ self.B]
