import numpy as np
import torch


class fom_class:
    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):
        if torch.is_tensor(q):
            return torch.eye(q.shape[1], device=q.device, dtype=q.dtype)
        else:
            return np.eye(q.shape[1], dtype=q.dtype)

    def apply_output_adjoint(self, e, q):
        # compute_output is the identity, so its adjoint is the identity too;
        # this skips ever forming the dense (N, N) output-derivative matrix.
        return e
