import numpy as np
import torch

class myAdaptiveLineSearcher:
    """Adaptive line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        sufficient_decrease=0.5,
        max_iterations=10,
        initial_step_size=1,
        rank=0,
    ):
        self._contraction_factor = contraction_factor
        self._sufficient_decrease = sufficient_decrease
        self._max_iterations = max_iterations
        self._initial_step_size = initial_step_size
        self._rank = rank
        self._oldalpha = None

    def search(self, objective, manifold, x, d, f0, df0):
        norm_d = manifold.norm(x, d)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self._initial_step_size / norm_d
        alpha = float(alpha)

        try:
            newx = manifold.retraction(x, alpha * d)
        except np.linalg.LinAlgError:  # Added by Cole: catches singular matrix error
            if self._rank == 0: print("Singular matrix encountered, reducing step size.")
            alpha *= self._contraction_factor
            newx = manifold.retraction(x, alpha * d)
        newf = objective(newx)
        cost_evaluations = 0

        while (
            newf > f0 + self._sufficient_decrease * alpha * df0
            and cost_evaluations <= self._max_iterations
        ):
            # Reduce the step size.
            alpha *= self._contraction_factor

            # Look closer down the line.
            try:
                newx = manifold.retraction(x, alpha * d)
            except np.linalg.LinAlgError:  # Added by Cole
                alpha *= self._contraction_factor
                cost_evaluations += 1
                continue
            newf = objective(newx)

            cost_evaluations += 1

        # ----- Added by Alby --------
        if alpha <= 1e-12: 
            if self._rank == 0: print("Attention: allowing for cost function to increase by 1 percent")
            alpha = float(self._initial_step_size / norm_d)
            self._oldalpha = alpha

            newx = manifold.retraction(x, alpha * d)
            newf = objective(newx)
            cost_evaluations = 1

            while (
                newf > 1.01*f0 and cost_evaluations <= self._max_iterations
            ):
                # Reduce the step size.
                alpha *= self._contraction_factor

                # Look closer down the line.
                newx = manifold.retraction(x, alpha * d)
                newf = objective(newx)

                cost_evaluations += 1
        # -----------------------------

        # Alby: uncomment back
        # if newf > f0:
        #     alpha = 0
        #     newx = x

        step_size = alpha * norm_d



        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about step_size but about alpha. This is
        # the reason why this line search is not invariant under rescaling of
        # the search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = 10 * alpha # Modified by Alby: used to be 1 * alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 100 * alpha # Modified by Alby: used to be 2 * alpha

        # ## ------- Introduced by Alby 
        # if alpha <= 1e-7: 
        #     self._oldalpha = None
        #     print("Resetting _old_alpha. Alpha = %1.5e"%(alpha))
        # ## -------------------------
        self._oldalpha = None
        # print(alpha)

        return step_size, newx


class myGPUAdaptiveLineSearcher:
    """GPU-accelerated adaptive line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        sufficient_decrease=0.5,
        max_iterations=10,
        initial_step_size=1,
        rank=0,
        device='cuda'
    ):
        self._contraction_factor = contraction_factor
        self._sufficient_decrease = sufficient_decrease
        self._max_iterations = max_iterations
        self._initial_step_size = initial_step_size
        self._rank = rank
        self._oldalpha = None
        self._device = device

    def search(self, objective, manifold, x, d, f0, df0):
        x_torch = self._to_torch_tuple(x)
        d_torch = self._to_torch_tuple(d)

        norm_d = self._manifold_norm(manifold, x_torch, d_torch)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self._initial_step_size / norm_d
        alpha = float(alpha)

        try:
            newx_torch = self._manifold_retraction(manifold, x_torch, d_torch, alpha)
            newx = self._to_numpy_tuple(newx_torch)
        except Exception as e:
            if self._rank == 0: print(f"Error during retraction: {e}. Reducing step size.")
            alpha *= self._contraction_factor
            newx_torch = self._manifold_retraction(manifold, x_torch, d_torch, alpha)
            newx = self._to_numpy_tuple(newx_torch)

        newf = objective(newx)
        cost_evaluations = 0

        while (
            newf > f0 + self._sufficient_decrease * alpha * df0
            and cost_evaluations <= self._max_iterations
        ):
            # Reduce the step size.
            alpha *= self._contraction_factor

            # Look closer down the line.
            try:
                newx_torch = self._manifold_retraction(manifold, x_torch, d_torch, alpha)
                newx = self._to_numpy_tuple(newx_torch)
            except Exception as e:
                if self._rank == 0: print(f"Error in retraction: {e}")
                alpha *= self._contraction_factor
                cost_evaluations += 1
                continue

            newf = objective(newx)
            cost_evaluations += 1

        # ----- Added by Alby --------
        if alpha <= 1e-12: 
            if self._rank == 0: print("Attention: allowing for cost function to increase by 1 percent")
            alpha = float(self._initial_step_size / norm_d)
            self._oldalpha = alpha

            newx_torch = self._manifold_retraction(manifold, x_torch, d_torch, alpha)
            newx = self._to_numpy_tuple(newx_torch)
            newf = objective(newx)
            cost_evaluations = 1

            while (
                newf > 1.01*f0 and cost_evaluations <= self._max_iterations
            ):
                # Reduce the step size.
                alpha *= self._contraction_factor

                # Look closer down the line.
                newx_torch = self._manifold_retraction(manifold, x_torch, d_torch, alpha)
                newx = self._to_numpy_tuple(newx_torch)
                newf = objective(newx)

                cost_evaluations += 1
        # -----------------------------

        step_size = alpha * norm_d

        if cost_evaluations == 2:
            self._oldalpha = 10 * alpha # Modified by Alby: used to be 1 * alpha
        else:
            self._oldalpha = 100 * alpha # Modified by Alby: used to be 2 * alpha

        self._oldalpha = None

        return step_size, newx
    
    def _to_torch_tuple(self, x_tuple):
        """Convert NumPy tuple to PyTorch tuple on the correct device."""
        if isinstance(x_tuple, (list, tuple)):
            return tuple(self._to_torch_tuple(xi) for xi in x_tuple)
        else:
            return torch.tensor(x_tuple, device=self._device)

    def _to_numpy_tuple(self, x_torch):
        """Convert PyTorch tuple back to NumPy tuple."""
        if isinstance(x_torch, (list, tuple)):
            return tuple(self._to_numpy_tuple(xi) for xi in x_torch)
        else:
            return x_torch.cpu().numpy()
        
    def _manifold_norm(self, manifold, x, d):
        """Compute manifold norm using PyTorch."""
        # Product manifolds
        if hasattr(manifold, "manifolds"):
            total_norm = 0.0
            for i, submanifold in enumerate(manifold.manifolds):
                total_norm += self._gpu_norm(submanifold, x[i], d[i]) ** 2
            return torch.sqrt(total_norm)
        else:
            return self._gpu_norm(manifold, x, d)
        
    def _gpu_norm(self, manifold, x, d):
        """Norm on specific manifolds using PyTorch."""
        if manifold.__class__.__name__ == "Stiefel":
            return torch.norm(d)
        elif manifold.__class__.__name__ == "Grassmann":
            return torch.norm(d)
        elif manifold.__class__.__name__ == "Euclidean":
            return torch.norm(d)
        else:
            return torch.tensor(manifold.norm(x.cpu().numpy(), d.cpu().numpy()), device=self._device)
    
    def _manifold_retraction(self, manifold, x, d, alpha):
        """Compute manifold retraction using PyTorch."""
        # Product manifolds
        if hasattr(manifold, "manifolds"):
            return tuple(self._gpu_retraction(
                manifold.manifolds[i], x[i], d[i], alpha)
                for i in range(len(manifold.manifolds))
            )
        else:
            return self._gpu_retraction(manifold, x, d, alpha)

    def _gpu_retraction(self, manifold, x, d, alpha):
        """Retraction for specific manifolds using PyTorch."""
        if manifold.__class__.__name__ == "Stiefel":
            y = x + alpha * d
            q, _ = torch.linalg.qr(y)
            return q
        elif manifold.__class__.__name__ == "Grassmann":
            y = x + alpha * d
            q, _ = torch.linalg.qr(y)
            return q
        elif manifold.__class__.__name__ == "Euclidean":
            return x + alpha * d
        else:
            print(manifold.__class__.__name__)
            return torch.tensor(
                manifold.retraction(x.cpu().numpy(), (alpha * d).cpu().numpy()),
                device=self._device
            )