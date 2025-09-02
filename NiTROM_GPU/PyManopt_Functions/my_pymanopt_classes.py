import numpy as np

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
            if self._rank == 0: print('Test 1')
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
                if self._rank == 0: print('Test 2')
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
