r"""Matrix-manifold optimization primitives and a Riemannian L-BFGS.

Supports the **Grassmann** and **Stiefel** manifolds under the *embedded*
(Euclidean/Frobenius) metric, plus the trivial ``"euclidean"`` manifold on which
every operation reduces to the ordinary vector-space one.  Because Euclidean is
handled as a (trivial) manifold, a mixed list of parameters -- some on matrix
manifolds, some free -- is treated uniformly as a **product manifold**.

The four geometric ingredients an optimizer needs:

* :func:`project` -- Euclidean gradient :math:`G \mapsto` Riemannian gradient
  (orthogonal projection onto the tangent space :math:`T_X\mathcal{M}`);
* :func:`to_manifold` / :func:`retract` -- move a point back onto the manifold
  (QR-based retraction), used to take a step;
* :func:`transport` -- **vector transport** of a tangent vector from one tangent
  space to another (by projection, a valid vector transport for embedded
  submanifolds);
* :func:`inner` -- the Riemannian metric (Frobenius) used in every inner product.

:func:`riemannian_lbfgs` is a self-contained L-BFGS that carries its curvature
memory correctly across tangent spaces: every stored ``(s_k, y_k)`` pair *and*
the search direction are vector-transported into the current tangent space
before the two-loop recursion, ``s_k``/``y_k`` are built from a transported step
and a transported gradient, and all inner products use the Riemannian metric.
The step length comes from :func:`strong_wolfe_line_search`, a Riemannian
strong-Wolfe search along the retraction.
"""

from ..backend import get_backend


def project(X, G, manifold):
    r"""Project the Euclidean gradient ``G`` onto :math:`T_X\mathcal{M}`.

    This is the Riemannian gradient under the embedded metric.
    Grassmann: :math:`G - X X^\top G`.  Stiefel:
    :math:`G - X\,\mathrm{sym}(X^\top G)`.  Euclidean: identity.
    """
    if manifold == "grassmann":
        return G - X @ (X.T @ G)
    if manifold == "stiefel":
        XtG = X.T @ G
        return G - X @ (0.5 * (XtG + XtG.T))
    return G


def to_manifold(X, manifold):
    """Retract an ambient point ``X`` onto the manifold (QR factor with a sign
    fix so the map is smooth); identity on the Euclidean manifold."""
    if manifold == "euclidean":
        return X
    bkend = get_backend()
    Q, R = bkend.qr(X)
    ph = bkend.sign(bkend.diagonal(R))
    ph = bkend.where(ph == 0, bkend.ones_like(ph), ph)
    return Q * ph[None, :]


def retract(X, xi, manifold):
    r"""Retraction :math:`R_X(\xi)`: step from ``X`` along the tangent vector
    ``xi`` and return to the manifold.  QR-based for Stiefel/Grassmann,
    :math:`X + \xi` for Euclidean."""
    return to_manifold(X + xi, manifold)


def transport(X_new, xi, manifold):
    r"""Vector transport of ``xi`` into :math:`T_{X_\text{new}}\mathcal{M}`.

    Uses transport *by projection* (``xi`` projected onto the new tangent
    space), which is a valid vector transport for embedded submanifolds.
    """
    return project(X_new, xi, manifold)


def inner(U, V):
    """Riemannian (embedded/Frobenius) inner product of two tangent vectors."""
    return float((U * V).sum())


def _list_inner(Us, Vs):
    return sum(inner(u, v) for u, v in zip(Us, Vs, strict=True))


def strong_wolfe_line_search(
    cost_fn, rgrad_fn, x, d, manifolds, f0, g0,
    c1=1e-4, c2=0.9, max_iter=25, t_init=1.0,
):
    r"""Riemannian strong-Wolfe line search along the retraction.

    With :math:`\varphi(t) = f(R_x(t\,d))` and the (transported) directional
    derivative :math:`\varphi'(t) = \langle \mathrm{grad}\,f(y_t),\,
    \mathcal{T}_{x\to y_t}(d)\rangle` where :math:`y_t = R_x(t\,d)`, find a step
    ``t`` satisfying the strong-Wolfe conditions

    .. math::

        \varphi(t) \le \varphi(0) + c_1 t\,\varphi'(0),
        \qquad |\varphi'(t)| \le -c_2\,\varphi'(0).

    Implemented as the standard bracket-then-``zoom`` scheme (Nocedal & Wright,
    Alg. 3.5/3.6), with every trial point produced by the retraction and every
    slope by the vector-transported search direction.  The gradient (often the
    expensive part -- e.g. an adjoint solve) is evaluated **lazily**: a trial's
    cost is computed first, and its gradient only when the trial passes the
    Armijo test, so trials rejected for insufficient decrease cost no gradient.

    :param cost_fn: ``xs -> scalar`` cost on the product manifold
    :param rgrad_fn: ``xs -> list`` Riemannian gradient
    :param x: current point (list of factors)
    :param d: search direction (list of tangent factors)
    :param manifolds: per-factor manifold list
    :param f0: ``cost_fn(x)`` (the value :math:`\varphi(0)`)
    :param g0: ``rgrad_fn(x)`` (used for :math:`\varphi'(0)`)
    :returns: ``(t, y, f, g)`` -- step, new point, its cost and Riemannian
        gradient -- when both strong-Wolfe conditions are met; otherwise
        ``None`` (the caller can fall back to Armijo backtracking).
    """
    dphi0 = _list_inner(g0, d)
    if dphi0 >= 0.0:
        return None

    def eval_cost(t):  # cheap probe: retraction + cost only
        y = [retract(xi, t * di, m)
             for xi, di, m in zip(x, d, manifolds, strict=True)]
        f = cost_fn(y)
        return y, f

    def eval_grad(y):  # expensive probe: gradient + transported slope at y
        g = rgrad_fn(y)
        dphi = sum(inner(gi, transport(yi, di, m))
                   for gi, yi, di, m in zip(g, y, d, manifolds, strict=True))
        return g, dphi

    def trial_step(lo, f_lo, dphi_lo, hi, f_hi):
        # Minimizer of the quadratic matching phi(lo), phi'(lo), phi(hi); safe-
        # guarded to stay well inside the bracket, else bisect.  Interpolation
        # (vs plain bisection) is what keeps the step count low on ill-
        # conditioned / non-smooth costs.
        width = hi - lo
        denom = 2.0 * (f_hi - f_lo - dphi_lo * width)
        t = lo - dphi_lo * width * width / denom if denom > 0.0 else None
        lo_b, hi_b = min(lo, hi), max(lo, hi)
        margin = 0.1 * abs(width)
        if t is None or not (lo_b + margin <= t <= hi_b - margin):
            t = 0.5 * (lo + hi)
        return t

    def zoom(lo, f_lo, dphi_lo, hi, f_hi):
        for _ in range(max_iter):
            t = trial_step(lo, f_lo, dphi_lo, hi, f_hi)
            y, f = eval_cost(t)
            if f > f0 + c1 * t * dphi0 or f >= f_lo:
                hi, f_hi = t, f  # rejected for insufficient decrease -> no gradient
            else:
                g, dphi = eval_grad(y)  # candidate: now the gradient is needed
                if abs(dphi) <= -c2 * dphi0:  # strong Wolfe satisfied
                    return t, y, f, g
                if dphi * (hi - lo) >= 0.0:
                    hi, f_hi = lo, f_lo
                lo, f_lo, dphi_lo = t, f, dphi
        return None

    t_prev, f_prev, dphi_prev = 0.0, f0, dphi0
    t = t_init
    for i in range(max_iter):
        y, f = eval_cost(t)
        if f > f0 + c1 * t * dphi0 or (i > 0 and f >= f_prev):
            return zoom(t_prev, f_prev, dphi_prev, t, f)
        g, dphi = eval_grad(y)  # Armijo held -> need the slope for the curvature test
        if abs(dphi) <= -c2 * dphi0:  # strong Wolfe satisfied
            return t, y, f, g
        if dphi >= 0.0:
            return zoom(t, f, dphi, t_prev, f_prev)
        t_prev, f_prev, dphi_prev = t, f, dphi
        t *= 2.0

    return None  # budget exhausted -> signal failure (caller falls back)


def _armijo_backtracking(
    cost_fn, rgrad_fn, x, d, manifolds, f0, g0, c1=1e-4, max_bt=40,
):
    r"""Riemannian Armijo backtracking along the retraction.

    Halve the step from ``t = 1`` until sufficient decrease
    :math:`\varphi(t) \le \varphi(0) + c_1 t\,\varphi'(0)` holds.  Fallback for
    :func:`strong_wolfe_line_search` on guarded / non-smooth costs where the
    curvature condition cannot be met.

    :returns: ``(t, y, f, g)`` or ``None`` if no decrease is found.
    """
    gd = _list_inner(g0, d)
    if gd >= 0.0:
        return None
    t = 1.0
    for _ in range(max_bt):
        y = [retract(xi, t * di, m)
             for xi, di, m in zip(x, d, manifolds, strict=True)]
        f = cost_fn(y)
        if f <= f0 + c1 * t * gd:
            return t, y, f, rgrad_fn(y)
        t *= 0.5
    return None


def _armijo_backtracking_increase(
    cost_fn, rgrad_fn, x, d, manifolds, f0, g0, max_bt=40,
):
    r"""Fallback backtracking allowing the cost function to increase by up to 1%."""
    norm_d = _list_inner(d, d) ** 0.5
    t = 1.0 / max(norm_d, 1.0)
    print(f"    [Armijo-Increase LineSearch] norm_d = {norm_d:.6e} | initial cost = {f0:.6e}")
    for bt in range(max_bt):
        y = [retract(xi, t * di, m)
             for xi, di, m in zip(x, d, manifolds, strict=True)]
        f = cost_fn(y)
        print(f"      [Armijo-Increase Trial] bt_iter = {bt} | t = {t:.6e} | cost = {f:.6e} (target <= {1.01 * f0:.6e})")
        if f <= 1.01 * f0:
            print("Attention: allowing for cost function to increase by up to 1 percent")
            return t, y, f, rgrad_fn(y)
        t *= 0.5
    return None


class SGDDirection:
    """Steepest descent: search direction is ``-lr * grad``.  The strong-Wolfe
    line search then rescales it, so ``lr`` is only a nominal initial step."""

    def __init__(self, lr=1.0):
        self.lr = lr

    def direction(self, xs, g, manifolds):
        return [-self.lr * gk for gk in g]

    def update(self, xs_new, d, t, g, g_new, manifolds):
        pass

    def reset(self):
        pass


class AdamDirection:
    """Riemannian Adam direction: a per-coordinate preconditioned step whose
    first/second moments are vector-transported to the new tangent space each
    iteration.  The strong-Wolfe line search sets the actual step length."""

    def __init__(self, lr=1.0, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = None
        self.v = None
        self.step = 0

    def direction(self, xs, g, manifolds):
        bkend = get_backend()
        if self.m is None:
            self.m = [bkend.zeros_like(gk) for gk in g]
            self.v = [bkend.zeros_like(gk) for gk in g]
        self.step += 1
        d = []
        for i, (xk, gk, man) in enumerate(zip(xs, g, manifolds, strict=True)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gk
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (gk * gk)
            mhat = self.m[i] / (1 - self.b1 ** self.step)
            vhat = self.v[i] / (1 - self.b2 ** self.step)
            dk = -self.lr * mhat / (bkend.sqrt(vhat) + self.eps)
            d.append(project(xk, dk, man))  # keep the step tangent
        return d

    def update(self, xs_new, d, t, g, g_new, manifolds):
        if self.m is None:  # reset() just cleared the moments -> nothing to move
            return
        bkend = get_backend()
        # Transport the first moment (a tangent vector) and the second moment (a
        # per-coordinate variance -> |.| keeps it non-negative) to T_{xs_new}.
        self.m = [transport(xn, mi, man)
                  for xn, mi, man in zip(xs_new, self.m, manifolds, strict=True)]
        self.v = [bkend.abs(transport(xn, vi, man))
                  for xn, vi, man in zip(xs_new, self.v, manifolds, strict=True)]

    def reset(self):
        self.m = None
        self.v = None
        self.step = 0


class LBFGSDirection:
    """L-BFGS two-loop recursion whose curvature memory ``(s_k, y_k)`` is
    vector-transported into the current tangent space every iteration."""

    def __init__(self, history_size=100):
        self.history_size = history_size
        self.S, self.Y, self.RHO = [], [], []

    def direction(self, xs, g, manifolds):
        S, Y, RHO = self.S, self.Y, self.RHO
        q = [gk.copy() for gk in g]
        alpha = [0.0] * len(S)
        for i in range(len(S) - 1, -1, -1):
            alpha[i] = RHO[i] * _list_inner(S[i], q)
            q = [qk - alpha[i] * yk for qk, yk in zip(q, Y[i], strict=True)]
        if S:
            yy = _list_inner(Y[-1], Y[-1])
            gamma = _list_inner(S[-1], Y[-1]) / yy if yy > 0 else 1.0
        else:
            gamma = 1.0 / max(_list_inner(g, g) ** 0.5, 1.0)
        r = [gamma * qk for qk in q]
        for i in range(len(S)):
            beta = RHO[i] * _list_inner(Y[i], r)
            r = [rk + (alpha[i] - beta) * sk for rk, sk in zip(r, S[i], strict=True)]
        return [-rk for rk in r]

    def update(self, xs_new, d, t, g, g_new, manifolds):
        s_k = [transport(xn, t * dk, m)
               for xn, dk, m in zip(xs_new, d, manifolds, strict=True)]
        Tg = [transport(xn, gk, m)
              for xn, gk, m in zip(xs_new, g, manifolds, strict=True)]
        y_k = [gnk - tgk for gnk, tgk in zip(g_new, Tg, strict=True)]
        # Transport the stored history into the new tangent space.
        self.S = [[transport(xn, sk, m)
                   for xn, sk, m in zip(xs_new, s, manifolds, strict=True)]
                  for s in self.S]
        self.Y = [[transport(xn, yk, m)
                   for xn, yk, m in zip(xs_new, y, manifolds, strict=True)]
                  for y in self.Y]
        sy = _list_inner(s_k, y_k)
        if sy > 1e-10:
            self.S.append(s_k)
            self.Y.append(y_k)
            self.RHO.append(1.0 / sy)
            if len(self.S) > self.history_size:
                self.S.pop(0)
                self.Y.pop(0)
                self.RHO.pop(0)

    def reset(self):
        self.S, self.Y, self.RHO = [], [], []


def riemannian_optimize(
    cost_fn, rgrad_fn, x0, manifolds, direction,
    max_iter=200, gtol=1e-10, ftol=1e-9, callback=None,
    allow_increase=False,
):
    r"""Generic first-order Riemannian optimizer on a product manifold.

    Every iteration: (1) the ``direction`` strategy proposes a search direction
    from the current Riemannian gradient; (2) a **strong-Wolfe line search with
    an Armijo backtracking fallback** (:func:`strong_wolfe_line_search`,
    :func:`_armijo_backtracking`) picks the step along the retraction; (3) the
    strategy's internal state is vector-transported into the new tangent space.
    The *same* line search is thus used for every optimizer -- L-BFGS, Adam, and
    SGD differ only in ``direction``.

    ``cost_fn(xs)`` / ``rgrad_fn(xs)`` map a product-manifold point (a list of
    factors) to the scalar cost / the Riemannian gradient (a list of tangent
    factors).  ``direction`` is an :class:`SGDDirection`, :class:`AdamDirection`,
    or :class:`LBFGSDirection` (anything with ``direction``/``update``/``reset``).

    :returns: ``(x_star, f_star)`` -- the best point found and its cost.
    """
    import numpy as np

    xs = [to_manifold(np.asarray(a, dtype=float).copy(), m)
          for a, m in zip(x0, manifolds, strict=True)]
    g = rgrad_fn(xs)
    fval = cost_fn(xs)
    fprev = fval
    best_xs = [a.copy() for a in xs]
    best_f = fval

    for it in range(max_iter):
        gnorm = _list_inner(g, g) ** 0.5
        if gnorm < gtol:
            break

        d = direction.direction(xs, g, manifolds)
        if _list_inner(g, d) >= 0.0:  # not a descent direction -> steepest descent
            d = [-gk for gk in g]
            direction.reset()

        # For fixed-step optimizers (Adam, SGD), skip line search and take the proposed direction directly (t=1.0)
        if hasattr(direction, "lr"):
            t = 1.0
            xs_new = [retract(xi, t * di, m) for xi, di, m in zip(xs, d, manifolds, strict=True)]
            f_new = cost_fn(xs_new)
            g_new = rgrad_fn(xs_new)
            ls = (t, xs_new, f_new, g_new)
        else:
            # Strong-Wolfe line search, Armijo backtracking as the fallback; both
            # return the new point with its cost and Riemannian gradient.
            ls = strong_wolfe_line_search(cost_fn, rgrad_fn, xs, d, manifolds, fval, g)
            if ls is None:
                ls = _armijo_backtracking(cost_fn, rgrad_fn, xs, d, manifolds, fval, g)
            if ls is None and allow_increase:
                ls = _armijo_backtracking_increase(cost_fn, rgrad_fn, xs, d, manifolds, fval, g)

        if ls is None:  # both line searches failed -> stop
            print("    [LineSearch Failed] Line search failed completely. Stopping optimization.")
            break
        t, xs_new, f_new, g_new = ls

        direction.update(xs_new, d, t, g, g_new, manifolds)

        xs, g, fval = xs_new, g_new, f_new
        if f_new < best_f:
            best_xs = [a.copy() for a in xs_new]
            best_f = f_new
        if callback is not None:
            callback(it, fval, gnorm)
        if 0.0 <= fprev - fval <= ftol * max(abs(fprev), 1.0):
            break
        fprev = fval

    return best_xs, best_f


def riemannian_lbfgs(
    cost_fn, rgrad_fn, x0, manifolds, max_iter=200, history_size=100,
    gtol=1e-10, ftol=1e-9, callback=None, allow_increase=False,
):
    r"""Riemannian L-BFGS: :func:`riemannian_optimize` with an
    :class:`LBFGSDirection` (vector-transported curvature memory).

    :returns: ``(x_star, f_star)`` -- the best point found and its cost.
    """
    return riemannian_optimize(
        cost_fn, rgrad_fn, x0, manifolds, LBFGSDirection(history_size),
        max_iter=max_iter, gtol=gtol, ftol=ftol, callback=callback,
        allow_increase=allow_increase,
    )
