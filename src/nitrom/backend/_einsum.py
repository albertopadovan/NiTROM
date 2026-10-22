"""Cached NumPy contractions, retaining the original BLAS plans and thresholds."""

import importlib


def _resolve_c_einsum(xp):
    """Bind numpy's raw C einsum, bypassing the Python dispatcher.

    ``numpy.einsum`` runs an array-function dispatcher and kwargs validation
    before reaching the C loop.  :meth:`Backend.einsum` calls it in a tight loop
    where that prologue is a visible fraction of the contraction, so prefer the
    C entry point and fall back to the public API if numpy ever moves it.
    """
    for mod_name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            return importlib.import_module(mod_name).c_einsum
        except (ImportError, AttributeError):
            continue
    return xp.einsum


class NumpyEinsum:
    """Compile and replay contractions without repeated NumPy path parsing."""

    #: Naive-contraction cost above which compiling a plan pays off.  Below it
    #: the replay's own transpose/reshape/``dot`` calls cost more than letting
    #: the C loop run once.
    #:
    #: Replay moved the real crossover well below this: measured on Sapphire
    #: Rapids, ``'abc,db,dc->da'`` at ``r = 8``, batch 7 (cost 3.6e3) already
    #: favours the plan, and at ``r = 12``, batch 7 it wins 6.6 us to 13.7 us.
    #: The bound is deliberately left where it was anyway.  Decomposing a
    #: contraction into BLAS reassociates its sums, so lowering it perturbs
    #: results at the 1e-16 level, and for the ``r = 50`` models this library is
    #: trained at it buys nothing -- every hot kernel is already far above the
    #: line.  Lower it only alongside a small-``r`` benchmark that shows the win.
    _EINSUM_OPTIMIZE_MIN_COST = 20_000

    #: Memory limit (elements) of the intermediates ``einsum_path`` may form.
    #: numpy's default is the size of the largest operand, which for the
    #: polynomial RHS ``'abc,db,dc->da'`` (H of r^3 entries) forbids the batch
    #: outer product z (x) z (batch x r^2) as soon as batch > r -- numpy then
    #: falls back to one naive three-operand loop, ~40x slower (r = 50: 0.23 ms
    #: at batch 45, 10 ms at batch 90).  A generous fixed cap keeps the BLAS path.
    _EINSUM_MEMORY_LIMIT = 100_000_000

    #: Cap on :attr:`_einsum_paths`.  Keys include operand shapes, so a caller
    #: that varies batch sizes freely could otherwise grow the cache without
    #: bound; the hot equations are few, so a flush is cheap and rare.
    _EINSUM_CACHE_MAX = 512

    @staticmethod
    def _einsum_naive_cost(equation, operands):
        """Cost of the *unoptimized* contraction.

        ``numpy.einsum`` without a path runs one C loop over the product of
        every distinct index dimension, so that product is the cost to beat.

        This only steers a performance choice, so every ``zip`` here is
        deliberately non-strict: a subscript that does not line up with its
        operand should make the estimate wrong, never raise.
        """
        lhs = equation.split("->")[0].replace(" ", "")
        dims = {}
        broadcast = 1
        for subscript, op in zip(lhs.split(","), operands, strict=False):
            shape = getattr(op, "shape", ())
            if "..." in subscript:
                head, tail = subscript.split("...")
                n_ell = len(shape) - len(head) - len(tail)
                size = 1
                for s in shape[len(head) : len(head) + n_ell]:
                    size *= s
                broadcast = max(broadcast, size)
                labelled = [
                    *zip(head, shape, strict=False),
                    *zip(tail, shape[len(head) + n_ell :], strict=False),
                ]
            else:
                labelled = zip(subscript, shape, strict=False)
            for c, s in labelled:
                dims[c] = s
        cost = broadcast
        for s in dims.values():
            cost *= s
        return cost

    def einsum(self, equation, *operands):
        """Contract using cached BLAS plans above the naive-cost threshold.

        Cache by equation and operand shapes, compiling transposes, reshapes,
        and matrix products once. Small contractions use NumPy's C einsum;
        unsupported plans fall back to NumPy's optimized public API.
        """
        try:
            # A list comprehension, not a generator: this key is built on every
            # contraction and at small r it is a visible fraction of one.
            key = (equation, *[op.shape for op in operands])
            plan = self._einsum_paths[key]
        except AttributeError:  # an operand without .shape -- not worth caching
            return self.xp.einsum(equation, *operands)
        except KeyError:
            cost = self._einsum_naive_cost(equation, operands)
            plan = None
            if cost >= self._EINSUM_OPTIMIZE_MIN_COST:
                plan = self._compile_einsum_plan(equation, operands)
                if plan is None:
                    # Compiling can fail on a subscript the shape algebra
                    # below does not model, or on a numpy whose einsum_path
                    # internals have moved.  Either way the contraction is
                    # still far too big for the naive C loop -- letting numpy
                    # optimize it costs a per-call re-parse but stays an order
                    # of magnitude ahead -- so degrade to that, never to no
                    # optimization at all.
                    plan = self._numpy_einsum_path(equation, operands)
            if len(self._einsum_paths) >= self._EINSUM_CACHE_MAX:
                self._einsum_paths.clear()
            self._einsum_paths[key] = plan

        if plan is None:
            return self._c_einsum(equation, *operands)
        if plan.__class__ is not tuple:  # a numpy path, not a compiled plan
            return self.xp.einsum(equation, *operands, optimize=plan)

        ops = list(operands)
        for inds, arg, perm in plan:
            tmp = [ops.pop(i) for i in inds]
            if arg.__class__ is str:
                ops.append(self._c_einsum(arg, *tmp))
                continue
            axes_a, shape_a, axes_b, shape_b, out_shape = arg
            a, b = tmp
            if axes_a is not None:
                a = a.transpose(axes_a)
            if axes_b is not None:
                b = b.transpose(axes_b)
            view = self.xp.dot(a.reshape(shape_a), b.reshape(shape_b))
            view = view.reshape(out_shape)
            # ``transpose`` leaves a non-contiguous view, which is what numpy's
            # own optimized einsum hands back too (it closes with ``asanyarray``
            # at ``order='K'``).  The buffer underneath is this step's fresh
            # ``dot`` output, so nothing else aliases it.
            ops.append(view if perm is None else view.transpose(perm))
        return ops[0]

    def _compile_einsum_plan(self, equation, operands):
        """Compile ``equation`` into a replayable list of contraction steps.

        Each step is ``(inds, arg, perm)``: ``inds`` are the operand slots it
        consumes (reverse-sorted, so popping them in order is safe), and ``arg``
        is either an einsum subscript string -- evaluated by the C loop, for the
        steps numpy judged ineligible for BLAS -- or the pre-solved
        ``tensordot`` decomposition ``(axes_a, shape_a, axes_b, shape_b,
        out_shape)``.  ``perm`` is a trailing axis permutation, or ``None``.

        Pre-solving the tensordot matters as much as caching the path: at these
        sizes ``numpy.tensordot`` spends more time deriving its transposes and
        reshapes than BLAS spends on the multiply.  Every shape is known at
        compile time, so all of it is hoisted out of the hot loop and only
        ``transpose``/``reshape``/``dot`` remain.

        Returns ``None`` when no plan can be built, so the caller falls back.
        """
        try:
            _, contraction_list = self.xp.einsum_path(
                equation,
                *operands,
                optimize=("optimal", self._EINSUM_MEMORY_LIMIT),
                einsum_call=True,
            )
        except TypeError:
            # ``einsum_call`` is numpy-internal; if it ever goes away, degrade
            # to the naive contraction rather than failing.
            return None

        shapes = [tuple(op.shape) for op in operands]
        steps = []
        try:
            for entry in contraction_list:
                # numpy <= 2.3 reports (inds, idx_rm, einsum_str, remaining,
                # blas); 2.4 dropped the derived fields and reports
                # (inds, einsum_str, remaining).  Take the two that are always
                # there and re-derive the rest, so the plan does not silently
                # stop compiling -- and the hot kernels do not silently drop to
                # the naive C loop -- the next time numpy reshuffles this.
                if len(entry) == 5:
                    inds, idx_rm, einsum_str, _remaining, blas = entry
                elif len(entry) == 3:
                    inds, einsum_str, _remaining = entry
                    idx_rm, blas = None, None
                else:
                    return None

                if "..." in einsum_str:
                    return None  # broadcasting: leave the shape algebra to numpy
                in_str, out_str = einsum_str.split("->")
                in_subs = in_str.split(",")
                tmp_shapes = [shapes.pop(i) for i in inds]

                dims = {}
                for sub, shape in zip(in_subs, tmp_shapes, strict=True):
                    for c, s in zip(sub, shape, strict=True):
                        dims[c] = s
                shapes.append(tuple(dims[c] for c in out_str))

                if idx_rm is None:
                    idx_rm = set("".join(in_subs)) - set(out_str)
                if blas is None:
                    blas = self._can_tensordot(in_subs, out_str, idx_rm)

                if not blas:
                    steps.append((tuple(inds), einsum_str, None))
                    continue

                left, right = in_subs
                # numpy pairs the contracted axes by sorted label; match it
                # exactly so the plan reproduces its result bit for bit.
                removed = sorted(idx_rm)
                arg = self._plan_tensordot(
                    tmp_shapes[0],
                    tmp_shapes[1],
                    [left.index(s) for s in removed],
                    [right.index(s) for s in removed],
                )
                tensor_result = left + right
                for s in idx_rm:
                    tensor_result = tensor_result.replace(s, "")
                perm = (
                    tuple(tensor_result.index(c) for c in out_str)
                    if tensor_result != out_str
                    else None
                )
                steps.append((tuple(inds), arg, perm))
        except (ValueError, IndexError, KeyError):
            # Any subscript this shape algebra does not model (repeated or
            # unpaired labels in a layout we did not anticipate) is a reason to
            # hand the contraction back to numpy, never to fail the solve.
            return None
        return tuple(steps)

    def _numpy_einsum_path(self, equation, operands):
        """numpy's own contraction path for ``equation``, or ``None``."""
        try:
            return self.xp.einsum_path(
                equation, *operands, optimize=("optimal", self._EINSUM_MEMORY_LIMIT)
            )[0]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _can_tensordot(in_subs, out_str, idx_rm):
        """Whether a contraction step reduces to a single ``tensordot``.

        Needed because numpy 2.4 no longer reports its own BLAS verdict.  The
        conditions are the classical ones: exactly two operands, at least one
        summed label, no label repeated inside an operand (that is a diagonal,
        not a product), every summed label present in both operands and absent
        from the output, and every surviving label kept.

        The shared labels must be *exactly* the summed ones.  A label carried by
        both operands and by the output is a batch index -- ``'Ibc,Ic->Ib'``,
        which is what an ellipsis over a batched contraction expands to -- and
        ``tensordot`` cannot express it: it would build the full outer product
        over that axis instead of pairing it up.

        Anything else is handed to the C loop, so a false negative only costs
        speed.
        """
        if len(in_subs) != 2 or not idx_rm:
            return False
        left, right = in_subs
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            return False
        if set(left) & set(right) != set(idx_rm):
            return False
        for s in idx_rm:
            if s in out_str:
                return False
        return set(left) | set(right) == set(out_str) | set(idx_rm)

    @staticmethod
    def _plan_tensordot(shape_a, shape_b, axes_a, axes_b):
        """Pre-solve ``numpy.tensordot``'s shape algebra for known shapes.

        Mirrors the body of :func:`numpy.tensordot`: the contracted axes move to
        the end of ``a`` and the front of ``b``, both operands collapse to 2-D,
        and the matrix product is reshaped back.  A permutation that is already
        the identity is returned as ``None`` so the replay can skip the call.
        """
        nda, ndb = len(shape_a), len(shape_b)
        notin_a = [k for k in range(nda) if k not in axes_a]
        notin_b = [k for k in range(ndb) if k not in axes_b]

        n_contract = 1
        for k in axes_a:
            n_contract *= shape_a[k]
        rows = 1
        for k in notin_a:
            rows *= shape_a[k]
        cols = 1
        for k in notin_b:
            cols *= shape_b[k]

        newaxes_a = tuple(notin_a + list(axes_a))
        newaxes_b = tuple(list(axes_b) + notin_b)
        out_shape = tuple([shape_a[k] for k in notin_a] + [shape_b[k] for k in notin_b])
        return (
            None if newaxes_a == tuple(range(nda)) else newaxes_a,
            (rows, n_contract),
            None if newaxes_b == tuple(range(ndb)) else newaxes_b,
            (n_contract, cols),
            out_shape,
        )
