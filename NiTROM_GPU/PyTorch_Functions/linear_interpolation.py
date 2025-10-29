import torch

class Interp1D:
    '''
    1D linear interpolation for PyTorch tensors.
    '''
    def __init__(self, t, x, extrapolate=False):
        assert t.ndim == 1, "Input tensor t must be 1D."
        assert t.shape[0] == x.shape[-1], "t and x must match along the last (time) dimension."
        self.t = t
        self.x = x
        self.extrapolate = extrapolate

    def __call__(self, tq):
        t, x = self.t, self.x
        # Promote scalar tq to 1D for searchsorted/index_select
        scalar_input = False
        if tq.ndim == 0:
            tq = tq.unsqueeze(0)
            scalar_input = True

        idx = torch.searchsorted(t, tq, right=True).clamp(1, len(t) - 1)  # (Q,)
        t0, t1 = t[idx - 1], t[idx]                                        # (Q,)

        x0 = torch.index_select(x, dim=-1, index=idx - 1)
        x1 = torch.index_select(x, dim=-1, index=idx)

        m = (tq - t0) / (t1 - t0)                                          # (Q,)
        m = m.reshape(*([1] * (x0.ndim - 1)), -1)                          # (..., Q)

        xq = x0 + m * (x1 - x0)                                            # (..., Q)

        if not self.extrapolate:
            below = tq < t[0]
            above = tq > t[-1]
            if below.any():
                xq[..., below] = x[..., 0].unsqueeze(-1)
            if above.any():
                xq[..., above] = x[..., -1].unsqueeze(-1)

        return xq.squeeze(-1) if scalar_input else xq