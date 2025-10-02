import torch

class Interp1D:
    '''
    1D linear interpolation for PyTorch tensors.
    '''
    def __init__(self, t, x, extrapolate=False):
        assert t.ndim == 1, "Input tensor t must be 1D."
        assert t.shape[0] == x.shape[1], "Input tensors t and x must match along the time dimension."
        self.t = t
        self.x = x
        self.extrapolate = extrapolate

    def __call__(self, tq):
        t, x = self.t, self.x
        idx = torch.searchsorted(t, tq, right=True).clamp(1, len(t) - 1)
        t0, t1 = t[idx - 1], t[idx]
        x0, x1 = x[:, idx - 1], x[:, idx]

        m = (tq - t0) / (t1 - t0)
        m = m.unsqueeze(0)
        xq = x0 + m * (x1 - x0)

        if not self.extrapolate:
            below = tq < t[0]
            above = tq > t[-1]

            if below.any():
                xq[below] = x[:,0].unsqueeze(1)
            if above.any():
                xq[above] = x[:, -1].unsqueeze(1)

        return xq