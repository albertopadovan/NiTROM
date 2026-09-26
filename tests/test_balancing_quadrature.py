"""Non-uniform time quadrature in ``compute_data_driven_balancing``.

* the helpers: composite Gauss-Legendre and local Lagrange interpolation are
  exact where they should be;
* regression: on a uniform grid, the quadrature path with nodes at the sample
  offsets and weights ``dt`` reproduces the default path to round-off;
* convergence: on a linear system, balancing from unevenly sampled data (fine
  early, coarse late) with a composite Gauss-Legendre rule matches balancing
  from a much finer uniform sampling far better than the coarse uniform data
  with the rectangle rule does.
"""

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from nitrom.backend import set_backend
from nitrom.utils import (
    composite_gauss_legendre,
    compute_data_driven_balancing,
    lagrange_stencils,
)


class _Pool:
    def __init__(self, X, time):
        self.X, self.time = X, time
        self.world_size, self.rank, self.comm = 1, 0, None
        self.N = X.shape[1]


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")
    yield
    set_backend("torch")


def test_composite_gauss_legendre_exact():
    breaks = np.array([0.0, 0.1, 0.35, 2.0, 5.0])
    x, w = composite_gauss_legendre(breaks, 4)
    for deg in range(8):  # exact to degree 2n - 1 on every panel
        assert abs(np.sum(w * x**deg) - 5.0**(deg + 1)/(deg + 1)) <= 1e-12*5.0**(deg + 1)
    assert np.all(np.diff(x) > 0)


def test_lagrange_stencils_exact_on_nonuniform_grid():
    t = np.concatenate([np.arange(0, 2, 0.02), np.arange(2, 10.01, 0.2)])
    te = np.random.default_rng(0).uniform(0, 10, 300)
    start, W = lagrange_stencils(t, te, 6)
    for deg in range(6):  # degree < order is reproduced exactly
        f = t**deg
        approx = np.array([W[e] @ f[start[e]:start[e] + 6] for e in range(len(te))])
        assert np.abs(approx - te**deg).max() <= 1e-9 * max(1.0, 10.0**deg)
    # a node on a sample returns that sample
    s, Wn = lagrange_stencils(t, t[[0, 57, 120]], 6)
    for e, i in enumerate([0, 57, 120]):
        assert Wn[e, i - s[e]] == 1.0 and np.count_nonzero(Wn[e]) == 1


def _families(rng, n_traj, N, nt):
    X = rng.standard_normal((n_traj, N, nt)) * np.exp(-0.05*np.arange(nt))
    return X, [[0, 1], [2]]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_quadrature_path_reproduces_uniform_path(backend):
    set_backend(backend)
    rng = np.random.default_rng(1)
    X, fams = _families(rng, 3, 40, 30)
    dt = 0.1
    time = dt*np.arange(30)
    m, q = 4, 3
    if backend == "torch":
        Xb, tb = torch.as_tensor(X), torch.as_tensor(time)
    else:
        Xb, tb = X, time
    ref = compute_data_driven_balancing(_Pool(Xb, tb), n_checkpoints=m,
                                        checkpoint_stride=q, families=fams)
    L = 30 - (m - 1)*q
    quad = dict(nodes=dt*np.arange(L), weights=np.full(L, dt),
                checkpoint_times=time[[q*i for i in range(m)]], interp_order=6)
    new = compute_data_driven_balancing(_Pool(Xb, tb), families=fams,
                                        quadrature=quad)
    to_np = (lambda a: a.numpy()) if backend == "torch" else np.asarray
    Sig0, Sig1 = to_np(ref[2]), to_np(new[2])
    assert np.allclose(Sig1, Sig0, rtol=1e-12, atol=0)
    r = int((Sig0 > 1e-8*Sig0[0]).sum())
    for a, b in ((ref[0], new[0]), (ref[1], new[1])):   # Phi, Psi up to sign
        a, b = to_np(a)[:, :r], to_np(b)[:, :r]
        sgn = np.sign(np.sum(a*b, axis=0))
        assert np.abs(a - b*sgn).max() <= 1e-9*np.abs(a).max()


def test_nonuniform_quadrature_converges_on_linear_system():
    rng = np.random.default_rng(2)
    N = 30
    # stable, non-normal A with a fast and a slow time scale
    A = -np.diag(np.concatenate([np.full(10, 20.0), np.linspace(0.3, 2.0, 20)]))
    A += 3.0*np.triu(rng.standard_normal((N, N)), 1)/np.sqrt(N)
    x0s = rng.standard_normal((N, 3))
    T_end, t_ck = 8.0, [0.0, 0.5, 1.0, 1.5]
    T_w = T_end - t_ck[-1]

    def sample(times):
        E = [sla.expm(A*t) for t in times]
        return np.stack([np.stack([Et @ x0 for Et in E], axis=1)
                         for x0 in x0s.T])

    # reference: very fine uniform grid, rectangle rule
    dtf = 0.002
    tf = dtf*np.arange(int(round(T_end/dtf)) + 1)
    stride = int(round(0.5/dtf))
    ref = compute_data_driven_balancing(
        _Pool(sample(tf), tf), n_checkpoints=4, checkpoint_stride=stride)

    # uneven data: 0.01 up to t = 2, then 0.25; composite GL rule
    tn = np.concatenate([np.arange(0, 2, 0.01), np.arange(2, T_end + 1e-9, 0.25)])
    breaks = np.concatenate([np.arange(0, 1.0, 0.05), np.arange(1.0, T_w + 1e-9, 0.5)])
    nodes, weights = composite_gauss_legendre(breaks, 4)
    quad = dict(nodes=nodes, weights=weights, checkpoint_times=t_ck,
                interp_order=6)
    new = compute_data_driven_balancing(_Pool(sample(tn), tn), quadrature=quad)

    # coarse uniform data (0.25) with the rectangle rule, for contrast
    tc = 0.25*np.arange(int(round(T_end/0.25)) + 1)
    coarse = compute_data_driven_balancing(
        _Pool(sample(tc), tc), n_checkpoints=4, checkpoint_stride=2)

    k = 6
    err_new = np.abs(new[2][:k] - ref[2][:k]).max()/ref[2][0]
    err_coarse = np.abs(coarse[2][:k] - ref[2][:k]).max()/ref[2][0]
    assert err_new < 2e-2
    assert err_new < 0.25*err_coarse


@pytest.mark.parametrize("use_quadrature", [False, True])
def test_qr_preprojection_is_exact(use_quadrature):
    """Balancing the untruncated QR coefficients of the snapshots and lifting
    gives the full-size balancing: only inner products of snapshot
    combinations enter, and those are preserved by an orthonormal basis."""
    rng = np.random.default_rng(3)
    n_traj, N, nt = 3, 200, 25
    X = rng.standard_normal((n_traj, N, nt)) * np.exp(-0.1*np.arange(nt))
    if use_quadrature:
        time = np.concatenate([0.05*np.arange(10), 0.5 + 0.2*np.arange(nt - 10)])
        L = 10
        kw = dict(quadrature=dict(nodes=0.2*np.arange(L), weights=np.full(L, 0.2),
                                  checkpoint_times=[0.0, 0.1, 0.25, 0.9, 1.5],
                                  interp_order=4))
    else:
        time = 0.2*np.arange(nt)
        kw = dict(n_checkpoints=4, checkpoint_stride=3)
    fams = [[0, 1], [2]]
    Phi0, Psi0, S0, _ = compute_data_driven_balancing(
        _Pool(X, time), families=fams, **kw)

    Y = np.concatenate(list(X), axis=1)
    U, Rm = np.linalg.qr(Y)
    C = np.stack([Rm[:, k*nt:(k + 1)*nt] for k in range(n_traj)])
    Phic, Psic, S1, _ = compute_data_driven_balancing(
        _Pool(C, time), families=fams, **kw)

    assert np.allclose(S1, S0, rtol=1e-11, atol=0)
    r = int((S0 > 1e-8*S0[0]).sum())
    for a, b in ((Phi0, U @ Phic), (Psi0, U @ Psic)):
        a, b = a[:, :r], b[:, :r]
        sgn = np.sign(np.sum(a*b, axis=0))
        assert np.abs(a - b*sgn).max() <= 1e-8*np.abs(a).max()
