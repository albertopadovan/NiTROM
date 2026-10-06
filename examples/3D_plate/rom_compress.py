"""Lossless (0-error) compression of the 3D training trajectories, and the two
streaming operations that are the only places the full dimension N (~8e7)
appears afterwards.

Weighted snapshots q~ = W^{1/2} q', W = V/V_min (the energy inner product
becomes the Euclidean one), each trajectory scaled by 1/sqrt(Ebar_k), Ebar_k
its time-averaged energy (the airfoil's normalization; it only orders the
modes, the span is the same), stacked into

    Y = [s_0 Q~_0, s_1 Q~_1, ...]      (N x m, m = n_traj * n_t).

A tall-skinny QR, streamed over row chunks of Y on the GPU (Y = Q R, Q never
formed), followed by the SVD R = U_R S V^T, gives the POD of Y with QR
accuracy (no squared conditioning, unlike the snapshot Gram matrix):

    Y = U S V^T,   U = Q U_R = Y V S^{-1}   (N x m, orthonormal).

Only directions with S_i/S_0 below the float32 round-off of the stored data
are dropped (``tol``), so the coordinates x = U^T q~ reproduce every
snapshot exactly and ||x|| = ||q~||_2 = ||q'||_W.  The basis U is never
stored (it would be N x m: 390 GB in float64); it is applied implicitly
through the data, which stay on disk:

    lift(A)    = U A     = Y (V S^{-1} A)      one pass over the data
    project(G) = U^T G   = S^{-1} V^T (Y^T G)  one pass over the data

Everything downstream (POD/balanced bases, OpInf) lives in the r_l-dim
coordinates (r_l <= m).
"""

from __future__ import annotations

import os
import time as timer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 1_000_000          # rows of Y per GPU block (m = 600: 4.8 GB in float64)


def cell_volume_weights_from_snapshot(path):
    """W = V/V_min of every u, v, w DOF, from a snapshot's staggered grids."""
    vols = []
    with np.load(path) as s:
        for c in "uvw":
            x, y, z = (s[f"{a}{c}"][1:-1] for a in "xyz")
            vols.append((np.gradient(z)[:, None, None] * np.gradient(y)[None, :, None]
                         * np.gradient(x)[None, None, :]).ravel())
    V = np.concatenate(vols)
    return V/V.min()


class SnapshotData:
    """The stacked, weighted, scaled snapshot matrix Y, read lazily from the
    time-major float32 trajectory files (n_t, N)."""

    def __init__(self, data_dir, trajs, scale):
        meta = np.load(os.path.join(data_dir, "meta.npz"))
        self.time = meta["time"]
        self.baseflow = str(meta["baseflow"])
        self.trajs = list(trajs)
        self.Q = [np.load(os.path.join(data_dir, f"fluct_{k:03d}.npy"), mmap_mode="r")
                  for k in self.trajs]
        self.n_t = len(self.time)
        self.N = self.Q[0].shape[1]
        self.m = len(self.trajs)*self.n_t
        self.scale = np.asarray(scale, dtype=np.float64)
        self.sqrt_w = np.sqrt(cell_volume_weights_from_snapshot(self.baseflow))
        if self.sqrt_w.size != self.N:
            raise ValueError("weights do not match the trajectories")

    def chunks(self, size=CHUNK):
        for a in range(0, self.N, size):
            yield a, min(a + size, self.N)

    def block(self, a, b, xp):
        """Rows a:b of Y on the device, float64, (b - a, m)."""
        Yc = xp.empty((b - a, self.m))
        for j, Qk in enumerate(self.Q):
            Yc[:, j*self.n_t:(j + 1)*self.n_t] = xp.asarray(
                np.ascontiguousarray(Qk[:, a:b]).T, dtype=xp.float64)*self.scale[j]
        Yc *= xp.asarray(self.sqrt_w[a:b])[:, None]
        return Yc


def tsqr_R(data, xp, verbose=True):
    """R of the thin QR of Y, streamed over row blocks: R <- qr([R; Y_c])."""
    R = None
    t0 = timer.time()
    n_blk = -(-data.N//CHUNK)
    for i, (a, b) in enumerate(data.chunks()):
        Yc = data.block(a, b, xp)
        R = xp.linalg.qr(Yc if R is None else xp.concatenate([R, Yc]), mode="r")
        del Yc
        if verbose and (i % 10 == 0 or i == n_blk - 1):
            print(f"    TSQR block {i + 1}/{n_blk}  [{timer.time() - t0:.0f} s]", flush=True)
    return R


class Compressed:
    """The saved compression (compressed.npz): coordinates X (n_traj, r_l, n_t),
    S, V (m x r_l), and the implicit basis U = Y V S^{-1}."""

    def __init__(self, path):
        c = np.load(path, allow_pickle=True)
        self.path = path
        self.X, self.S, self.V = c["X"], c["S"], c["V"]
        self.time, self.params = c["time"], c["parameters"]
        self.energy = c["energy"]
        self.trajs = list(c["trajs"])
        self.data_dir, self.baseflow = str(c["data_dir"]), str(c["baseflow"])
        self.scale = c["scale"]
        self.r_l = self.S.size
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = SnapshotData(self.data_dir, self.trajs, self.scale)
        return self._data

    def lift(self, A, xp, out=None, verbose=True):
        """U A for A (r_l, b): weighted full states, host float32 (N, b)."""
        d = self.data
        B = xp.asarray(self.V @ (np.asarray(A)/self.S[:, None]))   # (m, b)
        if out is None:
            out = np.empty((d.N, A.shape[1]), dtype=np.float32)
        t0 = timer.time()
        for a, b in d.chunks():
            out[a:b] = (d.block(a, b, xp) @ B).get() if hasattr(B, "get") else d.block(a, b, xp) @ B
        if verbose:
            print(f"    lift of {A.shape[1]} states: {timer.time() - t0:.0f} s", flush=True)
        return out

    def project(self, G, xp, verbose=True):
        """U^T G for host G (N, b) (weighted vectors): (r_l, b) float64."""
        d = self.data
        acc = xp.zeros((d.m, G.shape[1]))
        t0 = timer.time()
        for a, b in d.chunks():
            acc += d.block(a, b, xp).T @ xp.asarray(G[a:b], dtype=xp.float64)
        acc = acc.get() if hasattr(acc, "get") else acc
        if verbose:
            print(f"    projection of {G.shape[1]} vectors: {timer.time() - t0:.0f} s", flush=True)
        return (self.V.T @ acc)/self.S[:, None]
