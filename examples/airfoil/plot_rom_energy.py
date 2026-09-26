"""Energy vs time of balanced ROMs against the FOM, per impulse response.

E(t) = ||q~(t)||^2 = <q, q>_W, the perturbation energy of figure 10(b), on the
mesh the balancing was built on.  For each trajectory the panel shows the FOM,
its oblique projection Phi Psi^T q~ (the best the basis can do), and every
model given, forecast from z(0) = Psi^T q~(0) and decoded with
Phi (Psi^T Phi)^{-1} (which is Phi itself for a biorthogonal pair).

Training set by default; --data DIR plots the trajectories present in DIR
instead (e.g. fom_data_testing, while it is still being generated).

Usage: python plot_rom_energy.py models/opinf_balanced_r50.pkl [more.pkl ...]
                                 [--data DIR] [--balancing balancing_full]
                                 [--out NAME]
Writes figures/<NAME>.png (default: airfoil_rom_energy).
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_gas_opinf_balanced import forecast

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = {0: "leading edge", 1: "midchord", 2: "trailing edge"}


def load_dataset(args, bal):
    """FOM energies, balanced coordinates and titles for the plotted set.

    Training set (default): from balancing/reduced_*.npz.  With --data DIR
    (e.g. fom_data_testing): every fluct_%03d.npy present in DIR, restricted
    and weighted exactly as the balancing's data.  Returns a dict with t, sq
    (n, nt), Zbal (n, R, nt) balanced coordinates Psi_bal^T q~, X0 (N, n)
    initial states, PhiTPhi_bal and titles.
    """
    R = bal["Phi"].shape[1]
    if args.data is None:
        red = np.load(os.path.join(args.balancing, f"reduced_r{R}.npz"))
        params = red["parameters"]
        data = os.path.join(HERE, "fom_data")
        titles = [f"traj {k}: {STATIONS[k // 3]}, $\\beta = {b:g}$"
                  for k, b in enumerate(params[:, 0])]
        mask, sw = bal["mask"], bal["sqrt_w"]
        X0 = np.stack([np.array(np.load(os.path.join(data, f"fluct_{k:03d}.npy"),
                                        mmap_mode="r")[:, 0])[mask]*sw
                       for k in range(len(params))], axis=1)
        return dict(t=red["time"], sq=red["sq"], Zbal=red["Z"], X0=X0,
                    PhiTPhi_bal=red["PhiTPhi"], titles=titles)

    meta = np.load(os.path.join(args.data, "meta.npz"), allow_pickle=True)
    params = meta["parameters"]
    ks = [k for k in range(len(params))
          if os.path.exists(os.path.join(args.data, f"fluct_{k:03d}.npy"))]
    if not ks:
        raise SystemExit(f"no trajectories in {args.data} yet")
    mask, sw = bal["mask"], bal["sqrt_w"]
    Phi_b, Psi_b = bal["Phi"], bal["Psi"]
    sq, Zbal, X0, titles = [], [], [], []
    for k in ks:
        X = np.load(os.path.join(args.data, f"fluct_{k:03d}.npy"),
                    mmap_mode="r")[mask]*sw[:, None]
        sq.append((X*X).sum(0))
        Zbal.append(Psi_b.T @ X)
        X0.append(np.ascontiguousarray(X[:, 0]))
        st = str(meta["sets"][k]) if "sets" in meta.files else ""
        lab = str(meta["labels"][k]) if "labels" in meta.files else ""
        titles.append(f"test {k} (set {st}): {lab}, $\\beta = {params[k, 0]:.3f}$")
    print(f"{len(ks)} of {len(params)} trajectories available in {args.data}")
    return dict(t=meta["time"], sq=np.array(sq), Zbal=np.array(Zbal),
                X0=np.stack(X0, axis=1), PhiTPhi_bal=Phi_b.T @ Phi_b,
                titles=titles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="checkpoint .pkl files")
    ap.add_argument("--balancing", default=os.path.join(HERE, "balancing_full"))
    ap.add_argument("--data", default=None,
                    help="plot the trajectories in this directory (e.g. "
                         "fom_data_testing) instead of the training set")
    ap.add_argument("--out", default="airfoil_rom_energy")
    args = ap.parse_args()

    bal = np.load(os.path.join(args.balancing, "balancing.npz"), allow_pickle=True)
    ds = load_dataset(args, bal)
    t, sq, n = ds["t"], ds["sq"], len(ds["sq"])
    ncol = 3
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.85*nrow),
                             constrained_layout=True, squeeze=False)
    panels = axes.flat[:n]
    for ax in axes.flat[n:]:
        ax.axis("off")
    for k, ax in enumerate(panels):
        ax.plot(t, sq[k], "k", lw=2.2, label="FOM")
    r_proj = None

    for m, path in enumerate(args.models):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        r = ck["r"]
        A, H = [np.asarray(x) for x in ck["tensors"]]
        Phi, Psi = np.asarray(ck["Phi"]), np.asarray(ck["Psi"])
        if Phi.shape[0] != bal["Phi"].shape[0]:
            raise SystemExit(f"{path}: basis has {Phi.shape[0]} rows, the "
                             f"balancing in {args.balancing} has "
                             f"{bal['Phi'].shape[0]}")
        # decoder Phi (Psi^T Phi)^-1; energy ||D z||^2 = z^T (D^T D) z
        D = np.linalg.inv(Psi.T @ Phi)
        G = D.T @ (Phi.T @ Phi) @ D
        label = os.path.splitext(os.path.basename(path))[0]
        # z(0) = Psi^T q~(0): the balanced coordinates if Psi is the
        # balancing's own, otherwise from the initial states
        if Psi.shape[1] <= bal["Psi"].shape[1] and \
                np.allclose(Psi, bal["Psi"][:, :r]):
            z0 = ds["Zbal"][:, :r, 0]
            r_proj = r
        else:
            z0 = (Psi.T @ ds["X0"]).T
        for k, ax in enumerate(panels):
            Z = forecast(A, H, np.ascontiguousarray(z0[k]), t)
            ax.plot(t, (Z*(G @ Z)).sum(0), color=f"C{m}", lw=1.4, label=label)
            if np.any(~np.isfinite(Z)):
                ax.text(0.97, 0.9 - 0.08*m, f"{label}: blew up", color=f"C{m}",
                        transform=ax.transAxes, ha="right", fontsize=8)

    if r_proj is not None:
        G = ds["PhiTPhi_bal"][:r_proj, :r_proj]
        for k, ax in enumerate(panels):
            Zp = ds["Zbal"][k, :r_proj]
            ax.plot(t, (Zp*(G @ Zp)).sum(0), "k:", lw=1.2,
                    label=rf"$\Phi\Psi^T\tilde q$ (r = {r_proj})")

    for k, ax in enumerate(panels):
        ax.set_title(ds["titles"][k], fontsize=10)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1.3*sq[k].max())
        ax.grid(alpha=0.3)
        if k % ncol == 0:
            ax.set_ylabel(r"$E_{\mathrm{pert}}$")
        if k >= n - ncol:
            ax.set_xlabel("Time $t$")
    panels[0].legend(fontsize=8, loc="upper right")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", f"{args.out}.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
