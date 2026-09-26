"""
Energy ||x(t)||^2 of the r-dimensional OpInf ROMs against the FOM, for the seven
training impulse responses of eq. (5.5), on both bases:

  balanced  Phi != Psi, biorthogonal, from compute_data_driven_balancing
  POD       Phi = Psi = U, orthogonal -- the paper's own choice

Per basis two curves are drawn:
  ROM            ||Phi z(t)||^2 with z from the learned latent ODE
  projected FOM  ||Phi Psi^T x(t)||^2 -- the best a ROM on that basis could do
                 if its dynamics were exact.  The gap to the FOM is the
                 representation error of the basis; the gap from it to the ROM
                 is the error of the learned dynamics.

Regularizations are given explicitly (see train_pod_opinf.py for the sweep).

Models are always FITTED on the training impulses; --on chooses what they are
EVALUATED on.  "test" uses the held-out impulses of data/test_* (beta drawn at
random from [-1, 1], none of them seen in training), which is the only honest
check: on CGL the training cost kept falling while held-out accuracy degraded.

Usage: python plot_energy_cavity.py [--r 20] [--reg-pod 1e3] [--reg-bal 1e0]
                                    [--on train|test]
Writes figures/energy_cavity_r<r>[_test].png.
"""

import argparse
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from balanced_opinf import opinf, poly_index
from train_pod_opinf import get_pod, forecast
from train_models import load_train, HERE, DATA


def tensor_to_sym(T, r, degree=2):
    """Dense (r,)*(degree+1) tensor -> the (r, n_monomials) symmetric form ROM
    expects.  train_nitrom_bopinf.sym_to_tensor spread each monomial's weight
    evenly over its distinct index permutations, so summing over those
    permutations recovers it."""
    idx = poly_index(r, degree)
    Hs = np.zeros((r, len(idx)))
    for m, tup in enumerate(idx):
        for perm in set(itertools.permutations(tuple(tup))):
            Hs[:, m] += T[(slice(None),) + perm]
    return Hs


def load_nitrom(path, r):
    """Load a NiTROM model into the (decoder, Psi, Ar, Hr) form used here.

    Two conversions are needed and both are silent failures if skipped:
    NiTROM stores H_r as a dense tensor, and its Grassmann/Stiefel iterates are
    *not* biorthogonal -- Psi^T Phi != I -- so the decode is the oblique map
    Phi (Psi^T Phi)^-1 and not Phi."""
    d = np.load(path, allow_pickle=True).item()
    Phi, Psi = np.asarray(d["Phi"]), np.asarray(d["Psi"])
    H = np.asarray(d["Hr"])
    Hs = tensor_to_sym(H, r) if H.ndim > 2 else H
    biorth = np.linalg.norm(Psi.T @ Phi - np.eye(r))
    return dict(dec=Phi @ np.linalg.inv(Psi.T @ Phi), Psi=Psi,
                Ar=np.asarray(d["Ar"]), Hr=Hs, biorth=biorth,
                n_iters=d.get("n_iters"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=20)
    ap.add_argument("--reg-pod", type=float, default=1e3)
    ap.add_argument("--reg-bal", type=float, default=1e0)
    ap.add_argument("--on", choices=["train", "test"], default="train",
                    help="which impulses to evaluate on (fitting is always on "
                         "the training set)")
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs_fit = [abs(b)*T for b, T in zip(betas, trajs_n)]
    dt = time[1] - time[0]
    if args.on == "test":
        betas = np.load(os.path.join(DATA, "setup.npz"))["betas_test"]
        # stored divided by |beta|, float32 -> promote for the error metrics
        trajs = [abs(b)*np.load(os.path.join(DATA, f"test_traj_{j:03d}.npy"),
                                ).astype(np.float64)
                 for j, b in enumerate(betas)]
        print(f"evaluating on {len(betas)} HELD-OUT impulses, "
              f"beta = {list(np.round(betas, 3))}\n")
    else:
        trajs = trajs_fit

    d = np.load(os.path.join(HERE, "balancing_r50.npy"), allow_pickle=True).item()
    U = get_pod(trajs_fit, r)
    bases = {
        "balanced": (d["Phi"][:, :r], d["Psi"][:, :r], args.reg_bal,
                     "tab:red", "--"),
        "POD": (U, U, args.reg_pod, "tab:blue", "-."),
    }
    # The re-projected model (train_reprojection.py) shares the balanced basis,
    # so it needs no projection curve of its own -- only its dynamics differ.
    reproj_path = os.path.join(HERE, f"opinf_reproj_r{r}.npy")
    reproj = (np.load(reproj_path, allow_pickle=True).item()
              if os.path.exists(reproj_path) else None)
    print(f"r = {r}; balanced basis from {d['m_ckpt']} checkpoints, stride "
          f"{d['q_ckpt']}\n")

    den = sum(np.linalg.norm(X)**2 for X in trajs)
    fits = {}
    for name, (Ph, Ps, reg, col, ls) in bases.items():
        rom = opinf(trajs_fit, Ps, dt, "quadratic", reg=reg, C=None)
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        proj = np.sqrt(sum(np.linalg.norm(X - Ph @ (Ps.T @ X))**2
                           for X in trajs)/den)
        ev = np.linalg.eigvals(rom.Ar).real.max()
        print(f"{name:<9} lambda {reg:.0e}: maxRe(Ar) {ev:+.4f}, "
              f"||Hr|| {np.linalg.norm(rom.Hr):.2e}, J {J:.4e}, err {err:.4f}"
              + (f", {nbad} diverged" if nbad else ""))
        # ||Phi Psi^T||_2 without forming the N x N product: with Phi = Q1 R1
        # and Psi = Q2 R2 the nonzero singular values are those of R1 R2^T.
        R1 = np.linalg.qr(Ph, mode="r")
        R2 = np.linalg.qr(Ps, mode="r")
        print(f"{'':<9} projection error {proj:.4f}, "
              f"||Phi Psi^T||_2 = {np.linalg.norm(R1 @ R2.T, 2):.2f}")
        Es, oks = [], []
        for X in trajs:
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Xh = Ph @ np.where(ok, Z, 0.0)
            Es.append(np.where(ok, np.sum(Xh**2, axis=0), np.nan))
            oks.append(ok)
        fits[name] = dict(reg=reg, J=J, err=err, proj=proj, Es=Es, oks=oks,
                          Ep=[np.sum((Ph @ (Ps.T @ X))**2, axis=0) for X in trajs],
                          col=col, ls=ls)

    gas_path = os.path.join(HERE, f"opinf_gas_r{r}.npy")
    gas = (np.load(gas_path, allow_pickle=True).item()
           if os.path.exists(gas_path) else None)

    nitrom_path = os.path.join(HERE, f"roms_r{r}_nitrom_balanced.npy")
    nitrom = load_nitrom(nitrom_path, r) if os.path.exists(nitrom_path) else None

    gasnit_path = os.path.join(HERE, f"roms_r{r}_gasnitrom.npy")
    gasnit = load_nitrom(gasnit_path, r) if os.path.exists(gasnit_path) else None

    fg_path = os.path.join(HERE, f"roms_r{r}_nitrom_balanced_fromgas.npy")
    fromgas = load_nitrom(fg_path, r) if os.path.exists(fg_path) else None

    if reproj is not None:
        from balanced_opinf import ROM
        rom = ROM(reproj["Ar"], reproj["Hr"], 2)
        Ph, Ps = reproj["Phi"], reproj["Psi"]
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        print(f"{'reproj':<9} lambda {reproj['reg']:.0e}: maxRe(Ar) "
              f"{np.linalg.eigvals(reproj['Ar']).real.max():+.4f}, "
              f"J {J:.4e}, err {err:.4f}"
              + (f", {nbad} diverged" if nbad else ""))
        print(f"{'':<9} fit residual {reproj['residual']:.2e} "
              f"(exact Petrov-Galerkin operators)")
        Es, oks = [], []
        for X in trajs:
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Es.append(np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2,
                                          axis=0), np.nan))
            oks.append(ok)
        fits["reproj (balanced)"] = dict(
            reg=reproj["reg"], J=J, err=err, proj=reproj["proj"], Es=Es,
            oks=oks, Ep=None, col="tab:green", ls=(0, (3, 1, 1, 1)))

    if gas is not None:
        from balanced_opinf import ROM
        # Same balanced basis as the unconstrained fit; only the tensors differ,
        # so no projection curve of its own is needed.
        rom = ROM(gas["Ar"], gas["Hr"], 2)
        Ph, Ps = gas["Phi"], gas["Psi"]
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        print(f"{'GAS-OpInf':<9} {gas['epochs']} epochs: maxRe(Ar) "
              f"{gas['max_re_eig']:+.6f} (stable by construction), "
              f"J {J:.4e}, err {err:.4f}"
              + (f", {nbad} diverged" if nbad else ""))
        Es, oks = [], []
        for X in trajs:
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Es.append(np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2, axis=0),
                               np.nan))
            oks.append(ok)
        fits[f"GAS-OpInf ({gas['epochs']} ep)"] = dict(
            reg=gas["reg"], J=J, err=err, proj=None, Es=Es, oks=oks, Ep=None,
            col="tab:olive", ls=(0, (1, 1)))

    if nitrom is not None:
        from balanced_opinf import ROM
        rom = ROM(nitrom["Ar"], nitrom["Hr"], 2)
        Ph, Ps = nitrom["dec"], nitrom["Psi"]
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        prj = np.sqrt(sum(np.linalg.norm(X - Ph @ (Ps.T @ X))**2
                          for X in trajs)/den)
        print(f"{'NiTROM':<9} {nitrom['n_iters']} iters: maxRe(Ar) "
              f"{np.linalg.eigvals(nitrom['Ar']).real.max():+.4f}, "
              f"J {J:.4e}, err {err:.4f}"
              + (f", {nbad} diverged" if nbad else ""))
        print(f"{'':<9} projection error {prj:.4f}, "
              f"||Psi^T Phi - I|| = {nitrom['biorth']:.2e} "
              f"(oblique decode required)")
        Es, oks, Ep = [], [], []
        for X in trajs:
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Es.append(np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2,
                                          axis=0), np.nan))
            oks.append(ok)
            Ep.append(np.sum((Ph @ (Ps.T @ X))**2, axis=0))
        fits[f"NiTROM ({nitrom['n_iters']} it)"] = dict(
            reg=0.0, J=J, err=err, proj=prj, Es=Es, oks=oks, Ep=Ep,
            col="tab:purple", ls=(0, (5, 1)))

    if fromgas is not None:
        from balanced_opinf import ROM
        rom = ROM(fromgas["Ar"], fromgas["Hr"], 2)
        Ph, Ps = fromgas["dec"], fromgas["Psi"]
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        prj = np.sqrt(sum(np.linalg.norm(X - Ph @ (Ps.T @ X))**2
                          for X in trajs)/den)
        print(f"{'NiTROM*':<9} {fromgas['n_iters']} iters (from GAS-OpInf): "
              f"maxRe(Ar) {np.linalg.eigvals(fromgas['Ar']).real.max():+.6f}, "
              f"J {J:.4e}, err {err:.4f}"
              + (f", {nbad} diverged" if nbad else ""))
        print(f"{'':<9} projection error {prj:.4f}, "
              f"||Psi^T Phi - I|| = {fromgas['biorth']:.2e}")
        Es, oks, Ep = [], [], []
        for X in trajs:
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Es.append(np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2, axis=0),
                               np.nan))
            oks.append(ok)
            Ep.append(np.sum((Ph @ (Ps.T @ X))**2, axis=0))
        fits[f"NiTROM from GAS ({fromgas['n_iters']} it)"] = dict(
            reg=0.0, J=J, err=err, proj=prj, Es=Es, oks=oks, Ep=Ep,
            col="tab:pink", ls=(0, (4, 1, 1, 1, 1, 1)))

    if gasnit is not None:
        from balanced_opinf import ROM
        rom = ROM(gasnit["Ar"], gasnit["Hr"], 2)
        Ph, Ps = gasnit["dec"], gasnit["Psi"]
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        prj = np.sqrt(sum(np.linalg.norm(X - Ph @ (Ps.T @ X))**2
                          for X in trajs)/den)
        print(f"{'GasNiTROM':<9} {gasnit['n_iters']} iters: maxRe(Ar) "
              f"{np.linalg.eigvals(gasnit['Ar']).real.max():+.6f} "
              f"(stable by construction), J {J:.4e}, err {err:.4f}"
              + (f", {nbad} diverged" if nbad else ""))
        print(f"{'':<9} projection error {prj:.4f}, "
              f"||Psi^T Phi - I|| = {gasnit['biorth']:.2e}")
        Es, oks, Ep = [], [], []
        for X in trajs:
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            Es.append(np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2, axis=0),
                               np.nan))
            oks.append(ok)
            Ep.append(np.sum((Ph @ (Ps.T @ X))**2, axis=0))
        fits[f"GasNiTROM ({gasnit['n_iters']} it)"] = dict(
            reg=0.0, J=J, err=err, proj=prj, Es=Es, oks=oks, Ep=Ep,
            col="tab:brown", ls=(0, (3, 1, 1, 1)))

    fig, ax = plt.subplots(2, 4, figsize=(18, 7.2), sharex=True,
                           constrained_layout=True)
    for j, (b, X) in enumerate(zip(betas, trajs)):
        a = ax.flat[j]
        a.semilogy(time, np.sum(X**2, axis=0), "k", lw=2.2, label="FOM")
        for name, f in fits.items():
            if f["Ep"] is not None:
                a.semilogy(time, f["Ep"][j], color=f["col"], lw=1.0, ls=":",
                           alpha=0.75, label=f"{name}: projected FOM")
            a.semilogy(time, f["Es"][j], color=f["col"], lw=1.5, ls=f["ls"],
                       label=rf"{name} ROM ($\lambda$ = {f['reg']:.0e})")
            if not f["oks"][j].all():
                a.axvline(time[np.argmax(~f["oks"][j])], color=f["col"],
                          lw=0.8, alpha=0.6)
        a.set_title(rf"$\beta = {b:g}$", fontsize=10)
        a.grid(alpha=0.3, which="both")
        if j == 0:
            a.legend(fontsize=6.5, loc="lower left")
    ax.flat[-1].axis("off")
    txt = "\n".join(
        f"{n}  (lambda = {f['reg']:.0e})\n"
        f"   NiTROM cost J = {f['J']:.3e}\n"
        f"   rel. state err = {f['err']:.3f}\n"
        + (f"   projection err = {f['proj']:.3f}\n" if f["proj"] is not None else "")
        for n, f in fits.items())
    ax.flat[-1].text(0.0, 0.5, txt, fontsize=8.5, va="center", family="monospace")
    for a in ax[-1]:
        a.set_xlabel("$t$")
    for a in ax[:, 0]:
        a.set_ylabel(r"$\|x(t)\|^2$")
    what = ("HELD-OUT impulses" if args.on == "test" else "training impulses")
    fig.suptitle(f"Cavity: energy of the r = {r} ROMs vs the FOM on the {what} "
                 f"(dotted = projection onto each basis)", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tag = "_test" if args.on == "test" else ""
    out = os.path.join(HERE, "figures", f"energy_cavity_r{r}{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
