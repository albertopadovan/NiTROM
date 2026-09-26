"""
Energy ||x(t)||^2 of the three stability-guaranteed ROMs against the FOM:

  GAS-OpInf (POD)       re-projected data on the POD basis, GAS-constrained
  GAS-OpInf (balanced)  the same on the Lall-balanced basis
  GasNiTROM             GAS-constrained dynamics with Phi and Psi optimized
                        against the NiTROM trajectory cost

All three are globally asymptotically stable by construction, so none of them
can blow up at any horizon.  The comparison is therefore about accuracy alone,
and about what the objective buys: the two GAS-OpInf models minimize the
one-step residual against exact re-projected derivatives, while GasNiTROM
minimizes the trajectory error and is free to move the bases.

No projection curves are drawn -- these are model comparisons, not basis
diagnostics.

Usage: python plot_energy_gas.py [--r 30]
Writes figures/energy_gas_r<r>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from balanced_opinf import ROM
from plot_energy_cavity import load_nitrom, tensor_to_sym
from train_pod_opinf import forecast
from train_models import load_train, HERE, DATA


def load_gas_models(r, trajs_fit=None, with_nitrom=False, pre=0):
    """The three GAS-guaranteed models, as (name, decode, Psi, ROM, maxRe, colour, ls).

    GAS-OpInf keeps Phi and Psi fixed and biorthogonal, so its decode is Phi.
    GasNiTROM's iterates are not biorthogonal, so load_nitrom supplies the
    oblique decode Phi (Psi^T Phi)^-1 instead.
    """
    models = []
    for basis, col, ls in (("pod", "tab:blue", "-."),
                           ("balanced", "tab:olive", ":")):
        tag = f"_pre{pre}" if pre else ""
        path = os.path.join(HERE, f"opinf_gas_{basis}{tag}_r{r}.npy")
        if not os.path.exists(path):
            print(f"note: {os.path.basename(path)} missing -- run "
                  f"train_gas_opinf.py --basis {basis} --derivs reproj "
                  f"--seed reproj")
            continue
        d = np.load(path, allow_pickle=True).item()
        A = np.asarray(d["Ar"])
        models.append((f"GAS-OpInf, {basis} ({d['epochs']} ep)",
                       np.asarray(d["Phi"]), np.asarray(d["Psi"]),
                       ROM(A, np.asarray(d["Hr"]), 2),
                       np.linalg.eigvals(A).real.max(), col, ls))
    gpath = os.path.join(HERE, f"roms_r{r}_gasnitrom"
                         + (f"_pre{pre}" if pre else "") + ".npy")
    if os.path.exists(gpath):
        g = load_nitrom(gpath, r)
        models.append((f"GasNiTROM ({g['n_iters']} it)", g["dec"], g["Psi"],
                       ROM(g["Ar"], g["Hr"], 2),
                       np.linalg.eigvals(g["Ar"]).real.max(),
                       "tab:brown", (0, (3, 1, 1, 1))))
    else:
        print(f"note: {os.path.basename(gpath)} missing")

    # Same GAS-OpInf starting point, but the GAS parameterization dropped: A and
    # H are free, so this one is NOT stability-guaranteed -- it only happens to
    # have converged to a stable A.  Included for the like-for-like comparison.
    fpath = os.path.join(HERE, f"roms_r{r}_nitrom_balanced_fromgas.npy")
    if with_nitrom and os.path.exists(fpath):
        f = load_nitrom(fpath, r)
        models.append((f"NiTROM from GAS ({f['n_iters']} it, unconstrained)",
                       f["dec"], f["Psi"], ROM(f["Ar"], f["Hr"], 2),
                       np.linalg.eigvals(f["Ar"]).real.max(),
                       "tab:pink", (0, (4, 1, 1, 1, 1, 1))))
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--on", choices=["train", "test"], default="train",
                    help="which impulses to evaluate on; the models are always "
                         "fitted on the training set")
    ap.add_argument("--pre", type=int, default=0,
                    help="use the pre-projected models (opinf_gas_*_pre<N>_*)")
    ap.add_argument("--with-nitrom", action="store_true",
                    help="also draw the unconstrained NiTROM started from the "
                         "same GAS-OpInf model (not stability-guaranteed)")
    args = ap.parse_args()
    r = args.r

    trajs_n, betas, time = load_train()
    trajs = [abs(b)*T for b, T in zip(betas, trajs_n)]
    if args.on == "test":
        betas = np.load(os.path.join(DATA, "setup.npz"))["betas_test"]
        trajs = [abs(b)*np.load(os.path.join(DATA, f"test_traj_{j:03d}.npy")
                                ).astype(np.float64)
                 for j, b in enumerate(betas)]
        print(f"evaluating on {len(betas)} HELD-OUT impulses, "
              f"beta = {list(np.round(betas, 3))}")

    models = load_gas_models(r, with_nitrom=args.with_nitrom, pre=args.pre)
    print(f"r = {r}\n")
    n = len(trajs)
    ncol = 4 if n + 1 > 6 else 3
    nrow = int(np.ceil((n + 1)/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 3.6*nrow),
                           sharex=True, squeeze=False, constrained_layout=True)
    ax = ax.reshape(nrow, ncol)
    summary = []
    for name, Ph, Ps, rom, ev, col, ls in models:
        J, err, nbad = forecast(rom, Ph, Ps, trajs, time)
        print(f"{name:<28} maxRe(A) {ev:+.6f}   J {J:.4e}   err {err:.4f}"
              + (f"   {nbad}/{len(trajs)} diverged" if nbad else ""))
        summary.append(f"{name}\n   maxRe(A) = {ev:+.5f}\n"
                       f"   J = {J:.3e}\n   rel. state err = {err:.3f}\n")
        for j, X in enumerate(trajs):
            Z = rom.integrate(Ps.T @ X[:, 0], time, 1.0)
            ok = np.all(np.isfinite(Z), axis=0)
            E = np.where(ok, np.sum((Ph @ np.where(ok, Z, 0.0))**2, axis=0),
                         np.nan)
            ax.flat[j].semilogy(time, E, color=col, ls=ls, lw=1.6,
                                label=name if j == 0 else None)

    for j, (b, X) in enumerate(zip(betas, trajs)):
        a = ax.flat[j]
        a.semilogy(time, np.sum(X**2, axis=0), "k", lw=2.2,
                   label="FOM" if j == 0 else None, zorder=0)
        a.set_title(rf"$\beta = {b:g}$", fontsize=10)
        a.grid(alpha=0.3, which="both")
    handles, labels = ax.flat[0].get_legend_handles_labels()
    order = [labels.index("FOM")] + [i for i, l in enumerate(labels) if l != "FOM"]
    ax.flat[0].legend([handles[i] for i in order], [labels[i] for i in order],
                      fontsize=7, loc="lower left")
    ax.flat[n].axis("off")
    ax.flat[n].text(0.0, 0.5, "\n".join(summary), fontsize=8.5, va="center",
                    family="monospace")
    for a in ax.flat[n + 1:]:
        a.axis("off")
    for a in ax[-1]:
        a.set_xlabel("$t$")
    for a in ax[:, 0]:
        a.set_ylabel(r"$\|x(t)\|^2$")
    what = "HELD-OUT" if args.on == "test" else "training"
    extra = (" (and the unconstrained NiTROM from the same start)"
             if args.with_nitrom else "")
    fig.suptitle(f"Cavity, r = {r}: GAS-constrained ROMs{extra} vs the FOM, "
                 f"{what} impulses", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tag = "_test" if args.on == "test" else ""
    out = os.path.join(HERE, "figures",
                       f"energy_gas_r{r}{'_pre' + str(args.pre) if args.pre else ''}{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
