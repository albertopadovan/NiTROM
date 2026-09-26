"""
Output y = C q (Gaussian at branch II, eq. (4.2)) versus time for the r = 5 ROMs
of train_models.py:
  figures/impulse_responses_r5.png  : representative training and testing impulse responses
  figures/sinusoidal_responses_r5.png: sinusoidal inputs A sin(k omega t) B v/||B v||
Relative output errors ||y - y_rom|| / ||y|| over the plotted window are printed and
shown in the legends.
"""

import glob
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgl import CGL
from balanced_opinf import ROM, poly_index

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = 5
T_PLOT = 200.0
# 8 training impulses (2*(beta index) + column): show the smallest amplitude on
# the real column and the largest on the imaginary one.
TRAIN_SHOW = [2, 7]
STYLES = {"Balanced (Lall-averaged)": ("tab:green", "-."),
          "POD": ("tab:blue", ":")}
# One NiTROM curve per initialization, picked up from roms_r{R}_nitrom_*.npy.
NITROM_STYLES = {"NiTROM (Lall init)": ("tab:purple", (0, (3, 1, 1, 1))),
                 "NiTROM (POD init)": ("tab:orange", (0, (5, 1)))}
NITROM_GLOB = "roms_r{r}_nitrom_*.npy"


def tensor_to_sym(T, r, degree=3):
    """
    Inverse of train_nitrom.sym_to_tensor: dense (r,)*(degree+1) tensor back to
    the (r, n_monomials) symmetric form that ROM expects.

    sym_to_tensor spread each monomial's weight evenly over its distinct index
    permutations, so summing over those permutations recovers it.
    """
    idx = poly_index(r, degree)
    Hs = np.zeros((r, len(idx)))
    for m, triple in enumerate(idx):
        for perm in set(itertools.permutations(tuple(triple))):
            Hs[:, m] += T[(slice(None),) + perm]
    return Hs


def decoder(d):
    """Oblique decode map Phi (Psi^T Phi)^-1.

    Equals Phi for biorthogonal bases (the balanced ones, Psi^T Phi = I), but
    NiTROM's Grassmann/Stiefel iterates are *not* biorthogonal -- here
    ||Psi^T Phi - I|| ~ 2.7 -- so the inverse cannot be skipped."""
    Phi, Psi = d["Phi"], d["Psi"]
    return Phi @ np.linalg.inv(Psi.T @ Phi)


def load_roms(path_main, nitrom_glob):
    """Load the OpInf ROMs, merging in every NiTROM refinement found.

    One file per initialization (roms_r5_nitrom_lall.npy, ..._pod.npy, ...), so
    runs from different starting points appear as separate curves.
    train_nitrom.py stores its cubic tensor densely, so convert it back to the
    monomial layout the rest of this script assumes."""
    roms = np.load(path_main, allow_pickle=True).item()
    styles = dict(STYLES)
    found = sorted(glob.glob(nitrom_glob))
    for path in found:
        for name, d in np.load(path, allow_pickle=True).item().items():
            d = dict(d)
            H = np.asarray(d["Hr"])
            if H.ndim > 2:                       # dense -> symmetric monomials
                d["Hr"] = tensor_to_sym(H, d["Phi"].shape[1])
            roms[name] = d
            styles[name] = NITROM_STYLES.get(name, ("tab:purple", (0, (3, 1, 1, 1))))
            print(f"including '{name}' ({d.get('n_iters', '?')} iterations) "
                  f"from {os.path.basename(path)}")
    if not found:
        print(f"note: no {nitrom_glob} found -- run train_nitrom.py to include it")
    return roms, styles


def main():
    cgl = CGL()
    nx = cgl.nx
    C = np.exp(-((cgl.x + cgl.xbar)/cgl.s)**2)
    out = lambda X: C @ X[:nx] + 1j*(C @ X[nx:])
    roms, styles = load_roms(os.path.join(HERE, f"roms_r{R}.npy"),
                             os.path.join(HERE, NITROM_GLOB.format(r=R)))
    setup = np.load(os.path.join(DATA, "setup.npz"))
    bhat, omega = setup["bhat"], float(setup["omega"])
    time = np.load(os.path.join(DATA, "time.npy"))
    tf = np.load(os.path.join(DATA, "time_forced.npy"))
    alphas = np.load(os.path.join(DATA, "train_alpha.npy"))
    train_labels = np.load(os.path.join(DATA, "train_labels.npy"))
    test_alpha, test_labels = np.load(os.path.join(DATA, "test_alpha.npy")), np.load(os.path.join(DATA, "test_labels.npy"))

    # ---- impulse responses (amplitude-normalized: y/alpha)
    cases = [(f"Training: {train_labels[j]}",
              np.load(os.path.join(DATA, f"train_traj_{j:03d}.npy")), alphas[j]) for j in TRAIN_SHOW]
    cases += [(f"Testing: {test_labels[i]}", np.load(os.path.join(DATA, f"test_traj_{i:03d}.npy")),
               test_alpha[i]) for i in range(len(test_alpha))]
    nt = int(round(T_PLOT/(time[1] - time[0]))) + 1
    t = time[:nt]
    print(f"Relative output error, r = {R}, t in [0, {T_PLOT:g}]")
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5))
    for a_, (title, X, c) in zip(ax.flat, cases):
        y = out(X[:, :nt])
        a_.plot(t, y.real, "k", lw=2, label="FOM")
        line = f"  {title:<62}"
        for name, (col, ls) in styles.items():
            d = roms[name]
            yh = out(decoder(d) @ ROM(d["Ar"], d["Hr"], 3).integrate(
                d["Psi"].T @ X[:, 0], t, c))
            e = np.linalg.norm(y - yh)/np.linalg.norm(y)
            a_.plot(t, yh.real, color=col, ls=ls, lw=1.5, label=f"{name} (err {e:.1e})")
            line += f" {e:9.2e}"
        print(line)
        a_.set_title(title, fontsize=10)
        a_.set_xlim(0, T_PLOT)
        a_.set_ylim(-1.5*np.abs(y.real).max(), 1.5*np.abs(y.real).max())
        a_.legend(fontsize=8, loc="lower right")
    for a_ in ax[:, 0]:
        a_.set_ylabel(r"Re $y/\alpha$")
    for a_ in ax[1]:
        a_.set_xlabel("$t$")
    fig.suptitle(f"Impulse responses, r = {R} cubic OpInf ROMs", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"impulse_responses_r{R}.png"), dpi=150)

    # ---- sinusoidal inputs (physical units, zero IC)
    amps, ks = setup["force_amps"], setup["force_k"]
    fig, ax = plt.subplots(len(amps), len(ks), figsize=(15, 3.8*len(amps)), sharex=True)
    for i, A in enumerate(amps):
        for j, k in enumerate(ks):
            a_ = ax[i, j]
            y = out(np.load(os.path.join(DATA, f"forced_A{A:g}_k{k}.npy")))
            a_.plot(tf, y.real, "k", lw=2, label="FOM")
            line = f"  sinusoid {A:g} sin({k} omega t){'':<37}"
            for name, (col, ls) in styles.items():
                d = roms[name]
                b_r = d["Psi"].T @ (A*bhat)
                Z = ROM(d["Ar"], d["Hr"], 3).integrate(np.zeros(R), tf, 1.0,
                                                    forcing=lambda s: np.sin(k*omega*s)*b_r)
                yh = out(decoder(d) @ Z)
                e = np.linalg.norm(y - yh)/np.linalg.norm(y)
                a_.plot(tf, yh.real, color=col, ls=ls, lw=1.4, label=f"{name} (err {e:.1e})")
                line += f" {e:9.2e}"
            print(line)
            a_.set_title(f"$B u = {A:g}\\,\\sin({k if k > 1 else ''}\\omega t)\\,Bv/\\|Bv\\|$", fontsize=10)
            a_.set_ylim(-1.5*np.abs(y.real).max(), 1.5*np.abs(y.real).max())
            a_.legend(fontsize=8, loc="lower right")
        ax[i, 0].set_ylabel("Re $y$")
    for a_ in ax[-1]:
        a_.set_xlabel("$t$")
    fig.suptitle(f"Sinusoidal inputs ($\\omega$ = {omega:.3f}), r = {R} cubic OpInf ROMs", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"sinusoidal_responses_r{R}.png"), dpi=150)
    print(f"  (columns: {', '.join(styles)})")


if __name__ == "__main__":
    main()
