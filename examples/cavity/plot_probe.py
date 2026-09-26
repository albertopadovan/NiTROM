"""
Point-probe time series: vorticity at a single location, FOM against the models.

The energy ||x(t)||^2 says nothing about *where* the structures are or whether
they arrive at the right time.  A probe does: reading vorticity at one point
shows the phase and amplitude of the vortices passing it, so a model that has
the right energy but a shifted or distorted shear layer is immediately visible.

Default location (x, y) = (0.5, 0.1): the central slice, 0.1 above the bottom
wall, which is where the amplified shear-layer packets travel.

Vorticity is linear in q, so the probe is assembled once as a weight vector w
with four non-zeros and applied as w @ q -- no vorticity fields are formed.
It is checked against train_models.vorticity at start-up.

Usage: python plot_probe.py [--r 30] [--pre 1000] [--on train|test]
                            [--forced OMEGA K --w-amp A] [--x 0.5] [--y 0.1]
Writes figures/probe_r<r>_<case>.png.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cavity import Cavity
from plot_energy_gas import load_gas_models
from train_models import load_train, vorticity, HERE, DATA


def probe_weights(flow, x, y, half=0.0):
    """Weight vector w with w @ q = vorticity averaged over a box around (x, y).

    Vorticity is linear in q, so the average over a set of interior corners is
    just the mean of their individual weight vectors -- no vorticity field is
    ever formed, and the result matches train_models.vorticity exactly.

    Physical coordinates: ``vorticity()`` returns ``flipud(-(dvdx - dudy))``,
    so after the flip row j sits at y = (j+1)*dy with j = 0 at the BOTTOM wall
    and column i at x = (i+1)*dx.  Verified against the base flow (whose
    vorticity peaks at y = 0.99, under the lid) and the actuator (which peaks
    at (0.95, 0.94), matching its documented (0.95, 0.95)).

    :param half: half-width of the averaging box in physical units; 0 gives a
        single corner.  Averaging suppresses the grid-scale content that a
        point probe picks up from a 30-mode ROM.
    :returns: ``(w, (x0, x1, y0, y1), n)`` -- weights, the box actually covered,
        and how many corners it averages
    """
    nrows, ncols = flow.rowsv, flow.colsv - 1      # interior corners
    ii = [k for k in range(ncols)
          if abs((k + 1)*flow.dx - x) <= half + 0.5*flow.dx]
    jj = [k for k in range(nrows)
          if abs((k + 1)*flow.dy - y) <= half + 0.5*flow.dy]
    w = np.zeros(flow.szu + flow.rowsv*flow.colsv)
    for j_phys in jj:
        j = nrows - 1 - j_phys                      # undo the flipud
        for i in ii:
            # -(dv/dx):  -(V[j, i+1] - V[j, i])/dx
            w[flow.szu + j*flow.colsv + i + 1] -= 1.0/flow.dx
            w[flow.szu + j*flow.colsv + i] += 1.0/flow.dx
            # +(du/dy):  +(U[j+1, i] - U[j, i])/dy
            w[(j + 1)*flow.colsu + i] += 1.0/flow.dy
            w[j*flow.colsu + i] -= 1.0/flow.dy
    w /= len(ii)*len(jj)
    box = ((ii[0] + 1)*flow.dx, (ii[-1] + 1)*flow.dx,
           (jj[0] + 1)*flow.dy, (jj[-1] + 1)*flow.dy)
    return w, box, len(ii)*len(jj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--pre", type=int, default=1000)
    ap.add_argument("--on", choices=["train", "test"], default="train")
    ap.add_argument("--forced", nargs=2, type=float, default=None,
                    metavar=("OMEGA", "K"))
    ap.add_argument("--forced-all", action="store_true",
                    help="all six (omega, k) pairs of setup.npz at --w-amp")
    ap.add_argument("--w-amp", type=float, default=0.2)
    ap.add_argument("--x", type=float, default=0.5)
    ap.add_argument("--y", type=float, default=0.1)
    ap.add_argument("--half", type=float, default=0.02,
                    help="half-width of the averaging box in physical units "
                         "(default 0.02 = a 5x5 patch on this grid); 0 probes a "
                         "single corner")
    args = ap.parse_args()
    r = args.r

    cav = Cavity()
    w, box, npts = probe_weights(cav.flow, args.x, args.y, args.half)
    xa, ya = 0.5*(box[0] + box[1]), 0.5*(box[2] + box[3])
    print(f"probe: vorticity averaged over x in [{box[0]:.2f}, {box[1]:.2f}], "
          f"y in [{box[2]:.2f}, {box[3]:.2f}]  ({npts} corners, centre "
          f"({xa:.3f}, {ya:.3f}))")

    _, betas0, time = load_train()
    # check the linear probe against the field routine
    qt = np.load(os.path.join(DATA, "train_traj_000.npy"))[:, 100]
    W = vorticity(cav.flow, qt)
    d = cav.flow.dx
    i0, i1 = int(round(box[0]/d)) - 1, int(round(box[1]/d)) - 1
    j0, j1 = int(round(box[2]/d)) - 1, int(round(box[3]/d)) - 1
    ref = W[j0:j1 + 1, i0:i1 + 1].mean()
    print(f"probe check: w @ q = {w @ qt:+.6e}, field average = {ref:+.6e}, "
          f"diff = {abs(w @ qt - ref):.2e}")

    models = load_gas_models(r, pre=args.pre)

    if args.forced_all:
        setup = np.load(os.path.join(DATA, "setup.npz"))
        legacy = abs(args.w_amp - float(setup["w_amp"])) < 1e-12
        cases = []
        for om, k in setup["forced"]:
            om, k = float(om), int(k)
            nm = (f"forced_w{om:g}_k{k}.npy" if legacy
                  else f"forced_A{args.w_amp:g}_w{om:g}_k{k}.npy")
            path = os.path.join(DATA, nm)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{nm} not found -- run generate_forced.py --amp {args.w_amp:g}")
            cases.append((rf"$\omega = {om:g}$, $k = {k}$",
                          np.load(path).astype(np.float64), None, (om, k)))
        tag = f"forcedall_A{args.w_amp:g}"
    elif args.forced is not None:
        omega, k = float(args.forced[0]), int(args.forced[1])
        setup = np.load(os.path.join(DATA, "setup.npz"))
        legacy = abs(args.w_amp - float(setup["w_amp"])) < 1e-12
        name = (f"forced_w{omega:g}_k{k}.npy" if legacy
                else f"forced_A{args.w_amp:g}_w{omega:g}_k{k}.npy")
        cases = [(rf"$w = {args.w_amp:g}\sin({k}\omega t)$, $\omega = {omega:g}$",
                  np.load(os.path.join(DATA, name)).astype(np.float64),
                  None, (omega, k))]
        tag = f"forced_A{args.w_amp:g}_w{omega:g}_k{k}"
    elif args.on == "test":
        bt = np.load(os.path.join(DATA, "setup.npz"))["betas_test"]
        cases = [(rf"$\beta = {b:g}$",
                  abs(b)*np.load(os.path.join(DATA, f"test_traj_{j:03d}.npy")
                                 ).astype(np.float64), None, None)
                 for j, b in enumerate(bt)]
        tag = "test"
    else:
        trajs_n, betas, _ = load_train()
        cases = [(rf"$\beta = {b:g}$", abs(b)*T, None, None)
                 for b, T in zip(betas, trajs_n)]
        tag = "train"

    n = len(cases)
    ncol = 3 if n == 6 else (4 if n > 4 else min(n, 2))
    nrow = int(np.ceil(n/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.0*ncol, 3.3*nrow),
                           squeeze=False, sharex=True, constrained_layout=True)
    print(f"\n{'case':>22}" + "".join(f"{m[0].split(' (')[0]:>24}" for m in models))
    for c, (title, Y, _, forc) in enumerate(cases):
        a = ax.flat[c]
        a.plot(time, w @ Y, "k", lw=2.0, label="FOM" if c == 0 else None)
        line = f"{title.replace('$',''):>22}"
        for nm, Ph, Ps, rom, ev, col, ls in models:
            if forc is None:
                Z = rom.integrate(Ps.T @ Y[:, 0], time, 1.0)
            else:
                b_r = Ps.T @ (args.w_amp*cav.B)
                om, k = forc
                Z = rom.integrate(np.zeros(r), time, 1.0,
                                  forcing=lambda t, b=b_r: np.sin(k*om*t)*b)
            ok = np.all(np.isfinite(Z), axis=0)
            q = w @ (Ph @ np.where(ok, Z, 0.0))
            a.plot(time, np.where(ok, q, np.nan), color=col, ls=ls, lw=1.4,
                   label=nm if c == 0 else None)
            ref = w @ Y
            e = np.linalg.norm(ref - q)/np.linalg.norm(ref) if ok.all() else np.inf
            line += (f"{e:>24.3f}" if np.isfinite(e) else f"{'diverged':>24}")
        print(line)
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.3)
        a.axhline(0, color="0.7", lw=0.6)
    ax.flat[0].legend(fontsize=7, loc="upper right")
    for a in ax.flat[n:]:
        a.axis("off")
    for a in ax[-1]:
        a.set_xlabel("$t$")
    for a in ax[:, 0]:
        a.set_ylabel(r"vorticity at the probe")
    fig.suptitle(rf"Cavity, r = {r}: vorticity averaged over "
                 rf"$x \in [{box[0]:.2f}, {box[1]:.2f}]$, "
                 rf"$y \in [{box[2]:.2f}, {box[3]:.2f}]$ "
                 rf"-- phase and amplitude of the passing structures", fontsize=11)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", f"probe_r{r}_{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
