"""
Validation of the CGL solver in cgl.py.

1. Agreement with the original solver of Padovan, Vollmer & Bodony (SIADS 2024),
   retrieved from the NiTROM git history (Examples/cgl/fom_class_cgl.py):
   operators, and unforced / forced trajectories. (The original treats the forcing
   explicitly at t_{k-1}; cgl.py uses the trapezoidal rule, so forced runs agree
   to O(dt) only.)
2. Physics: least-stable eigenvalue (stable, omega ~ 0.648) and transient growth.
3. Convergence in time (dt halved) and space (nx doubled).
"""

import importlib.util
import os
import subprocess
import tempfile

import numpy as np
import scipy.sparse.linalg as spla

from cgl import CGL, TimeStepper

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = "8943652~1:Examples/cgl/fom_class_cgl.py"


def load_original():
    src = subprocess.run(["git", "show", ORIGINAL], cwd=HERE, capture_output=True,
                         text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "fom_class_cgl_original.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("fom_class_cgl_original", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rel(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(b)


def gaussian_ic(cgl, xbar, amp):
    q0 = cgl.gaussian(xbar)
    return amp*q0/np.linalg.norm(q0)


def main():
    cgl = CGL()
    nx = cgl.nx
    ok = True

    # ---- 1. Original solver
    orig = load_original()
    ref = orig.CGL(cgl.x, cgl.nu, cgl.gamma, cgl.mu0, cgl.mu2, cgl.a)
    print("1. Agreement with the original solver (git " + ORIGINAL.split(":")[0] + ")")
    for name, mine, theirs in [("D (d/dx)", cgl.D, ref.D), ("DD (d2/dx2)", cgl.DD, ref.DD),
                               ("A (linear operator)", cgl.A, ref.A)]:
        e = spla.norm(mine - theirs)/spla.norm(theirs)
        ok &= e < 1e-14
        print(f"   {name:<22} rel. diff = {e:.1e}")
    C_out = np.exp(-((cgl.x + cgl.xbar)/cgl.s)**2)
    B_act = np.exp(-((cgl.x - cgl.xbar)/cgl.s)**2)
    e = max(rel(C_out, ref.C[0, :nx]), rel(B_act, ref.B[:nx, 0]))
    ok &= e < 1e-14
    print(f"   {'C (branch II), B (branch I)':<22} rel. diff = {e:.1e}")
    q = np.random.default_rng(1).standard_normal(2*nx)
    e = rel(cgl.nonlinearity(q), ref.evaluate_cgl_nonlinearity(q, q, q))
    ok &= e < 1e-14
    print(f"   {'cubic nonlinearity':<22} rel. diff = {e:.1e}")

    dt, T, nsave = 1e-2, 100.0, 20
    time = dt*np.arange(int(round(T/dt)) + 1)
    ref_step = orig.time_step_cgl(ref, time)
    mine = TimeStepper(cgl, dt)
    for amp in [0.1, 1.0, 5.0]:
        q0 = gaussian_ic(cgl, 8.0, amp)
        Qr, _, _ = ref_step.time_step(ref, q0, nsave)
        _, Qm = mine.integrate(q0, T, nsave)
        e = rel(Qm, Qr)
        ok &= e < 1e-12
        print(f"   unforced trajectory, Gaussian xbar = 8, amplitude {amp:<4g}: rel. diff = {e:.1e}")

    omega = 0.648
    v = np.array([0.6, -0.8])
    b = 0.05*(ref.B @ v)/np.linalg.norm(ref.B @ v)
    fU = lambda t: np.sin(omega*t)*b
    Qr, _, _ = ref_step.time_step(ref, np.zeros(2*nx), nsave, fU, 1e12)
    _, Qm = mine.integrate(np.zeros(2*nx), T, nsave, forcing=fU)
    e_f = rel(Qm, Qr)
    _, Qm2 = TimeStepper(cgl, dt/2).integrate(np.zeros(2*nx), T, 2*nsave, forcing=fU)
    Qr2, _, _ = orig.time_step_cgl(ref, (dt/2)*np.arange(int(round(T/(dt/2))) + 1)).time_step(
        ref, np.zeros(2*nx), 2*nsave, fU, 1e12)
    e_f2 = rel(Qm2, Qr2)
    ok &= e_f2 < 0.6*e_f
    print(f"   forced (0.05 sin(omega t) B v): rel. diff = {e_f:.1e} (dt = {dt:g}), "
          f"{e_f2:.1e} (dt = {dt/2:g}) -> O(dt), from the explicit forcing in the original")

    # ---- 2. Physics
    ev = np.linalg.eigvals(cgl.A.toarray())
    lam = ev[np.argmax(ev.real)]
    ok &= (lam.real < 0) and abs(abs(lam.imag) - 0.648) < 1e-3
    print("\n2. Physics")
    print(f"   least-stable eigenvalue {lam.real:.5f} {lam.imag:+.5f}i  (stable, omega = {abs(lam.imag):.4f})")
    _, Q = mine.integrate(gaussian_ic(cgl, 8.0, 0.1), 200.0, 20)
    _, QL = TimeStepper(CGL(a=0.0), dt).integrate(gaussian_ic(cgl, 8.0, 0.1), 200.0, 20)
    g, gL = np.linalg.norm(Q, axis=0).max()/0.1, np.linalg.norm(QL, axis=0).max()/0.1
    print(f"   transient growth of ||q|| (Gaussian at branch I): linear {gL:.2f}, "
          f"nonlinear alpha = 0.1 {g:.2f}")

    # ---- 3. Convergence
    print("\n3. Convergence (output y = C q, alpha = 1 Gaussian at xbar = 8, t in [0, 100])")
    q0 = gaussian_ic(cgl, 8.0, 1.0)
    ys = {}
    for d in [2e-2, 1e-2, 5e-3]:
        _, Q = TimeStepper(cgl, d).integrate(q0, T, int(round(0.2/d)))
        ys[d] = C_out @ Q[:nx] + 1j*(C_out @ Q[nx:])
    e1, e2 = rel(ys[2e-2], ys[5e-3]), rel(ys[1e-2], ys[5e-3])
    print(f"   time: rel. diff vs dt = 5e-3: dt = 2e-2 {e1:.1e}, dt = 1e-2 {e2:.1e} "
          f"(ratio {e1/e2:.1f}; 2nd order ~ 3)")
    ev2 = np.linalg.eigvals(CGL(nx=512).A.toarray())
    lam2 = ev2[np.argmax(ev2.real)]
    print(f"   space: least-stable eigenvalue nx = 512: {lam2.real:.5f} {lam2.imag:+.5f}i "
          f"(|diff| vs nx = 256: {abs(lam2 - lam):.1e})")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
