"""
Training and testing data for balanced OpInf on the CGL (nonlinear, a = 0.1).

Training (data/train_*): the paper's impulse responses of eq. (4.5),
x(0) = beta B e_j / ||B e_j|| for beta in ALPHAS and j in {0, 1}, where the
actuator columns B e_j are the branch-I Gaussian g(x) = exp(-((x - xbar)/s)^2)
of eq. (4.3) placed in the real (j = 0) and in the imaginary (j = 1) part of the
state.  That is N_traj = 8 training trajectories, as in section 4 of Padovan,
Vollmer & Bodony, SIADS 2024.  Trajectories are stored normalized by the IC
amplitude, x~(t) = x(t)/beta; index is 2*(beta index) + j.

Testing (data/test_*), impulse responses at an unseen amplitude, also normalized:
  0: the *training* Gaussian (same centre, branch I) at amplitude TEST_AMP, so
     this isolates generalization in amplitude alone
  1: actuator impulse B u = TEST_AMP B v/||B v|| delta(t) (eq. (4.5)), B at branch I (eq. (4.3))
Forced (data/forced_*), zero IC, B u(t) = A sin(k omega t) B v/||B v||, A in FORCE_AMPS,
k in FORCE_K, omega = imaginary part of the least-stable eigenvalue (physical units).

Testing ensemble (data/ens_*), for the average-error curve of eq. (3.9): N_ENS
impulse responses beta B e_j / ||B e_j||, with beta drawn uniformly from
[-1, 1] and j alternating over the two actuator columns (Gaussian at branch I in
the real and in the imaginary part).  Stored in *physical* units, not normalized
by beta, since beta may be arbitrarily close to zero.
"""

import os
from multiprocessing import Pool

import numpy as np

from cgl import CGL, TimeStepper

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# One spatially-localized IC: the actuator profile of eq. (4.3), a Gaussian
# centred at branch I (x = xbar = -8.246).  CGL.gaussian(xbar) is centred at
# x = -xbar, so the stored xbar is -centre.  Amplitudes are the paper's
# beta in {-1, 0.01, 0.1, 1}; index j is simply the amplitude index.
CENTERS = np.array([CGL().xbar])
XBARS = -CENTERS
ALPHAS = [-1.0, 0.01, 0.1, 1.0]
TEST_AMP = 0.3
N_ENS = 50                                # testing ensemble size, eq. (30)
FORCE_AMPS = [0.01, 0.05, 0.2]
FORCE_K = [1, 2]
T, DT, NSAVE = 500.0, 1e-2, 20            # sampling interval 0.2
T_FORCED = 300.0
SEED = 0


def actuator(cgl):
    """Unit vector B v/||B v|| (B: Gaussian at branch I in Re and Im, v random) and omega."""
    nx = cgl.nx
    g = np.exp(-((cgl.x - cgl.xbar)/cgl.s)**2)
    B = np.zeros((2*nx, 2))
    B[:nx, 0], B[nx:, 1] = g, g
    v = np.random.default_rng(SEED).standard_normal(2)
    ev = np.linalg.eigvals(cgl.A.toarray())
    return (B @ v)/np.linalg.norm(B @ v), abs(ev[np.argmax(ev.real)].imag)


def unforced(args):
    q0, amp = args
    cgl = CGL()
    _, Q = TimeStepper(cgl, DT).integrate(amp*q0, T, NSAVE)
    return Q/amp


def forced(args):
    A, k, bhat, omega = args
    cgl = CGL()
    _, Q = TimeStepper(cgl, DT).integrate(np.zeros(2*cgl.nx), T_FORCED, NSAVE,
                                          forcing=lambda t: A*np.sin(k*omega*t)*bhat)
    return Q


def unit(v):
    return v/np.linalg.norm(v)


def main():
    os.makedirs(DATA, exist_ok=True)
    cgl = CGL()
    bhat, omega = actuator(cgl)

    # Actuator columns B e_j of eq. (4.3): the branch-I Gaussian in the real
    # (j = 0) and imaginary (j = 1) part, each unit-norm so that beta means the
    # same thing for training impulses and for the testing ensemble.
    g = np.exp(-((cgl.x - cgl.xbar)/cgl.s)**2)
    cols = [unit(np.concatenate((g, np.zeros(cgl.nx)))),
            unit(np.concatenate((np.zeros(cgl.nx), g)))]

    # Testing ensemble: beta B e_j/||B e_j||, beta ~ U[-1, 1], j alternating.
    betas = np.random.default_rng(SEED + 1).uniform(-1.0, 1.0, N_ENS)
    # amp = 1 so unforced() returns the physical trajectory.
    ens = [(b*cols[j % 2], 1.0) for j, b in enumerate(betas)]

    # The paper's 8 training impulses: every amplitude on both actuator columns.
    train = [(cols[j], a) for a in ALPHAS for j in (0, 1)]
    train_labels = np.array([f"$\\beta$ = {a:g}, $Be_{j}$ ({'Re' if j == 0 else 'Im'})"
                             for a in ALPHAS for j in (0, 1)])
    # Same spatial profile as training; only the amplitude is unseen.
    test = [(unit(cgl.gaussian(XBARS[0])), TEST_AMP), (bhat, TEST_AMP)]
    cases = [(A, k, bhat, omega) for A in FORCE_AMPS for k in FORCE_K]
    with Pool(10) as pool:
        Q_train = pool.map(unforced, train)
        Q_test = pool.map(unforced, test)
        Q_forced = pool.map(forced, cases)
        Q_ens = pool.map(unforced, ens)

    for j, Q in enumerate(Q_train):
        np.save(os.path.join(DATA, f"train_traj_{j:03d}.npy"), Q)
    np.save(os.path.join(DATA, "train_alpha.npy"), np.array([a for _, a in train]))
    np.save(os.path.join(DATA, "train_labels.npy"), train_labels)
    np.save(os.path.join(DATA, "train_col.npy"), np.array([j for _ in ALPHAS for j in (0, 1)]))
    # Every IC sits at branch I; only the component (Re/Im) differs.
    np.save(os.path.join(DATA, "train_xbar.npy"), np.full(len(train), XBARS[0]))
    np.save(os.path.join(DATA, "train_center.npy"), np.full(len(train), -XBARS[0]))
    for j, Q in enumerate(Q_test):
        np.save(os.path.join(DATA, f"test_traj_{j:03d}.npy"), Q)
    np.save(os.path.join(DATA, "test_alpha.npy"), np.array([a for _, a in test]))
    np.save(os.path.join(DATA, "test_labels.npy"),
            np.array([f"Training Gaussian (x = {-XBARS[0]:g}), unseen amplitude {TEST_AMP:g}",
                      f"actuator impulse (branch I), amplitude {TEST_AMP:g}"]))
    for j, Q in enumerate(Q_ens):
        np.save(os.path.join(DATA, f"ens_traj_{j:03d}.npy"), Q)
    np.save(os.path.join(DATA, "ens_beta.npy"), betas)
    for (A, k, _, _), Q in zip(cases, Q_forced):
        np.save(os.path.join(DATA, f"forced_A{A:g}_k{k}.npy"), Q)
    np.save(os.path.join(DATA, "time.npy"), DT*NSAVE*np.arange(int(round(T/DT))//NSAVE + 1))
    np.save(os.path.join(DATA, "time_forced.npy"), DT*NSAVE*np.arange(int(round(T_FORCED/DT))//NSAVE + 1))
    np.savez(os.path.join(DATA, "setup.npz"), bhat=bhat, omega=omega, force_amps=FORCE_AMPS,
             force_k=FORCE_K, alphas=ALPHAS, xbars=XBARS, centers=CENTERS)
    print(f"Saved {len(train)} training, {len(test)} testing, {len(cases)} forced and "
          f"{len(ens)} ensemble trajectories to {DATA} (omega = {omega:.4f})")


if __name__ == "__main__":
    main()
