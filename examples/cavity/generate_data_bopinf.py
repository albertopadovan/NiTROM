"""
Training and testing data for balanced OpInf on the lid-driven cavity (SIADS 2024, section 5).

Training (data/train_*): impulse responses q(0) = beta B (eq. (5.5)), beta in BETAS, t in [0, 40],
sampled every 0.1 (the paper used 0.25; finer sampling keeps the finite-difference time
derivatives used by OpInf accurate). Trajectories are stored normalized by the IC norm
|beta| (||B|| = 1), so all of them start from +-B.
Testing (data/test_*, float32): impulse responses with N_TEST values of beta uniform in [-1, 1].
Forced (data/forced_*, float32): zero IC (base flow), w(t) = 0.1 sin(k omega t) for the
(omega, k) pairs of Fig. 9 of the paper; physical units.
"""

import os
from multiprocessing import Pool

import numpy as np

from cavity import Cavity, NSTEP, HERE

DATA = os.path.join(HERE, "data")
BETAS = [-1.0, -0.25, -0.05, 0.01, 0.05, 0.25, 1.0]
N_TEST, SEED = 4, 1
FORCED = [(1.25, 1), (1.25, 2), (1.25, 4), (1.0, 2), (1.0, 3), (1.0, 4)]
W_AMP = 0.1
T, NSAVE = 40.0, NSTEP//10                    # sampling interval 0.1


def impulse(beta):
    cav = Cavity()
    _, Q = cav.integrate(beta*cav.B, T, NSAVE)
    return Q/abs(beta)


def forced(args):
    omega, k = args
    cav = Cavity()
    _, Q = cav.integrate(np.zeros(cav.N), T, NSAVE, forcing=lambda t: W_AMP*np.sin(k*omega*t))
    return Q


def main():
    betas_test = np.round(np.random.default_rng(SEED).uniform(-1, 1, N_TEST), 3)
    with Pool(10) as pool:
        Q_train = pool.map(impulse, BETAS)
        Q_test = pool.map(impulse, betas_test)
        Q_forced = pool.map(forced, FORCED)
    for j, Q in enumerate(Q_train):
        np.save(os.path.join(DATA, f"train_traj_{j:03d}.npy"), Q)
    for j, Q in enumerate(Q_test):
        np.save(os.path.join(DATA, f"test_traj_{j:03d}.npy"), Q.astype(np.float32))
    for (omega, k), Q in zip(FORCED, Q_forced):
        np.save(os.path.join(DATA, f"forced_w{omega:g}_k{k}.npy"), Q.astype(np.float32))
    np.save(os.path.join(DATA, "time.npy"), 0.1*np.arange(Q_train[0].shape[1]))
    np.savez(os.path.join(DATA, "setup.npz"), betas=BETAS, betas_test=betas_test,
             forced=np.array(FORCED), w_amp=W_AMP)
    print(f"Saved {len(BETAS)} training, {N_TEST} testing (beta = {betas_test}) and "
          f"{len(FORCED)} forced trajectories to {DATA}")


if __name__ == "__main__":
    main()
