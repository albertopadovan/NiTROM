"""
Data for the linearly *unstable* CGL (mu0 = 0.41): limit cycle, training and
testing impulse responses, all in the shifted coordinates x - m.

At mu0 = 0.41 the leading eigenvalue of the linearization about the origin is
+0.0125 +- 0.648i, so small disturbances grow and saturate on a limit cycle.
The CGL is equivariant under the phase rotation q -> q exp(i phi) (complex-
linear operator, nonlinearity -a|q|^2 q), and the cycle is a rotating wave,
q(x, t) = exp(-i omega t) q_hat(x): |q| is constant in time and its average
over a period vanishes.  m_FOM -- the average over the cycle, computed here
over an integer number of periods -- is therefore ~0, and the shift x - m is
kept only so the pipeline is the general one.

Training (data/train_*): the 8 impulse responses of examples/cgl,
x(0) = beta B e_j / ||B e_j||, beta in ALPHAS, j in {0, 1} (branch-I Gaussian
in the real / imaginary part), integrated to T = 1000 -- long enough for the
smallest impulses (beta = 0.01, growth rate 0.0125 after an initial non-normal
decay) to reach the cycle.  Stored as x(t) - m in physical units.

Testing (data/test_*), also x - m:
  0: training Gaussian, unseen amplitude 0.3 (inside the cycle)
  1: actuator impulse B v/||B v||, amplitude 0.3 (inside)
  2: training Gaussian, amplitude 6 (outside the cycle, ||q|| ~ 3.7 on it):
     off-attractor dynamics from the other side, which no training
     trajectory covers

Also data/limit_cycle.npz: m, one period of the cycle, its period/frequency and
amplitude, for the evaluation.

Usage: python generate_data.py
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "cgl"))  # after this dir: local names win
from cgl import CGL, TimeStepper  # noqa: E402

DATA = os.path.join(HERE, "data")
MU0 = 0.41
ALPHAS = [-1.0, 0.01, 0.1, 1.0]
TEST = [("gaussian", 0.3), ("actuator", 0.3), ("gaussian", 6.0)]
T, DT, NSAVE = 1000.0, 1e-2, 20           # sampling interval 0.2
T_LC, NSAVE_LC = 2000.0, 1                # cycle: integrate long, keep every step
SEED = 0


def cgl():
    return CGL(mu0=MU0)


def integrate(args):
    q0, T_, nsave = args
    return TimeStepper(cgl(), DT).integrate(q0, T_, nsave)[1]


def unit(v):
    return v/np.linalg.norm(v)


def limit_cycle(c):
    """Integrate from a small impulse onto the cycle; return m (average over an
    integer number of periods), one period of the cycle and its stats."""
    q0 = 1e-2*unit(c.gaussian(c.xbar))
    _, Q = TimeStepper(c, DT).integrate(q0, T_LC, NSAVE_LC)
    t = DT*NSAVE_LC*np.arange(Q.shape[1])
    late = t >= T_LC - 300.0
    Ql, tl = Q[:, late], t[late]
    nx = c.nx
    # rotation frequency from the phase at the point of largest amplitude
    i = int(np.argmax(np.hypot(Ql[:nx, -1], Ql[nx:, -1])))
    phase = np.unwrap(np.angle(Ql[i] + 1j*Ql[nx + i]))
    omega = np.polyfit(tl, phase, 1)[0]
    period = 2*np.pi/abs(omega)
    n_per = int((tl[-1] - tl[0])//period)
    win = tl >= tl[-1] - n_per*period
    m = Ql[:, win].mean(axis=1)
    one = tl >= tl[-1] - period
    nrm = np.linalg.norm(Ql, axis=0)
    mod = np.hypot(Ql[:nx], Ql[nx:])
    return dict(m=m, cycle=Ql[:, one], t_cycle=tl[one] - tl[one][0], omega=omega,
                period=period, n_periods=n_per, amp_mean=nrm[win].mean(),
                amp_min=nrm[win].min(), amp_max=nrm[win].max(),
                modulus_variation=float((mod.max(1) - mod.min(1)).max()/mod.max()),
                profile=mod[:, -1])


def main():
    os.makedirs(DATA, exist_ok=True)
    c = cgl()
    ev = np.linalg.eigvals(c.A.toarray())
    lead = ev[np.argmax(ev.real)]
    print(f"mu0 = {MU0}: leading eigenvalue of the linearization "
          f"{lead.real:+.5f} {lead.imag:+.4f}i")

    lc = limit_cycle(c)
    m = lc["m"]
    print(f"limit cycle: ||q|| = {lc['amp_mean']:.4f} (min {lc['amp_min']:.4f}, "
          f"max {lc['amp_max']:.4f}), omega = {lc['omega']:.4f}, period "
          f"{lc['period']:.3f}; |q(x,t)| varies by {lc['modulus_variation']:.1e} "
          f"of its max (rotating wave)")
    print(f"m_FOM = average over {lc['n_periods']} periods: ||m|| = "
          f"{np.linalg.norm(m):.2e} ({np.linalg.norm(m)/lc['amp_mean']:.1e} of the "
          f"cycle amplitude)")
    np.savez(os.path.join(DATA, "limit_cycle.npz"), mu0=MU0, **lc)

    g = np.exp(-((c.x - c.xbar)/c.s)**2)
    cols = [unit(np.concatenate((g, np.zeros(c.nx)))),
            unit(np.concatenate((np.zeros(c.nx), g)))]
    v = np.random.default_rng(SEED).standard_normal(2)
    B = np.stack(cols, axis=1)
    bhat = unit(B @ v)

    train = [(a*cols[j], T, NSAVE) for a in ALPHAS for j in (0, 1)]
    test_ic = [a*(cols[0] if kind == "gaussian" else bhat) for kind, a in TEST]
    test = [(q0, T, NSAVE) for q0 in test_ic]
    with Pool(min(11, len(train) + len(test))) as pool:
        Q_train = pool.map(integrate, train)
        Q_test = pool.map(integrate, test)

    for j, Q in enumerate(Q_train):
        np.save(os.path.join(DATA, f"train_traj_{j:03d}.npy"), Q - m[:, None])
    for j, Q in enumerate(Q_test):
        np.save(os.path.join(DATA, f"test_traj_{j:03d}.npy"), Q - m[:, None])
    np.save(os.path.join(DATA, "train_alpha.npy"),
            np.array([a for a in ALPHAS for _ in (0, 1)]))
    np.save(os.path.join(DATA, "train_labels.npy"),
            np.array([f"$\\beta$ = {a:g}, $Be_{j}$ ({'Re' if j == 0 else 'Im'})"
                      for a in ALPHAS for j in (0, 1)]))
    np.save(os.path.join(DATA, "test_alpha.npy"), np.array([a for _, a in TEST]))
    np.save(os.path.join(DATA, "test_labels.npy"),
            np.array([f"Gaussian, $\\beta$ = 0.3 (inside cycle)",
                      f"actuator impulse, amplitude 0.3 (inside cycle)",
                      f"Gaussian, $\\beta$ = 6 (outside cycle)"]))
    np.save(os.path.join(DATA, "time.npy"),
            DT*NSAVE*np.arange(int(round(T/DT))//NSAVE + 1))
    np.savez(os.path.join(DATA, "setup.npz"), mu0=MU0, alphas=ALPHAS, bhat=bhat,
             m=m, T=T, dt_save=DT*NSAVE)
    print(f"saved {len(train)} training and {len(test)} testing trajectories "
          f"(x - m, T = {T:g}, every {DT*NSAVE:g}) to {DATA}")
    for j, Q in enumerate(Q_train):
        print(f"   train {j}: ||q(T)|| = {np.linalg.norm(Q[:, -1]):.4f}")
    for j, Q in enumerate(Q_test):
        print(f"   test  {j}: ||q(T)|| = {np.linalg.norm(Q[:, -1]):.4f}")


if __name__ == "__main__":
    main()
