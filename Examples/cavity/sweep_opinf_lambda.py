r"""
Sweep the OpInf regularization weight for the cavity and pick the smallest
lambda that yields a model stable on the *training* data.

At small lambda the derivative fit is excellent but the inferred linear
operator picks up a few weakly unstable eigenvalues; integrated forward, the
quadratic term then turns that into a full divergence.  Section 5.1 of the
paper reports the same pathology ("after a first pass, our model exhibited
slightly unstable linear dynamics").

For each lambda this reports both diagnostics:

* ``max Re(lambda_i)`` of the reduced linear operator -- linear stability;
* the integrated training error ``e(t) = <||q - qhat||^2 / alpha_j>_j`` over
  the 7 training impulses -- what actually matters.

Stability is called on the integrated response: a model is "stable" if the
training error stays bounded over t in [0, 40].
"""

import numpy as np

from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import OpInfModule, solve_opinf
from nitrom.projections import LinearProjection
from nitrom.time_steppers.time_stepper import solve_ivp
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

set_backend("numpy")
dtype = np.float64
traj_path = "./trajectories/"
r = 50
E_STABLE = 1.0e2   # training error above this counts as divergence

amps = np.load(traj_path + "amps.npy")
pool = TrainingPool(
    n_traj=len(amps), fname_traj=traj_path + "traj_%03d.npy",
    fname_time=traj_path + "time.npy", dtype=dtype,
    fname_weights=traj_path + "weight_%03d.npy",
    fname_derivs=traj_path + "deriv_%03d.npy",
)
U, S, _ = compute_POD(pool, normalize=True)
Phi = np.ascontiguousarray(U[:, :r])
td = TrainingData(pool, which_trajs=list(range(len(amps))),
                  percent_time_length=1.0, leggauss_deg=5, nsave_rom=1)

t = np.load(traj_path + "time.npy")
Q = np.stack([np.load(traj_path + "traj_%03d.npy" % k) for k in range(len(amps))])
alpha = np.array([float(np.load(traj_path + "weight_%03d.npy" % k)[0])
                  for k in range(len(amps))])
i5 = int(np.argmin(np.abs(t - 5.0)))

print(f"{'lambda':>10} {'||H_r||':>10} {'maxRe':>10} {'J_opinf':>12} "
      f"{'e(t=5)':>12} {'max e(t)':>12}  stable")
rows = []
for lam in np.logspace(-3, 3, 25):
    m = OpInfModule(td, PolynomialModel(r, [1, 2], dtype=dtype),
                    LinearProjection([Phi, Phi]), reg=lam)
    solve_opinf(m)
    m._sync_to_rom()
    A, H = np.asarray(m.rom.A_1), np.asarray(m.rom.A_2)
    mx = float(np.linalg.eigvals(A).real.max())
    Z = solve_ivp(m.rom.evaluate_rhs, Q[:, :, 0] @ Phi, float(t[0]),
                  float(t[-1]), float(t[1] - t[0]) / 40, t, "rk4")
    Qh = np.einsum("nr,brt->bnt", Phi, Z)
    e = (np.sum((Q - Qh) ** 2, axis=1) / alpha[:, None]).mean(axis=0)
    e = np.where(np.isfinite(e), e, np.inf)
    stable = bool(e.max() < E_STABLE)
    rows.append((lam, mx, e.max(), stable))
    print(f"{lam:10.3e} {np.linalg.norm(H):10.4f} {mx:+10.5f} {float(m()):12.4e} "
          f"{e[i5]:12.4e} {e.max():12.4e}  {'yes' if stable else 'NO'}")

ok = [row for row in rows if row[3]]
if ok:
    lam_star = min(row[0] for row in ok)
    print(f"\nsmallest lambda stable on the training data: {lam_star:.6e}")
    lin = [row for row in rows if row[1] < 0]
    if lin:
        print(f"smallest lambda with max Re(lambda_i) < 0:  "
              f"{min(row[0] for row in lin):.6e}")
else:
    print("\nno lambda in the swept range gave a bounded training response")
