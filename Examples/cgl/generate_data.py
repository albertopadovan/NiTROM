r"""
Generate CGL training and testing data (section 4 of Padovan, Vollmer & Bodony,
SIADS 2024).

**Training.**  Eight impulse responses.  The impulse

.. math::

    \mathbf{B}\mathbf{u}(t) = \begin{cases}
        \beta \mathbf{B}\mathbf{e}_j & t = 0, \\
        0 & t \neq 0,
    \end{cases}

with :math:`\mathbf{e}_j \in \mathbb{R}^2` the standard basis vectors and
:math:`\beta \in \{-1.0, 0.01, 0.1, 1.0\}`, enters as the initial condition
:math:`\mathbf{q}(0) = \beta \mathbf{B}\mathbf{e}_j`; the forcing vanishes for
all :math:`t > 0`.  The output is collected at ``N = 1000`` uniformly spaced
instants :math:`t_i \in [0, 1000]`.  The trajectory weight is
:math:`\alpha_j = \langle \|\mathbf{y}^{(j)}\|^2 \rangle_t`, the time-averaged
output energy along the trajectory.

**Testing.**  Fifty impulse responses with :math:`\beta` drawn uniformly at
random from :math:`[-1, 1]`, plus two sinusoidally forced trajectories

.. math::

    \mathbf{B}\mathbf{u}(t) = 0.05 \sin(k\omega t)\,
        \mathbf{B}\mathbf{v} / \|\mathbf{B}\mathbf{v}\|,
    \qquad k = 1, 2,

with :math:`\mathbf{v} \in \mathbb{R}^2` random and :math:`\omega \approx
0.648` the natural frequency.  Sinusoids are never seen during training.

Runs in roughly five minutes on a laptop; the FOM is integrated with RK4 at
``dt = 0.01`` (the fourth-order stencil at ``dx = 0.2`` is stable up to
``dt ~ 0.0147``).
"""

import os
import time as timer

import fom_class
import numpy as np

from nitrom.backend import set_backend
from nitrom.time_steppers.time_stepper import solve_ivp

# Pure-numpy CPU run.
set_backend("numpy")

dtype = np.float64
rng = np.random.default_rng(0)

traj_path = "./trajectories/"
test_path = "./testing/"
os.makedirs(traj_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# %% Full-order model

L, n_nodes = 30.0, 301
x, dx, A, B, C = fom_class.build_operators(L=L, n=n_nodes, dtype=dtype)
fom = fom_class.full_order_model(A, B, C, dtype=dtype)
N = A.shape[0]  # = 2 * n_nodes

print(f"CGL FOM: n = {n_nodes} nodes on [-{L}, {L}], state dimension N = {N}")
print(f"  branch I at xbar = {fom_class.X_BAR:.4f}, natural frequency "
      f"omega = {fom_class.OMEGA:.4f}")

DT = 0.01  # RK4 step


def integrate(q0, time, forcing_field=None):
    """RK4-integrate the FOM from ``q0``, saving at ``time``."""
    n_sub = max(1, int(round(float(time[1] - time[0]) / DT)))
    dt = float(time[1] - time[0]) / n_sub
    field = np.zeros_like(q0) if forcing_field is None else forcing_field

    def rhs(t, q, _f=field):
        f = _f(t) if callable(_f) else _f
        return fom.evaluate_fom_dynamics(t, q, f)

    return solve_ivp(rhs, q0, float(time[0]), float(time[-1]), dt, time, "rk4")


def save_set(path, X, time, weights, tag):
    """Write a batch of trajectories in the layout TrainingPool expects."""
    for k in range(X.shape[0]):
        np.save(path + f"{tag}_%03d.npy" % k, X[k])
    np.save(path + "time.npy", time)
    np.save(path + f"{tag}_weights.npy", weights)


# %% Training data: 8 impulse responses

betas = np.array([-1.0, 0.01, 0.1, 1.0], dtype=dtype)
# One trajectory per (beta, e_j) pair -> N_traj = 8.
q0_train = np.stack(
    [b * B[:, j] for j in range(2) for b in betas]
).astype(dtype)
n_traj = q0_train.shape[0]

time_train = np.linspace(0.0, 1000.0, 1000, dtype=dtype)

print(f"\nintegrating {n_traj} training impulse responses "
      f"over t in [0, {time_train[-1]:.0f}] ...")
t0 = timer.perf_counter()
X_train = integrate(q0_train, time_train)  # (n_traj, N, nt)
print(f"  done in {timer.perf_counter() - t0:.1f} s")

# Exact time derivatives at the saved snapshots (the impulse forcing is zero
# for t > 0, so the RHS is unforced).
dX_train = fom.evaluate_fom_dynamics(
    0.0, np.transpose(X_train, (0, 2, 1)), np.zeros(N, dtype=dtype)
)
dX_train = np.transpose(dX_train, (0, 2, 1))  # (n_traj, N, nt)

Y_train = np.einsum("on,bnt->bot", C, X_train)  # (n_traj, 2, nt)
# alpha_j = time-averaged output energy along trajectory j.
alpha_train = np.mean(np.sum(Y_train**2, axis=1), axis=1)  # (n_traj,)

# The weight written to disk is alpha_j itself.  The cost normalization of
# eq. (3.7) is N_traj * N * alpha_j, but TrainingData supplies the N_traj * N
# factor internally -- adding it here as well would over-weight the
# regularizer by that factor.
for k in range(n_traj):
    np.save(traj_path + "traj_%03d.npy" % k, X_train[k])
    np.save(traj_path + "deriv_%03d.npy" % k, dX_train[k])
    np.save(traj_path + "weight_%03d.npy" % k, np.array([alpha_train[k]]))
np.save(traj_path + "alpha.npy", alpha_train)
np.save(traj_path + "time.npy", time_train)
np.save(traj_path + "outputs.npy", Y_train)
np.save(traj_path + "q0.npy", q0_train)

print("  trajectory weights (time-averaged output energy):")
for k in range(n_traj):
    b = betas[k % 4]
    comp = "Re" if k < 4 else "Im"
    print(f"    traj {k}: beta = {b:>6}, e_{comp}  peak|y| = "
          f"{np.abs(Y_train[k]).max():.4e}  alpha = {alpha_train[k]:.6e}")

# %% Testing data 1: 50 random impulse responses

n_test = 50
betas_test = rng.uniform(-1.0, 1.0, n_test).astype(dtype)
# Alternate the input direction e_j across the testing set.
dirs_test = rng.integers(0, 2, n_test)
q0_test = np.stack(
    [betas_test[k] * B[:, dirs_test[k]] for k in range(n_test)]
).astype(dtype)

time_test = np.linspace(0.0, 500.0, 500, dtype=dtype)
print(f"\nintegrating {n_test} testing impulse responses "
      f"over t in [0, {time_test[-1]:.0f}] ...")
t0 = timer.perf_counter()
X_test = integrate(q0_test, time_test)
print(f"  done in {timer.perf_counter() - t0:.1f} s")

Y_test = np.einsum("on,bnt->bot", C, X_test)
alpha_test = np.mean(np.sum(Y_test**2, axis=1), axis=1)

np.save(test_path + "q0.npy", q0_test)
np.save(test_path + "outputs.npy", Y_test)
np.save(test_path + "weights.npy", alpha_test)
np.save(test_path + "time.npy", time_test)

# %% Testing data 2: sinusoidal forcing at omega and 2 omega

omega = fom_class.OMEGA
v = rng.standard_normal(2).astype(dtype)
Bv_norm = float(np.linalg.norm(B @ v))
amp = 0.05
ks = (1, 2)

time_sin = np.linspace(0.0, 200.0, 400, dtype=dtype)
print(f"\nintegrating {len(ks)} sinusoidally forced testing trajectories "
      f"(omega = {omega:.4f}) ...")

q0_sin = np.zeros((len(ks), N), dtype=dtype)
Bv_unit = (B @ v) / Bv_norm


def sin_field(t):
    """Full-space forcing for both sinusoidal trajectories, shape (2, N)."""
    return np.stack([amp * np.sin(k * omega * t) * Bv_unit for k in ks])


t0 = timer.perf_counter()
X_sin = integrate(q0_sin, time_sin, forcing_field=sin_field)
print(f"  done in {timer.perf_counter() - t0:.1f} s")

Y_sin = np.einsum("on,bnt->bot", C, X_sin)
alpha_sin = np.mean(np.sum(Y_sin**2, axis=1), axis=1)

np.save(test_path + "sin_outputs.npy", Y_sin)
np.save(test_path + "sin_weights.npy", alpha_sin)
np.save(test_path + "sin_time.npy", time_sin)
np.save(test_path + "sin_v.npy", v)
# The ROM-level input is u(t) = amp sin(k omega t) v / ||Bv||, so that the FOM
# sees B u(t) = amp sin(k omega t) Bv / ||Bv||.  Only the scalars are stored;
# read_results.py rebuilds the callables (pickled closures do not survive the
# move between scripts).
np.save(test_path + "sin_meta.npy", np.array([amp, Bv_norm, omega, *ks]))

for i, k in enumerate(ks):
    print(f"    k = {k} (omega_f = {k * omega:.4f}): peak|y| = "
          f"{np.abs(Y_sin[i]).max():.4e}")

print(f"\nwrote training data -> {traj_path}")
print(f"wrote testing data  -> {test_path}")
