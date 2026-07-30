import dill
import numpy as np

import fom_class
from nitrom.backend import set_backend
from nitrom.time_steppers.time_stepper import solve_ivp

# Pure-numpy CPU run.
set_backend("numpy")

# %% Instantiate the full-order model class

dtype = np.float64

n = 3
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((n, n, n), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B = np.ones((n, 1), dtype=dtype)
C = np.ones((1, n), dtype=dtype)
time = np.linspace(0.0, 10.0, 100, dtype=dtype)

fom = fom_class.full_order_model(A2, A3, B, C, dtype=dtype)

# %% Generate training trajectories and save to file

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.pkl"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

betas = np.array([0.01, 0.1, 0.2, 0.248], dtype=dtype)
n_traj = len(betas)

# Fixed-step RK4 sub-step used to integrate the FOM between saved snapshots.
dt = float((time[1] - time[0]) / 100)

for k in range(n_traj):
    b = float(betas[k])

    # Full-space forcing field f(t) = B u = b * ones(n) that drives the FOM.
    u = lambda t, b=b: b * np.ones(n, dtype=dtype)

    # Integrate dx/dt = A2 x + A3 : x x^T + f(t) with the RK4 stepper.  The
    # forcing is baked into the RHS closure (rather than passed as an extra
    # positional arg) so it does not collide with the stepper's newton params.
    def rhs(t, x, _u=u):
        return fom.evaluate_fom_dynamics(t, x, _u)

    X = solve_ivp(
        rhs,
        np.zeros(n, dtype=dtype),
        float(time[0]),
        float(time[-1]),
        dt,
        time,
        "rk4",
    )  # (n, n_time)

    # Time derivatives at the saved snapshots.
    dX = np.zeros_like(X)
    for j in range(time.shape[0]):
        dX[:, j] = fom.evaluate_fom_dynamics(float(time[j]), X[:, j], u)

    # Analytic steady state -> trajectory weight = ||C x_ss||^2.
    id_ss = np.array(
        [
            -b / (-1 + beta / 5 * b),
            -b / (-2 + beta / 5 * b),
            b / 5,
        ],
        dtype=dtype,
    )
    weight = np.linalg.norm(fom.compute_output(id_ss)) ** 2

    # Saved forcing is the m = 1 ROM input u(t) = b (multiplied by B_r downstream).
    u_save = lambda t, b=b: b * np.ones(1)

    np.save(fname_traj % k, X)
    np.save(fname_deriv % k, dX)
    np.save(fname_weight % k, np.array([float(weight)]))
    with open(fname_forcing % k, "wb") as f:
        dill.dump(u_save, f)

np.save(fname_time, time)
