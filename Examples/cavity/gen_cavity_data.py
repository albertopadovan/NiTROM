"""
Headless generation of the cavity training and testing data.

Mirrors examples/cavity/generate_data.py and generate_data_testing.py exactly,
minus the interactive plotting (their ``plt.cm.get_cmap`` calls were removed in
matplotlib 3.9).  Produces, in ./trajectories/:

  traj_%03d.npy / deriv_%03d.npy / weight_%03d.npy   7 training trajectories
  traj_%03d_testing.npy / ...                        25 testing trajectories
  time.npy, amps.npy, amps_testing.npy, phi_pre.npy

Training impulses use beta in {-1, -0.25, -0.05, 0.01, 0.05, 0.25, 1.0} and 160
snapshots on t in [0, 40]; testing uses 25 impulses with beta ~ U(-1, 1), as in
section 5 of Padovan, Vollmer & Bodony (SIADS 2024).
"""
import os
import time as tlib

import numpy as np
import scipy.linalg as sciplin

import classes_cavity as classes
import time_steppers as tstep

Lx = Ly = 1
Nx = Ny = 100
Re = 8300
n = 400
dt = 1.0 / n
BC = [0, 0, 1, 0, 0, 0, 0, 0]
NSAVE = 100
N_PRE = 200

traj_path = "./trajectories/"
os.makedirs(traj_path, exist_ok=True)

flow = classes.flow_class(Lx, Ly, Nx, Ny, Re)
lops = classes.linear_operators_2D(flow, dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy" % (Re, Nx, Ny))
fom = classes.fom_class(flow, lops)
fom.assemble_forcing_profile(0.95, 0.05)
B = fom.f.copy()

time = dt * np.arange(0, n * 40, 1)
tsave = time[::NSAVE]
print(f"grid {Nx}x{Ny}, state dim {flow.szu + flow.szv}, "
      f"{len(tsave)} snapshots on t in [0, {tsave[-1]:.2f}]")


def run(amps, tag):
    """Integrate one impulse response per amplitude; return perturbation fields."""
    Q = np.zeros((flow.szu + flow.szv, len(amps) * len(tsave)))
    for k, a in enumerate(amps):
        t0 = tlib.time()
        qic = flow.q_sbf + a * B
        data, _ = tstep.solver_2D(flow, lops, qic, time, NSAVE, BC)
        data -= flow.q_sbf.reshape(-1, 1)      # perturbation about the base flow
        Q[:, k * len(tsave):(k + 1) * len(tsave)] = data
        print(f"  {tag} {k + 1}/{len(amps)} (beta = {a:+.3f}): "
              f"{tlib.time() - t0:.1f} s, peak energy "
              f"{np.max(np.linalg.norm(data, axis=0) ** 2):.4e}")
    return Q


def save(Q, amps, Phi_pre, suffix=""):
    """Project onto Phi_pre, form exact derivatives, and write to disk."""
    for k in range(len(amps)):
        data = Q[:, k * len(tsave):(k + 1) * len(tsave)]
        ddata = np.zeros_like(data)
        for j in range(data.shape[-1]):
            ddata[:, j] = fom.evaluate_fom_dynamics(
                data[:, j], BC, [0, 0, 0, 0, 0, 0, 0, 0]
            )
        data = Phi_pre.T @ data
        ddata = Phi_pre.T @ ddata
        # alpha_j = time-averaged energy of the trajectory
        weight = np.mean(np.linalg.norm(data, axis=0) ** 2)
        np.save(traj_path + f"traj_%03d{suffix}.npy" % k, data)
        np.save(traj_path + f"deriv_%03d{suffix}.npy" % k, ddata)
        np.save(traj_path + f"weight_%03d{suffix}.npy" % k, [weight])


# %% Training: 7 impulse responses
amps = [-1.0, -0.25, -0.05, 0.01, 0.05, 0.25, 1.0]
print(f"\ntraining: {len(amps)} impulse responses")
Q = run(amps, "train")

# Pre-projection basis: the leading 200 POD modes of the training snapshots.
# The paper pre-projects so that any learned basis inherits the snapshots'
# divergence-free property.
U, S, _ = sciplin.svd(Q, full_matrices=False)
Phi_pre = U[:, :N_PRE]
var = np.cumsum(S ** 2) / np.sum(S ** 2)
print(f"pre-projection: {N_PRE} modes hold {100 * var[N_PRE - 1]:.4f}% of the variance "
      f"(50 modes: {100 * var[49]:.2f}%)")

save(Q, amps, Phi_pre)
np.save(traj_path + "time.npy", tsave)
np.save(traj_path + "amps.npy", amps)
np.save(traj_path + "phi_pre.npy", Phi_pre)

# %% Testing: 25 impulse responses with beta ~ U(-1, 1)
rng = np.random.default_rng(0)
amps_test = rng.uniform(-1, 1, size=25)
print(f"\ntesting: {len(amps_test)} impulse responses")
Qt = run(amps_test, "test")
save(Qt, amps_test, Phi_pre, suffix="_testing")
np.save(traj_path + "amps_testing.npy", amps_test)

print(f"\nwrote -> {traj_path}")
