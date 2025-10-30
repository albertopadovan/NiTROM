import numpy as np
from scipy.linalg import qr
# import fom_class

n = 2000
r = 250
m = 100
time = np.linspace(0, 10, m)
traj_path = "./trajectories/"
fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"
n_traj = 30

weights = np.ones(n_traj)
u = 0.1
for k in range(n_traj):
    u_vec = u * np.ones(n)
    X = np.random.rand(n, m)
    dX = np.random.rand(n, m)

    np.save(fname_traj % k, X)
    np.save(fname_deriv % k, dX)
    np.save(fname_weight % k, [weights[k]])
    np.save(fname_forcing % k, u_vec)
np.save(fname_time, time)

Phi = np.random.rand(n, r)
Phi = qr(Phi, mode='economic')[0]
Psi = Phi.copy()
Q = np.random.rand(r, r)
Q = qr(Q)[0]
D = np.diag(np.random.uniform(-10, -1, r))
A2 = Q @ D @ np.linalg.inv(Q)
# A3 = np.random.rand(r, r, r)
A3 = np.zeros((r, r, r))
np.save(traj_path + "Phi.npy", Phi)
np.save(traj_path + "Psi.npy", Psi)
np.save(traj_path + "A2.npy", A2)
np.save(traj_path + "A3.npy", A3)