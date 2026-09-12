"""
Fluctuation energy of the cavity training trajectories.

Reproduces Figure 7(b) of Padovan, Vollmer & Bodony (SIADS 2024): the energy
(squared two-norm) of the seven training impulse responses.  The paper's text
describes the signature of non-normality here -- "after an initial decay the
energy spikes around t = 5 before decaying back to zero" -- so this doubles as
a check that the generated data reproduces the transient-growth mechanism the
whole example is about.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

traj_path = "./trajectories/"
amps = np.load(traj_path + "amps.npy")
t = np.load(traj_path + "time.npy")

energy = np.stack([
    np.linalg.norm(np.load(traj_path + "traj_%03d.npy" % k), axis=0) ** 2
    for k in range(len(amps))
])  # (n_traj, nt)

print(f"{'beta':>8} {'E(0)':>12} {'peak E':>12} {'t_peak':>8} {'E(40)/E(0)':>12}")
for k, a in enumerate(amps):
    e = energy[k]
    print(f"{a:>8.2f} {e[0]:>12.4e} {e.max():>12.4e} "
          f"{t[np.argmax(e)]:>8.2f} {e[-1] / e[0]:>12.2e}")

# Transient growth relative to the initial energy, after the initial decay.
print("\ntransient growth E_max / E_min after the initial dip:")
for k, a in enumerate(amps):
    e = energy[k]
    dip = int(np.argmin(e[: len(e) // 4]))
    late = dip + int(np.argmax(e[dip:]))
    print(f"  beta = {a:+.2f}: dip at t = {t[dip]:5.2f}, "
          f"spike at t = {t[late]:5.2f}, growth {e[late] / e[dip]:.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
for k, a in enumerate(amps):
    axes[0].plot(t, energy[k], "k", lw=1.2)
    axes[1].semilogy(t, energy[k], lw=1.2, label=rf"$\beta = {a:+.2f}$")
axes[0].set_xlabel(r"Time $t$")
axes[0].set_ylabel(r"Energy of perturbations")
axes[0].set_xlim(0, 40)
axes[0].set_ylim(0, None)
axes[0].set_title("linear scale (as in Figure 7b)")
axes[1].set_xlabel(r"Time $t$")
axes[1].set_ylabel(r"Energy of perturbations")
axes[1].set_xlim(0, 40)
axes[1].set_title("log scale, per trajectory")
axes[1].legend(ncol=2, fontsize=8)
out = "figures/cavity_training_energy.png"
fig.savefig(out, dpi=150)
print(f"\nsaved -> {out}")
