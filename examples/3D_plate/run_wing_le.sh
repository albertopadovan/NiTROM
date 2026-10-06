#!/bin/bash
# Wing, leading-edge impulse, beta = 1 (unit W-norm after the Leray
# projection), T = 20, snapshots every 0.2 -> fom_data_wing/fluct_002.npy
# (100 x 81M float32, ~32 GB).  Run interactively on a GPU node:
#     bash run_wing_le.sh 2>&1 | tee run_wing_le.log
set -e
source /u/padovan3/software/incompreso/.venv/bin/activate
cd /work/nvme/bgqw/padovan3/NiTROM/examples/3D_plate
export PYTHONUNBUFFERED=1

# Trajectory k = 3*station + beta index: station 0 = leading edge, beta index 2 = 1.0
# (builds forcing_wing/ first if missing, on the same setup: one exact-Schur build)
python generate_trajectories.py --config config_traj.yaml \
    --baseflow data_wing/snapshot_000024.npz --t-final 20 --dt-save 0.2 --only 2
