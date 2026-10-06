#!/bin/bash
# Wing, leading-edge impulse centred OFF mid-span at z = 0.1 (top hat flat on
# 0.04 <= z <= 0.16, zero outside 0 <= z <= 0.2), beta = 1 (unit W-norm after
# the Leray projection), T = 20, snapshots every 0.2
# -> forcing_wing_z0.1/, fom_data_wing_z0.1/fluct_002.npy (~32 GB).
#     bash run_wing_le_z01.sh > run_wing_le_z01.log 2>&1
set -e
source /u/padovan3/software/incompreso/.venv/bin/activate
cd /work/nvme/bgqw/padovan3/NiTROM/examples/3D_plate
export PYTHONUNBUFFERED=1
python generate_trajectories.py --config config_traj.yaml --z-center 0.1 \
    --baseflow data_wing/snapshot_000024.npz --t-final 20 --dt-save 0.2 --only 2
