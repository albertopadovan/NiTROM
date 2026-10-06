#!/bin/bash
# Wing at Re = 1000: leading-edge impulse (z-symmetric, forcing_wing/), beta = 1
# (unit W-norm after the Leray projection), T = 20, snapshots every 0.2, from
# the Re = 1000 base flow data_wing_re1000/snapshot_000041.npz (t = 82)
# -> fom_data_wing_re1000/fluct_002.npy (~32 GB).
#     bash run_wing_le_re1000.sh > run_wing_le_re1000.log 2>&1
set -e
source /u/padovan3/software/incompreso/.venv/bin/activate
cd /work/nvme/bgqw/padovan3/NiTROM/examples/3D_plate
export PYTHONUNBUFFERED=1
python generate_trajectories.py --config config_re1000.yaml \
    --baseflow data_wing_re1000/snapshot_000041.npz --out fom_data_wing_re1000 \
    --t-final 20 --dt-save 0.2 --only 2
