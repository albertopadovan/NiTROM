#!/bin/bash
#SBATCH -A bgqw-dtai-gh
#SBATCH --job-name=plate3d_traj
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --mem=110g
#SBATCH --output=trajectories.out
#SBATCH --error=trajectories.err

# Training trajectories for the 3D plate (alpha = 15 deg, Re = 500):
#   1. make_initial_conditions.py  -> forcing/ (3 Leray-projected impulse
#      profiles, skipped if forcing/Bf_profiles.npy already exists)
#   2. generate_trajectories.py    -> fom_data/fluct_000..008.npy
#      (3 stations x beta in {0.01, 0.1, 1}; t in [0, 40] saved every 0.2,
#      200 snapshots; ~27 min and ~36 GB each, ~325 GB in total)
# Base flow: base_flow.npz = data/snapshot_000043.npz (t = 35, converged to
# ||dq/dt||/||q|| ~ 3e-9).  Extra arguments are passed to
# generate_trajectories.py, e.g.  sbatch submit_trajectories.sh --only 0 1 2
# Finished trajectories are skipped, so a timed-out job can be resubmitted.

set -e    # stop at the first failing step

source /u/padovan3/software/incompreso/.venv/bin/activate
cd $WORKNVME/NiTROM/examples/3D_plate

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export PYTHONUNBUFFERED=1
# PLATE case (config_plate.yaml): config.yaml now describes the wing.
[ -f forcing/Bf_profiles.npy ] || python -u make_initial_conditions.py --config config_plate.yaml
# Trajectories 0-4 finished on 2026-10-03 (the run died at 5 when the
# /work/nvme/bgqw quota filled up); run the remaining four.  A later
# --only on the command line overrides this one.
python -u generate_trajectories.py --config config_plate.yaml --baseflow base_flow.npz --only 5 6 7 8 "$@"
