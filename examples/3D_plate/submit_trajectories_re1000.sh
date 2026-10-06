#!/bin/bash
#SBATCH -A bgqw-dtai-gh
#SBATCH --job-name=wing_re1000_traj
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --mem=110g
#SBATCH --array=0-2
#SBATCH --output=trajectories_re1000_%a.out
#SBATCH --error=trajectories_re1000_%a.err

# Training trajectories for the wing (NACA 0012, 6 deg) at Re = 1000:
#   3 z-symmetric impulse stations (forcing_wing/: leading edge, midchord,
#   trailing edge) x beta in {0.1, 1, 2} (unit W-norm after the Leray
#   projection), t in [0, 40] saved every 0.2 -> 200 snapshots, ~65 GB each
#   (~580 GB in total) in fom_data_wing_re1000_train/fluct_000..008.npy,
#   trajectory k = 3*station + beta index.
# One array task per station (its three betas after a single exact-Schur
# setup, ~5 min + 3 x ~50 min).  Finished trajectories are skipped, so a
# timed-out task can be resubmitted:  sbatch --array=1 submit_trajectories_re1000.sh
#
# Base flow: the Re = 1000 state at t = 86 (data_wing_re1000/, restarted
# from the converged Re = 600 flow at t = 48); snapshot-to-snapshot change
# 5.1e-7 (W-norm, relative, over 2 time units), decaying slowly (~1.4x per 2).

set -e

BASEFLOW=data_wing_re1000/snapshot_000043.npz     # t = 86
OUT=fom_data_wing_re1000_train

source /u/padovan3/software/incompreso/.venv/bin/activate
cd $WORKNVME/NiTROM/examples/3D_plate
[ -f "$BASEFLOW" ] || { echo "missing base flow $BASEFLOW"; exit 1; }

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

S=${SLURM_ARRAY_TASK_ID:-0}
export PYTHONUNBUFFERED=1
python -u generate_trajectories.py --config config_re1000.yaml \
    --baseflow "$BASEFLOW" --out "$OUT" --t-final 40 --dt-save 0.2 \
    --only $((3*S)) $((3*S + 1)) $((3*S + 2)) "$@"
