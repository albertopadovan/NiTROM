#!/bin/bash
#SBATCH -A bgqw-dtai-gh
#SBATCH --job-name=plate3d
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --mem=110g
#SBATCH --output=incompreso.out
#SBATCH --error=incompreso.err

source /u/padovan3/software/incompreso/.venv/bin/activate
cd $WORKNVME/NiTROM/examples/3D_plate

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export PYTHONUNBUFFERED=1
[ -f wing_geometry.npz ] || python generate_wing.py      # wing case (config.yaml)
python -u run.py
