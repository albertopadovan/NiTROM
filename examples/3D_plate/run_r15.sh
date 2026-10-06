#!/bin/bash
# Reprojection (POD + balanced, r = 15), then closed-form OpInf on both and the energy plot.
set -e
cd /work/nvme/bgqw/padovan3/NiTROM/examples/3D_plate
/u/padovan3/software/incompreso/.venv/bin/python -u compute_reprojected_derivatives.py --basis pod balanced --r 15
for b in pod balanced; do python -u train_opinf.py --basis $b --r 15 --skip-gas; done
python -u plot_rom_energy.py --r 15
echo ALL DONE
