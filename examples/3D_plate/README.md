# 3D plate at fixed angle of attack

Flow past a thin rectangular plate (chord 1, span 1) fixed at alpha = 15 deg,
Re = 500, with uniform (u, v, w) = (1, 0, 0) on the inflow and side/top
boundaries and zero Neumann at the outflow. Mesh and plate markers follow
`GreensFunctionPC_Paper_Examples/3D_pitching_plate`; (nz, ny, nx) =
(250, 250, 356).

1. `python generate_plate.py` writes `plate_geometry.npz` (6561 markers, LE at
   the origin, rotated nose up about the LE).
2. `sbatch submit.sh`, or `python -u run.py [--t1 T] [--mjump M]`, marches from
   the uniform flow; snapshots go to `data/`.
