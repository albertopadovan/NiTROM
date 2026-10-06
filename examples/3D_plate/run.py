"""March the flow past the fixed 3D plate (alpha = 15 deg, Re = 500).

Everything is set by config.yaml; t1 and mjump can be overridden here without
editing it.  To continue a finished run, switch the config's initial
condition to `type: file` pointing at the newest snapshot (it carries the
time and the snapshot numbering over) and raise t1.

The output directory in the config is taken relative to the config file, so
the script can be launched from anywhere.

Usage (incompreso venv, on a GPU node):
    python -u run.py [--config config.yaml] [--t1 T] [--mjump M]
"""

from __future__ import annotations

import argparse
import os

from incompreso import parse_input_file

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--t1", type=float, default=None, help="override config t1")
    ap.add_argument("--mjump", type=int, default=None,
                    help="override config mjump (steps between snapshots)")
    args = ap.parse_args()

    from case import Case
    body = Case(args.config).body_file
    if not os.path.exists(body):
        raise FileNotFoundError(f"{os.path.basename(body)} missing: run generate_wing.py "
                                f"(wing) or generate_plate.py (plate) first")

    sim = parse_input_file(args.config)
    t0 = sim["t0"]
    t1 = sim["t1"] if args.t1 is None else args.t1
    mjump = sim["mjump"] if args.mjump is None else args.mjump
    dt = sim["time_stepper"].dt
    save_path = sim["save_path"]
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), save_path)
    os.makedirs(save_path, exist_ok=True)

    if t1 <= t0:
        raise ValueError(f"t1 = {t1:g} is not beyond the start time t0 = {t0:g}")

    q0 = sim["q0"]
    q0_norm = float((q0*q0).sum())**0.5          # reduce on device (CuPy-safe)
    print(f"start: t = {t0:g}, ||q|| = {q0_norm:.6f}, snapshot numbering from "
          f"{sim['start_step']}", flush=True)
    print(f"marching t = {t0:g} -> {t1:g} ({int(round((t1 - t0)/dt))} steps of "
          f"dt = {dt:g}), saving every {mjump} steps (= {mjump*dt:g} time units) "
          f"to {save_path}", flush=True)

    sim["time_stepper"].solve(t0, t1, mjump, q0, save_path=save_path,
                              start_step=sim["start_step"])
    print("\nmarch finished", flush=True)


if __name__ == "__main__":
    main()
