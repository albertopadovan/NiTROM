"""Which test case a config describes, and where that case's files live.

This folder holds two cases that share the mesh/BC layout and the scripts:

    wing   config.yaml        NACA 0012 wing, 6 deg, Re 600 (the 2D airfoil case in 3D)
    plate  config_plate.yaml  flat plate, 15 deg (the earlier test case)

Everything case-specific is derived from the config a script is given:
the body markers (its immersed_body.geometry.path), the post-processing
config (mesh + BCs only), the base-flow snapshot directory (output.save_path)
and the impulse/trajectory directories.  The plate keeps the original names
(forcing/, fom_data/, data/) so its existing data stays valid.
"""

import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")


class Case:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = os.path.abspath(config)
        with open(self.config) as f:
            cfg = yaml.safe_load(f)
        base = os.path.dirname(self.config)
        geom = cfg["immersed_body"]["geometry"]
        self.body_file = os.path.join(base, geom["path"])
        self.name = "wing" if "wing" in os.path.basename(geom["path"]) else "plate"
        self.post_config = os.path.join(
            HERE, "config_post.yaml" if self.name == "wing" else "config_post_plate.yaml")
        self.data = os.path.join(base, cfg.get("output", {}).get("save_path", "data/"))
        suffix = "" if self.name == "plate" else f"_{self.name}"
        self.forcing = os.path.join(HERE, f"forcing{suffix}")
        self.fom_data = os.path.join(HERE, f"fom_data{suffix}")
        self.figures = os.path.join(HERE, "figures" if self.name == "plate" else f"figures{suffix}")

    def body(self):
        """Marker file contents (xi, eta, zeta, dS, alpha[, thickness, span])."""
        return np.load(self.body_file)

    @property
    def alpha(self):
        b = self.body()
        return float(b["alpha"]) if "alpha" in b else 15.0

    def midspan_outline(self):
        """(x, y) of the body's cross-section at the mid-span plane, ordered
        for plotting: the plate's chord line, or the wing's closed section."""
        b = self.body()
        zeta = b["zeta"]
        mid = np.abs(zeta) <= np.abs(zeta).min() + 1e-12
        x, y = b["xi"][mid], b["eta"][mid]
        if self.name == "plate":
            o = np.argsort(x)
            return x[o], y[o]
        o = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
        return np.append(x[o], x[o][0]), np.append(y[o], y[o][0])

    def stations(self, offset=0.05):
        """Impulse stations [(label, x, y)].

        wing : the 2D airfoil's stations (examples/airfoil/plot_baseflow.py),
               body-frame (x/c, y/c) with the half-chord at the origin, mapped
               through the same nose-up rotation as the markers;
        plate: chord fraction s from the leading edge and normal offset d
               above the suction side of the plate rotated about its LE.
        """
        a = np.deg2rad(self.alpha)
        ca, sa = np.cos(a), np.sin(a)
        if self.name == "wing":
            body = (("leading edge", 0.02, 0.04), ("midchord", 0.50, 0.07),
                    ("trailing edge", 0.98, 0.02))
            return [(label, (xc - 0.5)*ca + yc*sa, -(xc - 0.5)*sa + yc*ca)
                    for label, xc, yc in body]
        return [(label, s*ca + offset*sa, -s*sa + offset*ca)
                for label, s in (("leading edge", 0.02), ("midchord", 0.50),
                                 ("trailing edge", 0.98))]
