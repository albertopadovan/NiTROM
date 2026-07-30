"""Shared matplotlib styling for NiTROM figures.

Call :func:`set_plot_style` once near the top of a plotting script to apply a
consistent, publication-quality look (serif/LaTeX fonts, inward ticks, etc.)
across the whole repository::

    from nitrom.plotting import set_plot_style
    set_plot_style()

``text.usetex`` is enabled, so a working LaTeX installation is required.
"""

import matplotlib.pyplot as plt

#: Publication-quality rcParams shared across NiTROM figures.
PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": True,
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "legend.handlelength": 2.8,
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

#: Qualitative colors used for the ROM families across figures.
COLORS = {
    "galerkin": "#1b9e77",
    "opinf": "#d95f02",
    "nitrom": "#386cb0",
}

STYLES = {
    "galerkin": (0, (1.0, 1.2)),
    "notgas": (0, (5.0, 2.4)),
    "gas": "solid"
}


def set_plot_style() -> None:
    """Apply :data:`PAPER_RCPARAMS` (and the amsmath LaTeX preamble) globally."""
    plt.rcParams.update(PAPER_RCPARAMS)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
