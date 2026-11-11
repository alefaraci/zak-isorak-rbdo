"""Figures of the package: shared styles, figure helpers and post-processing plots.

Importing this package applies the Matplotlib settings stored in `rc_params`. They render all
text with LaTeX, so a LaTeX installation is required to produce the figures.
"""

from typing import Any

import matplotlib.pyplot as plt

from . import plot_2d, plot_tolerance_2d
from .utils import (
    save,
    fig_size,
    set_square_ratio,
    set_scientific_notation,
)
from .boxplot_comparator import ModelComparison

__all__ = [
    "ModelComparison",
    "fig_size",
    "plot_2d",
    "plot_tolerance_2d",
    "plt",
    "save",
    "set_scientific_notation",
    "set_square_ratio",
]


# Axis limits of the zoomed views of the design space, for each analysis and benchmark.
register_zoom: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "ddo": {
        "highly_nonlinear": {
            "zoom_xlim": (2.4, 3.4),
            "zoom_ylim": (2.7, 3.7),
        },
        "three_nonlinear": {
            "zoom_xlim": (1.7, 5),
            "zoom_ylim": (1.55, 5.8),
        },
        "four_branches": {
            "zoom_xlim": (-5.4, -1.2),
            "zoom_ylim": (-2.5, 1.2),
        },
        "column_under_compression": {
            "zoom_xlim": (200, 300),
            "zoom_ylim": (200, 300),
        },
    },
    "rbdo": {
        "highly_nonlinear": {
            "zoom_xlim": (2.4, 3.4),
            "zoom_ylim": (2.7, 3.7),
        },
        "three_nonlinear": {
            "zoom_xlim": (1.7, 5),
            "zoom_ylim": (1.55, 5.8),
        },
        "four_branches": {
            "zoom_xlim": (-5.4, -1.2),
            "zoom_ylim": (-2.5, 1.2),
        },
        "column_under_compression": {
            "zoom_xlim": (200, 300),
            "zoom_ylim": (200, 300),
        },
        "linear_stackup": {
            "zoom_xlim": (5, 7),
            "zoom_ylim": (3, 5),
        },
        "cantilever_2d": {
            "zoom_xlim": (0.5, 2),
            "zoom_ylim": (5, 17.5),
        },
        "cantilever_2d_mean_shift": {
            "zoom_xlim": (0.5, 2),
            "zoom_ylim": (5, 17.5),
        },
    },
    "zakrbdo": {
        "highly_nonlinear": {
            "zoom_xlim": (2.4, 3.4),
            "zoom_ylim": (2.7, 3.7),
        },
        "three_nonlinear": {
            "zoom_xlim": (1.7, 5),
            "zoom_ylim": (1.55, 5.8),
        },
        "four_branches": {
            "zoom_xlim": (-5.4, -1.2),
            "zoom_ylim": (-2.5, 1.2),
        },
        "column_under_compression": {
            "zoom_xlim": (210, 260),
            "zoom_ylim": (210, 260),
        },
        "linear_stackup": {
            "zoom_xlim": (5, 7),
            "zoom_ylim": (3, 5),
        },
        "cantilever_2d": {
            "zoom_xlim": (0.5, 2),
            "zoom_ylim": (5, 17.5),
        },
        "cantilever_2d_mean_shift": {
            "zoom_xlim": (0.5, 2),
            "zoom_ylim": (5, 17.5),
        },
        "truss_bridge": {
            "zoom_xlim": (0.001, 0.003),
            "zoom_ylim": (0.0003, 0.0015),
        },
    },
    "isorakrbdo": {
        "highly_nonlinear": {
            "zoom_xlim": (2.4, 3.4),
            "zoom_ylim": (2.7, 3.7),
        },
        "three_nonlinear": {
            "zoom_xlim": (1.7, 5),
            "zoom_ylim": (1.55, 5.8),
        },
        "four_branches": {
            "zoom_xlim": (-5.4, -1.2),
            "zoom_ylim": (-2.5, 1.2),
        },
        "column_under_compression": {
            "zoom_xlim": (200, 300),
            "zoom_ylim": (200, 300),
        },
        "linear_stackup": {
            "zoom_xlim": (5, 7),
            "zoom_ylim": (3, 5),
        },
        "cantilever_2d": {
            "zoom_xlim": (0.5, 2),
            "zoom_ylim": (5, 17.5),
        },
        "cantilever_2d_mean_shift": {
            "zoom_xlim": (0.5, 2),
            "zoom_ylim": (5, 17.5),
        },
        "truss_bridge": {
            "zoom_xlim": (0.001, 0.003),
            "zoom_ylim": (0.0003, 0.0015),
        },
    },
}

# Line styles: "0" to "7" for the successive curves of a plot, "Pf_thr" for the threshold lines
# and "trace" for the path followed by the best design.
plot_style: dict[str, dict[str, Any]] = {
    "0": {
        "color": "#D11942",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "1": {
        "color": "#008C8C",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "2": {
        "color": "#A6BFCB",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "3": {
        "color": "#564787",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "4": {
        "color": "#F7B538",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "5": {
        "color": "#19285F",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "6": {
        "color": "#1F964E",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "7": {
        "color": "#FFA07A",
        "linestyle": "-",
        "linewidth": 0.85,
        "marker": ".",
        "markersize": 1.55,
    },
    "Pf_thr": {
        "color": "k",
        "linestyle": "--",
        "linewidth": 0.55,
    },
    "trace": {
        "c": "#0C6F84",
        "zorder": 1116,
        "lw": 0.3,
    },
}

# Marker styles of the scatter plots. `scale_factor` shrinks the markers that are drawn in large
# numbers (training points, enrichment points, individuals of the population).
scale_factor = 0.816 * 0.821**1.2
scatter_style: dict[str, dict[str, Any]] = {
    "best": {
        "s": 24,
        "c": "#4FFA7B",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35,
    },
    "mcs_best": {
        "s": 24,
        "c": "#FFFFFF",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35,
    },
    "mcs_safe": {
        "s": 2.175,
        "marker": ".",
        "linewidths": 0,
        "color": "#3F9F4F",
        "zorder": 100,
        "alpha": 0.325,
        "rasterized": True,
    },
    "mcs_fail": {
        "s": 2.175,
        "marker": ".",
        "linewidths": 0,
        "color": "#D32A00",
        "zorder": 100,
        "alpha": 0.325,
        "rasterized": True,
    },
    "nebula": {
        "s": 2.175,
        "marker": ".",
        "linewidths": 0,
        "color": "#A56EF9",
        "zorder": 10,
        "alpha": 0.075,
        "rasterized": True,
    },
    "nebula_best": {
        "s": 2.175,
        "marker": ".",
        "linewidths": 0,
        "color": "#5D24D4",
        "zorder": 9995,
        "alpha": 0.075,
        "rasterized": True,
    },
    "train": {
        "s": 24 * scale_factor,
        "c": "#3FA2B7",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "train_new": {
        "s": 24 * scale_factor,
        "c": "#5CFFF5",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "enrich_base_circle_1": {
        "s": 45 * scale_factor,
        "c": "#FFFF80",
        "zorder": 10001,
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "enrich_base_circle_2": {
        "s": 45 * scale_factor,
        "c": "#FF8F7F",
        "zorder": 10001,
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "enrich_base_circle_3": {
        "s": 45 * scale_factor,
        "c": "#FF94D0",
        "zorder": 10001,
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "enrich_base_circle_4": {
        "s": 45 * scale_factor,
        "c": "#00D896",
        "zorder": 10001,
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "enrich_base_circle_5": {
        "s": 45 * scale_factor,
        "c": "#25BADA",
        "zorder": 10001,
        "edgecolor": "black",
        "linewidth": 0.35 * (scale_factor + scale_factor / 4),
    },
    "nom_tol_1": {
        "s": 24,
        "c": "#D2A6FF",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35,
    },
    "nom_tol_2": {
        "s": 24,
        "c": "#F7B538",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35,
    },
    "nom_tol_3": {
        "s": 24,
        "c": "#564787",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35,
    },
    "nom_tol_4": {
        "s": 24,
        "c": "#7AD9FF",
        "zorder": 10001,
        "marker": "P",
        "edgecolor": "black",
        "linewidth": 0.35,
    },
    "trace": {
        "s": 24,
        "c": "#3EA2B7",
        "marker": ".",
        "zorder": 1117,
        "edgecolor": "black",
    },
    "ga_individual": {
        "s": 20 * scale_factor,
        "c": "#9AA2CC",
        "zorder": 9990,
        "marker": ".",
        "edgecolor": "#737AA2",
        "linewidth": 0.3 * scale_factor,
        "alpha": 0.7,
    },
    "F_label": {
        "c": "k",
        "s": 12,
        "marker": r"$\mathcal{A}$",
        "zorder": 1e9,
    },
    "U_label": {
        "c": "k",
        "s": 12,
        "marker": r"$\mathrm{U}$",
        "zorder": 1e9,
    },
}

# Matplotlib settings applied when the package is imported.
rc_params = {
    # DPI
    "figure.dpi": 140,
    # LaTeX
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath} \usepackage{bm} \usepackage{physics}",
    # Axes
    "axes.linewidth": 0.3,
    # Grid
    "grid.color": "black",
    "grid.linestyle": ":",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.25,
    "xtick.major.width": 0.25,
    "xtick.minor.width": 0.20,
    "ytick.major.width": 0.25,
    "ytick.minor.width": 0.20,
    # Legend
    "legend.edgecolor": "black",
    "patch.linewidth": 0.25,
    "legend.fontsize": 7.5,
    "legend.fancybox": False,
    # Linewidth
    "lines.linewidth": 0.5,
    # Label
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

plt.rcParams.update(rc_params)
