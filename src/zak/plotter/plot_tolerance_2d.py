"""Dimensionless backgrounds for the tolerance-design plots.

Both functions open a new square figure and draw only the background regions; the callers add the
points of the designs on top. Starred quantities are scaled by the bounds of the corresponding
design variable, so that every dimension of a benchmark can be shown in the same plane.
"""

import torch

from .. import plotter


def plot_nom_tol_adim_space(*, mean_shift: bool = False) -> None:
    """Draw the dimensionless nominal/tolerance plane of the cantilever beam benchmarks.

    The unit square (T*, t*) is split by the two lines ``t* = 2 T*`` and ``t* = 2 - 2 T*``: the
    triangle below both of them, where the interval ``[T* - t*/2, T* + t*/2]`` stays inside
    ``[0, 1]``, is filled in green and the rest in red.

    Args:
        mean_shift: Label the two boundaries with the expressions that include the mean shift
            ``|delta*|``. The regions themselves are the same in both cases.
    """
    x1, x2, x1x2 = plotter.plot_2d.meshgrid_2D(x1=torch.tensor([0, 1]), x2=torch.tensor([0, 1]))
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
    y1 = x1x2[:, 1] - 2 * x1x2[:, 0]
    y2 = x1x2[:, 1] + 2 * x1x2[:, 0] - 2
    plotter.plot_2d.feasible_region(plotter.plt, x1, x2, y1)
    plotter.plot_2d.feasible_region(plotter.plt, x1, x2, y2)
    plotter.plot_2d.failure_region(plotter.plt, x1, x2, -y1)
    plotter.plot_2d.failure_region(plotter.plt, x1, x2, -y2)
    cs1 = plotter.plot_2d.contour(x1, x2, y1)
    cs2 = plotter.plot_2d.contour(x1, x2, y2)
    manual_pos1 = [(0.2, 0.4)]
    manual_pos2 = [(0.6, 0.4)]
    fmt_1 = r"$T^*-\frac{t^*}{2}$"
    fmt_1_delta = r"$T^*-|\delta^*|-\frac{t^*}{2}$"
    fmt_2 = r"$T^*+\frac{t^*}{2}$"
    fmt_2_delta = r"$T^*+|\delta^*|+\frac{t^*}{2}$"

    plotter.plt.gca().clabel(
        cs1,
        inline=True,
        fontsize=6.5,
        fmt=fmt_1 if not mean_shift else fmt_1_delta,
        manual=manual_pos1,
    )
    plotter.plt.gca().clabel(
        cs2,
        inline=True,
        fontsize=6.5,
        fmt=fmt_2 if not mean_shift else fmt_2_delta,
        manual=manual_pos2,
    )
    plotter.plt.xlabel(r"$T^*$")
    plotter.plt.ylabel(r"$t^*$")


def plot_tol_mean_shift_adim_space() -> None:
    """Draw the dimensionless mean-shift plane of the benchmarks with mean shift.

    The plane has the mean shift ``delta*`` in [-1, 1] on the horizontal axis and ``sigma*`` in
    [0, 1] on the vertical one. The conformity triangle below the two lines
    ``sigma* = 1 + 2 delta*`` and ``sigma* = 1 - 2 delta*`` is filled in dark grey and the
    non-conformity region in light grey.
    """
    x1, x2, x1x2 = plotter.plot_2d.meshgrid_2D(
        x1=torch.tensor([-1, 1]),
        x2=torch.tensor([0, 1]),
    )
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
    y1 = x1x2[:, 1] - 2 * x1x2[:, 0] - 1
    y2 = x1x2[:, 1] + 2 * x1x2[:, 0] - 1
    plotter.plot_2d.conformity_area(plotter.plt, x1, x2, y1)
    plotter.plot_2d.conformity_area(plotter.plt, x1, x2, y2)
    plotter.plot_2d.unconformity_region(plotter.plt, x1, x2, -y1)
    plotter.plot_2d.unconformity_region(plotter.plt, x1, x2, -y2)
    _ = plotter.plot_2d.contour(x1, x2, y1, color="#282C33")
    _ = plotter.plot_2d.contour(x1, x2, y2, color="#282C33")
    plotter.plt.xlabel(r"$\delta^*$")
    plotter.plt.ylabel(r"$\sigma^*$")
