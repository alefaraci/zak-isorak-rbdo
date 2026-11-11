"""Low-level 2D contour helpers shared by every plot of the package.

Each helper receives the two coordinate grids returned by :func:`meshgrid_2D` together with the
values of a scalar field on the flattened grid, and draws on the current matplotlib axes.

The region helpers paint one side of a level set with a flat colour. They call ``contourf`` with a
single, vanishingly thin band ``[level, level + 1e-06]`` and let the ``extend`` colour fill
everything above it (``extend="max"``, "over" colour) or below it (``extend="min"``, "under"
colour); the band itself is too thin to be visible.
"""

from typing import Any

from torch import Tensor
from colour import Color
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.contour import QuadContourSet
import torch
import matplotlib.pyplot as plt


def cmapLightBlue() -> LinearSegmentedColormap:
    """Return the light-blue colormap used for the cost contours."""
    CustomCMAP = ["#E9F0FA", "#6D88A6"]
    return LinearSegmentedColormap.from_list("my_list", [Color(c1).rgb for c1 in CustomCMAP])


def meshgrid_2D(
    x1: Tensor,
    x2: Tensor,
    res: int = 100,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a regular grid over a rectangle.

    Args:
        x1: Lower and upper bound of the first coordinate.
        x2: Lower and upper bound of the second coordinate.
        res: Number of grid points per axis.

    Returns:
        The two ``(res, res)`` coordinate grids (``"xy"`` indexing) and the same points flattened
        to a ``(res * res, 2)`` tensor, ready to be passed to a test function.
    """
    X1 = torch.linspace(x1[0], x1[1], res)
    X2 = torch.linspace(x2[0], x2[1], res)
    X1, X2 = torch.meshgrid(X1, X2, indexing="xy")
    X1X2 = torch.stack((X1, X2), dim=-1).reshape(-1, 2)
    return X1, X2, X1X2


def contourf_cost_mask(plt, x1, x2, f_x1x2, level, alpha=0.665, zorder=9996) -> QuadContourSet:
    """Grey out the region where the field exceeds ``level``.

    Used to mask the designs lying outside the current iso-cost region.
    """
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="max",
        alpha=alpha,
        zorder=zorder,
    )
    cs.cmap.set_over("#586471")
    cs.changed()
    return cs


def contour_cost_mask(plt, x1, x2, f_x1x2, level, color="#586471", zorder=9996) -> QuadContourSet:
    """Draw the contour line of the field at ``level`` (the iso-cost boundary)."""
    return plt.contour(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level],
        colors=color,
        linewidths=0.5,
        alpha=0.87,
        linestyles="solid",
        zorder=zorder,
    )


def safe_region(plt, x1, x2, f_x1x2, level=0, alpha=1) -> QuadContourSet:
    """Fill in green the region where the field is above ``level`` (safe domain, g > 0)."""
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="max",
        alpha=alpha,
    )
    cs.cmap.set_over("#D4EECC")
    cs.cmap.set_under("#FFC8C8")
    cs.changed()
    return cs


def failure_region(plt, x1, x2, f_x1x2, level=0, alpha=1) -> QuadContourSet:
    """Fill in red the region where the field is below ``level`` (failure domain, g <= 0)."""
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="min",
        alpha=alpha,
    )
    cs.cmap.set_over("#D4EECC")
    cs.cmap.set_under("#FFC8C8")
    cs.changed()
    return cs


def feasible_region(plt, x1, x2, f_x1x2, level=0, alpha=1) -> QuadContourSet:
    """Fill in green the region where the field is below ``level`` (constraint satisfied)."""
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="min",
        alpha=alpha,
    )
    cs.cmap.set_under("#D4EECC")
    cs.cmap.set_over("#FFC8C8")
    cs.changed()
    return cs


def unconformity_region(plt, x1, x2, f_x1x2, level=0, alpha=1) -> QuadContourSet:
    """Fill in light grey the region where the field is below ``level`` (non-conformity region)."""
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="min",
        alpha=alpha,
    )
    cs.cmap.set_over("#F0F0F0")
    cs.cmap.set_under("#F0F0F0")
    cs.changed()
    return cs


def conformity_area(plt, x1, x2, f_x1x2, level=0, alpha=1) -> QuadContourSet:
    """Fill in dark grey the region where the field is below ``level`` (conformity area)."""
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="min",
        alpha=alpha,
    )
    cs.cmap.set_under("#B8B8B9")
    cs.cmap.set_over("#B8B8B9")
    cs.changed()
    return cs


def not_feasible_region(
    plt,
    x1,
    x2,
    f_x1x2,
    level=0,
    alpha: float = 1,
) -> QuadContourSet:
    """Fill in blue the region where the field is above ``level`` (soft constraint violated)."""
    cs = plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=[level, level + 1e-06],
        colors=["#808080", "#A0A0A0"],
        extend="max",
        alpha=alpha,
    )

    cs.cmap.set_under("#D4EECC")
    cs.cmap.set_over("#0077B6")
    cs.changed()
    return cs


def contour(x1, x2, f_x1x2, color="#fa4931", **kwargs: Any) -> QuadContourSet:
    """Draw the boundary of a constraint (limit state or soft constraint) as a single line.

    ``levels=0`` is read by matplotlib as a number of levels, not as the value zero: the level is
    picked by its automatic locator, which lands on zero whenever the field changes sign over the
    grid.

    Args:
        x1: Grid of the first coordinate.
        x2: Grid of the second coordinate.
        f_x1x2: Field values on the flattened grid.
        color: Line colour.
        **kwargs: Forwarded to ``matplotlib.pyplot.contour`` (e.g. ``linestyles``, ``zorder``).

    Returns:
        The matplotlib contour set, which can be passed to ``clabel``.
    """
    return plt.contour(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=0,
        colors=color,
        linewidths=0.5,
        **kwargs,
    )


def contours(x1, x2, f_x1x2, levels=100, cmap=None) -> QuadContourSet:
    """Draw ``levels`` iso-lines of the field, light blue by default (cost contours)."""
    cmap = cmapLightBlue() if cmap is None else cmap
    return plt.contour(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=levels,
        cmap=cmap,
        linewidths=0.5,
    )


def contourfill(
    plt,
    x1,
    x2,
    f_x1x2,
    levels=100,
    cmap=None,
    **kwargs: Any,
) -> QuadContourSet:
    """Draw the filled contours of the field, light blue by default (cost map)."""
    cmap = cmapLightBlue() if cmap is None else cmap
    return plt.contourf(
        x1,
        x2,
        f_x1x2.reshape(len(x1), len(x2)),
        levels=levels,
        cmap=cmap,
        alpha=0.75,
        **kwargs,
    )
