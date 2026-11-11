"""Deterministic constraints f_j(d) of the benchmark problems (the "soft" constraints).

Sign convention: a constraint is satisfied when f <= 0 and violated when f > 0. These constraints
depend on the design only and are cheap to evaluate, unlike the limit-state functions.
"""

from abc import abstractmethod

from attrs import field, define
from torch import Tensor
import torch


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class SoftConstraint:
    """Base class of the deterministic constraints."""

    @abstractmethod
    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the constraint at the designs stored along the last dimension of X.

        Returns:
            One value per design: the constraint is satisfied when the value is <= 0.
        """

    def visualize(self, design_bounds: Tensor) -> None:
        """Shade, on the current figure, the designs of a 2D domain that violate the constraint.

        The boundary f = 0 is drawn and labelled.

        Args:
            design_bounds: Lower and upper bounds of the two plotted variables, one row each.
        """
        # Imported here because the plotter package itself imports the benchmarks, which import
        # this module.
        from .. import plotter  # noqa: PLC0415

        x1, x2, x1x2 = plotter.plot_2d.meshgrid_2D(
            x1=design_bounds[0],
            x2=design_bounds[1],
            res=200,
        )
        _ = plotter.plot_2d.not_feasible_region(plotter.plt, x1, x2, self.fn(x1x2), alpha=0.275)
        cs = plotter.plot_2d.contour(x1, x2, self.fn(x1x2), color="#0277B6")
        string = rf"$\mathbf{{\mathfrak{{f_{1}}}}}$"
        plotter.plt.gca().clabel(cs, inline=True, fontsize=6.5, fmt=string)


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Greater(SoftConstraint):
    """Ordering constraint d1 >= d2.

    f(d) = d2 - d1
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate f(d) = d2 - d1."""
        x1, x2 = X[..., 0], X[..., 1]
        return x2 - x1


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class NominalToleranceGreater(SoftConstraint):
    """Lower-bound constraint of a nominal-tolerance design.

    The design vector stores the nominal values T followed by the tolerances t. For the variable
    of index `var`, the lower end of the tolerance interval must not fall below the lower bound
    of the variable:

    f(d) = lb - T + t / 2

    Attributes:
        var: Index of the constrained variable.
        bounds_xvar: Lower and upper bounds of the variables, one row per variable.
    """

    var: int
    bounds_xvar: Tensor

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate f(d) = lb - T + t / 2 for the variable of index `var`."""
        # First half of the design vector: nominal values; second half: tolerances.
        nominal, tolerance = (
            X[..., : int(X.shape[-1] / 2)],
            X[..., int(X.shape[-1] / 2) :],
        )
        f = self.bounds_xvar.t()[0].to(nominal.device) - nominal + tolerance / 2
        return f[..., self.var]


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class NominalToleranceLower(SoftConstraint):
    """Upper-bound constraint of a nominal-tolerance design.

    The design vector stores the nominal values T followed by the tolerances t. For the variable
    of index `var`, the upper end of the tolerance interval must not exceed the upper bound of
    the variable:

    f(d) = T + t / 2 - ub

    Attributes:
        var: Index of the constrained variable.
        bounds_xvar: Lower and upper bounds of the variables, one row per variable.
    """

    var: int
    bounds_xvar: Tensor

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate f(d) = T + t / 2 - ub for the variable of index `var`."""
        # First half of the design vector: nominal values; second half: tolerances.
        nominal, tolerance = (
            X[..., : int(X.shape[-1] / 2)],
            X[..., int(X.shape[-1] / 2) :],
        )
        f = -self.bounds_xvar.t()[1].to(nominal.device) + nominal + tolerance / 2
        return f[..., self.var]


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Cantilever2dNominalToleranceGreater(NominalToleranceGreater):
    """Lower-bound constraint of the 2D cantilever problem, with the bounds of (h, L)."""

    bounds_xvar: Tensor = field(default=torch.tensor([[0.5, 2.0], [5.0, 20.0]]))


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Cantilever2dNominalToleranceLower(NominalToleranceLower):
    """Upper-bound constraint of the 2D cantilever problem, with the bounds of (h, L)."""

    bounds_xvar: Tensor = field(default=torch.tensor([[0.5, 2.0], [5.0, 20.0]]))


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Cantilever4dNominalToleranceGreater(NominalToleranceGreater):
    """Lower-bound constraint of the 4D cantilever problem, with the bounds of (h, L, b, f)."""

    bounds_xvar: Tensor = field(
        default=torch.tensor([[0.5, 2.0], [5.0, 20.0], [1.0, 3], [0.5, 1.5]]),
    )


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Cantilever4dNominalToleranceLower(NominalToleranceLower):
    """Upper-bound constraint of the 4D cantilever problem, with the bounds of (h, L, b, f)."""

    bounds_xvar: Tensor = field(
        default=torch.tensor([[0.5, 2.0], [5.0, 20.0], [1.0, 3], [0.5, 1.5]]),
    )
