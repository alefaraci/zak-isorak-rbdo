"""Cost functions c(d) of the benchmark problems.

Besides the cost itself, every cost function exposes a cost normalised to [0, 1] over the design
domain and a table of isolevels: quantiles of the normalised cost over the design domain.
ISORAK-RBDO uses them to restrict the search to isocost regions, i.e. the designs whose normalised
cost does not exceed a given isolevel.
"""

from abc import abstractmethod

from attrs import field, define
from torch import Tensor
import numpy as np
import torch


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class CostFunction:
    """Base class of the cost functions.

    The normalisation bounds and the isolevels are not computed here: the benchmark that owns
    the cost function fills them in when it is created.

    Attributes:
        _min_max: Minimum and maximum of the cost over the design domain, used by `fn_norm`.
        isolevels: Map from a probability level alpha to the alpha-quantile of the normalised
            cost over the design domain.
        _alpha: Level of the isocost region currently used by ISORAK-RBDO.
        _alpha_ub: Upper bound of the level, estimated by the searcher during ISORAK-RBDO.
    """

    _min_max: Tensor = field(init=True, default=torch.empty(0))
    isolevels: dict[float, float] = field(init=True, default={})
    _alpha: float = field(init=True, default=None)
    _alpha_ub: float = field(init=True, default=None)

    @property
    def alpha(self) -> float:
        """Level of the current isocost region."""
        return self._alpha

    @property
    def alpha_ub(self) -> float:
        """Upper bound of the level of the isocost region."""
        return self._alpha_ub

    @abstractmethod
    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the cost of the designs stored along the last dimension of X."""

    def fn_norm(self, X: Tensor) -> Tensor:
        """Evaluate the cost normalised with its minimum and maximum over the design domain.

        Args:
            X: Designs, stored along the last dimension.

        Returns:
            (c(X) - c_min) / (c_max - c_min), which lies in [0, 1] inside the design domain.
        """
        c_min, c_max = self._min_max
        return (self.fn(X) - c_min) / (c_max - c_min)

    def pop_in_isolevels(self, pop: Tensor, alpha: float) -> Tensor:
        """Flag the designs that belong to the isocost region of level alpha.

        Args:
            pop: Designs, stored along the last dimension.
            alpha: Level of the isocost region; it must be a key of `isolevels`.

        Returns:
            Boolean mask, True where the normalised cost does not exceed the isolevel.
        """
        return self.fn_norm(pop) <= self.isolevels[alpha]

    def visualize(
        self,
        design_bounds: Tensor,
        fig_size: tuple[float, float] | None = None,
        bench_id: str | None = None,
    ) -> None:
        """Plot the filled contours of the cost over a two-dimensional design domain.

        A new figure is created, with a colour bar.

        Args:
            design_bounds: Lower and upper bounds of the two plotted variables, one row each.
            fig_size: Not used: the figure always has the "squareColorbar" size.
            bench_id: Benchmark the axis labels are taken from. For nominal-tolerance designs
                the labels of the tolerances are used. Generic labels are used when omitted.
        """
        # Imported here because the plotter package itself imports the benchmarks, which import
        # this module.
        from .. import plotter  # noqa: PLC0415
        from ..benchmarks import register_benchmark  # noqa: PLC0415

        if bench_id:
            benchmark = register_benchmark[bench_id]()
            labels = (
                benchmark.label_d[int(benchmark.dim / 2) :]
                if benchmark.nominal_tolerance_design
                else benchmark.label_d
            )
            ylabel = (
                r"$\mathfrak{c}(\bm{t})$"
                if benchmark.nominal_tolerance_design or "linear_stackup" in benchmark.id
                else r"$\mathfrak{c}(\mathbf{d})$"
            )
        else:
            labels = [r"$d_1$", r"$d_2$"]
            ylabel = r"$\mathfrak{c}(\bm{d})$"
        fig_size = plotter.fig_size("squareColorbar")
        fig = plotter.plt.figure(figsize=fig_size)
        x1, x2, x1x2 = plotter.plot_2d.meshgrid_2D(
            x1=design_bounds[0],
            x2=design_bounds[1],
            res=200,
        )
        cf = plotter.plot_2d.contourfill(plotter.plt, x1, x2, self.fn(x1x2), levels=30)
        cs = plotter.plot_2d.contours(x1, x2, self.fn(x1x2), levels=30)
        cbar = fig.colorbar(cf, label=ylabel)
        cbar.add_lines(cs)
        plotter.plt.xlabel(labels[0])
        plotter.plt.ylabel(labels[1])
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()

    def visualize_mask_iso(self, design_bounds: Tensor) -> None:
        """Shade, on the current figure, the designs outside the current isocost region.

        The region is the one of level `alpha`; its boundary is drawn and labelled.

        Args:
            design_bounds: Lower and upper bounds of the two plotted variables, one row each.
        """
        from .. import plotter  # noqa: PLC0415

        x1, x2, x1x2 = plotter.plot_2d.meshgrid_2D(
            x1=design_bounds[0],
            x2=design_bounds[1],
            res=200,
        )
        plotter.plot_2d.contourf_cost_mask(
            plotter.plt,
            x1,
            x2,
            self.fn_norm(x1x2),
            self.isolevels[self._alpha],
            zorder=int(1e13),
        )
        cs = plotter.plot_2d.contour_cost_mask(
            plotter.plt,
            x1,
            x2,
            self.fn_norm(x1x2),
            self.isolevels[self._alpha],
            zorder=int(1e14),
        )
        plotter.plt.gca().clabel(cs, inline=True, fontsize=6.5, fmt=r"$\mathfrak{c}_{\alpha}$")

    def visualize_mask_iso_range(self, design_bounds: Tensor, alpha: float, delta: float) -> None:
        """Highlight, on the current figure, the designs between two isolevels.

        The band lies between the isolevels alpha and alpha + delta; the upper isolevel is drawn
        as a dashed, labelled line.

        Args:
            design_bounds: Lower and upper bounds of the two plotted variables, one row each.
            alpha: Level of the lower isolevel.
            delta: Increment of the level; alpha + delta must be a key of `isolevels`.
        """
        from .. import plotter  # noqa: PLC0415

        x1, x2, x1x2 = plotter.plot_2d.meshgrid_2D(
            x1=design_bounds[0],
            x2=design_bounds[1],
            res=600,
        )

        cost_norm = self.fn_norm(x1x2)
        cost_norm_2d = cost_norm.reshape(len(x1), len(x2))

        # The keys of `isolevels` are rounded floats: the sum is rounded to look the level up.
        mask = (cost_norm_2d > self.isolevels[alpha]) & (
            cost_norm_2d <= self.isolevels[round(alpha + delta, 4)]
        )

        # Mask everything outside the band so that only the band is filled.
        masked_values = np.ma.masked_where(~mask, cost_norm_2d)

        plotter.plt.contourf(
            x1,
            x2,
            masked_values,
            levels=[self.isolevels[alpha], self.isolevels[round(float(alpha + delta), 4)]],
            colors=["#FBCEB1"],
            alpha=0.36,
            zorder=int(1e13),
        )

        cs_upper = plotter.plt.contour(
            x1,
            x2,
            cost_norm_2d,
            levels=[self.isolevels[round(float(alpha + delta), 4)]],
            colors=["#C2B280"],
            linewidths=0.5,
            linestyles="dashed",
            zorder=int(1e14),
        )
        plotter.plt.gca().clabel(
            cs_upper,
            inline=True,
            fontsize=6.5,
            fmt=r"$\mathfrak{c}_{\alpha+\Delta\alpha}$",
        )


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class CircularParaboloid(CostFunction):
    """Circular paraboloid, cost of the highly nonlinear problem.

    c(d) = (d1 - 3.7)^2 + (d2 - 4)^2
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the circular paraboloid."""
        x1, x2 = X[..., 0], X[..., 1]
        return (x1 - 3.7) ** 2 + (x2 - 4.0) ** 2


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class SimpleSum(CostFunction):
    """Sum of the two design variables, cost of the three nonlinear problem.

    c(d) = d1 + d2
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the sum of the two design variables."""
        x1, x2 = X[..., 0], X[..., 1]
        return x1 + x2


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class EggHolder(CostFunction):
    """Scaled Eggholder function with the sign changed, cost of the four-branch problem.

    With (x1, x2) = (512 / 54) * (d1, d2),

    c(d) = (x2 + 47) * sin(sqrt(|x1 / 2 + x2 + 47|)) + x1 * sin(sqrt(|x1 - (x2 + 47)|))
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the scaled Eggholder function with the sign changed."""
        SCALE_FACTOR = 512.0 / (1.5 * 36.0)
        X = X * SCALE_FACTOR
        x1, x2 = X[..., 0], X[..., 1]
        return -1 * (
            -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x1 / 2 + (x2 + 47))))
            - x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
        )


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class SimpleMultiplication(CostFunction):
    """Product of the two design variables, cost of the column under compression.

    c(d) = d1 * d2, the area b * h of the cross-section of the column.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the product of the two design variables."""
        x1, x2 = X[..., 0], X[..., 1]
        return torch.mul(x1, x2)


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Bracket(CostFunction):
    """Mass of the bracket structure.

    With the design d = (w_AB, w_CD, t) given in centimetres and converted to metres,

    c(d) = rho * t * L * (4 * sqrt(3) / 9 * w_AB + w_CD),    rho = 7860, L = 5
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the mass of the bracket structure."""
        rho, L = 7860, 5
        wAB, wCD, t = X[..., 0] * 1e-2, X[..., 1] * 1e-2, X[..., 2] * 1e-2  # [m]
        return rho * t * L * (4 * torch.sqrt(torch.tensor(3)) / 9 * wAB + wCD)


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Truss(CostFunction):
    """Volume of material of the truss bridge.

    c(d) = L1 * d1 + L2 * d2,    L1 = (6 + 5) * 4, L2 = 12 * 2 * sqrt(2)

    d1 is the cross-section of the 11 horizontal bars, of length 4, and d2 the cross-section of
    the 12 diagonal bars, of length 2 * sqrt(2); L1 and L2 are their cumulated lengths.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the volume of material of the truss bridge."""
        L1, L2 = (6 + 5) * 4, 12 * 2.8284271247461903
        x1, x2 = X[..., 0], X[..., 1]
        return L1 * x1 + L2 * x2


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class Stackup(CostFunction):
    """Reciprocal tolerance-cost model of the linear stack-up problems.

    c(t) = 1 / t1 + 5 / t2, so that the cost decreases when the tolerances widen.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the reciprocal tolerance cost."""
        x1, x2 = X[..., 0], X[..., 1]
        return 1 / x1 + 5 / x2


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class ToleranceVolume(CostFunction):
    """Negative product of the tolerances, cost of the cantilever tolerance-design problems.

    c(t) = -prod_i t_i, so that minimising the cost maximises the volume of the tolerance box.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the negative product of the tolerances.

        Args:
            X: Tolerances only (not the nominal values): a single vector or a batch of shape
                (n, n_tolerances).
        """
        if X.dim() == 1:
            return -X.prod()
        return -X.prod(dim=1)
