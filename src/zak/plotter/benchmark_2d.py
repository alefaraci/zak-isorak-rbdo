"""2D contour plots of a benchmark: cost, limit states (true or surrogate) and soft constraints."""

from collections.abc import Sequence

from attrs import field, define
from torch import Tensor
from shapely.geometry import Polygon, box
from shapely.plotting import plot_polygon
from matplotlib.patches import Rectangle
import torch

from zak.benchmarks import BenchmarkBase

from . import plt, plot_2d, fig_size, set_square_ratio
from ..gaussianprocess.gp import GPRBoTorch
from ..test_functions.hard import HardConstraint


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class BaseModel2D:
    """2D view of a benchmark in the plane of its first two variables.

    Attributes:
        benchmark: Benchmark providing the bounds, the cost and the soft constraints.
        hard: Limit states to draw. Defaults to the true limit states of the benchmark; pass the
            Kriging surrogates instead to draw their mean predictions.
    """

    benchmark: BenchmarkBase
    hard: Sequence[HardConstraint] | list[GPRBoTorch] | None = field(init=True, default=None)

    def __attrs_post_init__(self) -> None:
        """Fall back to the true limit states of the benchmark when none are given."""
        if self.hard is None and self.benchmark.hard is not None:
            self.hard = self.benchmark.hard

    def _get_label(self, k) -> str:
        """Return the contour label of the k-th limit state (hatted for a surrogate)."""
        str_true_model = rf"$\mathbf{{\mathfrak{{g_{k + 1}}}}}$"
        str_surrogate = rf"$\mathbf{{\hat{{\mathfrak{{g}}}}_{k + 1}}}$"
        if self.hard is not None and isinstance(self.hard[k], HardConstraint):
            return str_true_model
        return str_surrogate

    @property
    def bounds(self) -> Tensor:
        """Return design space bounds."""
        return self.benchmark.bounds

    @property
    def dim(self) -> int:
        """Return the number of design variables."""
        return self.benchmark.dim

    @property
    def aug_bounds(self) -> Tensor:
        """Return augmented space bounds."""
        return self.benchmark.aug_bounds

    def true_model(self, color="k") -> None:
        """Overlay the limit states held by this instance as dashed, labelled contours.

        The contours are drawn over the augmented bounds on the current axes, on top of a figure
        created by :meth:`visualize`. Despite the name, the method draws whatever ``hard`` contains:
        the post-processing calls it on the surrogate view to superimpose the Kriging limit states
        on the true ones.

        Args:
            color: Colour of the contour lines.
        """
        x1, x2, x1x2 = plot_2d.meshgrid_2D(x1=self.aug_bounds[0], x2=self.aug_bounds[1])
        if self.hard is not None:
            for k, h in enumerate(self.hard):
                try:
                    cs = plot_2d.contour(
                        x1,
                        x2,
                        h.fn(x1x2).cpu(),
                        color=color,
                        linestyles="dashed",
                        zorder=9996,
                    )
                except Exception as _:
                    # The limit state has more inputs than the two plotted ones: the environmental
                    # variables are frozen at their mean values to obtain a 2D slice.
                    if self.benchmark.id == "column_under_compression":
                        loc_z = torch.tensor([0.6, 1e4, 3.0e3])
                    else:
                        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
                    x1x2z = torch.cat([x1x2, loc_z.expand(x1x2.shape[0], -1)], dim=1)
                    cs = plot_2d.contour(
                        x1,
                        x2,
                        h.fn(x1x2z).cpu(),
                        color=color,
                        linestyles="dashed",
                        zorder=9996,
                    )
                plt.gca().clabel(cs, inline=True, fontsize=6.5, fmt=self._get_label(k))

    def visualize(
        self,
        *,
        color: str = "#fa4931",
        linestyles: str = "solid",
        augmented_space: bool = False,
        nominal_tolerance: bool = False,
    ) -> None:
        """Open a new figure with the complete 2D picture of the benchmark.

        In drawing order: the cost contours with their colour bar, the iso-cost mask (ISORAK-RBDO
        only), the safe (green) and failure (red) domains of every limit state, the labelled limit
        states, the infeasible regions and boundaries of the soft constraints and, in the augmented
        space, the outline of the design box.

        Args:
            color: Colour of the limit-state contours.
            linestyles: Line style of the limit-state contours.
            augmented_space: Draw over the augmented bounds (design space enlarged by the scatter
                of the random variables) and label the axes with the random variables, instead of
                the design bounds and the design variables.
            nominal_tolerance: The design vector is ``[nominal values, tolerances]``: skip the cost
                contours and the soft constraints, which are not functions of the plotted plane.
        """
        bounds = self.benchmark.aug_bounds if augmented_space else self.benchmark.bounds
        # column_under_compression keeps the design-variable labels and gets no design-box overlay,
        # although the grid built from `bounds` above already spans its augmented bounds.
        augmented_space = (
            False if self.benchmark.id == "column_under_compression" else augmented_space
        )
        fig = plt.figure(figsize=fig_size("squareColorbar"))
        x1, x2, x1x2 = plot_2d.meshgrid_2D(x1=bounds[0], x2=bounds[1])
        # In the tolerance benchmarks the cost is a function of the tolerances, whereas this figure
        # is meant for the plane of the toleranced dimensions: no cost contours there.
        if not nominal_tolerance and "linear_stackup" not in self.benchmark.id:
            x1_c, x2_c, x1x2_c = plot_2d.meshgrid_2D(x1=bounds[0], x2=bounds[1], res=200)
            cs = plot_2d.contours(x1_c, x2_c, self.benchmark.cost.fn(x1x2_c), levels=30)
            fig.colorbar(cs, label=r"$\mathfrak{c}(\mathbf{d})$")
        # alpha stays at 1 unless an iso-cost level is active (ISORAK-RBDO), in which case the
        # designs beyond that level are greyed out.
        if self.benchmark.cost._alpha < 1.0 and not (
            self.benchmark.nominal_tolerance_design or "linear_stackup" in self.benchmark.id
        ):
            self.benchmark.cost.visualize_mask_iso(design_bounds=self.benchmark.bounds)
        # All the safe domains are filled before all the failure domains, so that with several
        # limit states the failure domain of one is never hidden by the safe domain of another.
        if self.hard is not None:
            for _, h in enumerate(self.hard):
                try:
                    plot_2d.safe_region(plt, x1, x2, h.fn(x1x2).cpu())
                except Exception as _:
                    # Limit state with more than two inputs: environmental variables frozen at
                    # their mean values (see `true_model`).
                    if self.benchmark.id == "column_under_compression":
                        loc_z = torch.tensor([0.6, 1e4, 3.0e3])
                    else:
                        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
                    x1x2z = torch.cat([x1x2, loc_z.expand(x1x2.shape[0], -1)], dim=1)
                    plot_2d.safe_region(plt, x1, x2, h.fn(x1x2z).cpu())
            for _, h in enumerate(self.hard):
                try:
                    plot_2d.failure_region(plt, x1, x2, h.fn(x1x2).cpu())
                except Exception as _:
                    if self.benchmark.id == "column_under_compression":
                        loc_z = torch.tensor([0.6, 1e4, 3.0e3])
                    else:
                        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
                    x1x2z = torch.cat([x1x2, loc_z.expand(x1x2.shape[0], -1)], dim=1)
                    plot_2d.failure_region(plt, x1, x2, h.fn(x1x2z).cpu())
            for k, h in enumerate(self.hard):
                try:
                    cs = plot_2d.contour(
                        x1,
                        x2,
                        h.fn(x1x2).cpu(),
                        color=color,
                        linestyles=linestyles,
                    )
                except Exception as _:
                    if self.benchmark.id == "column_under_compression":
                        loc_z = torch.tensor([0.6, 1e4, 3.0e3])
                    else:
                        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
                    x1x2z = torch.cat([x1x2, loc_z.expand(x1x2.shape[0], -1)], dim=1)
                    cs = plot_2d.contour(
                        x1,
                        x2,
                        h.fn(x1x2z).cpu(),
                        color=color,
                        linestyles=linestyles,
                    )
                plt.gca().clabel(cs, inline=True, fontsize=6.5, fmt=self._get_label(k))
        if not nominal_tolerance and self.benchmark.soft is not None:
            for _, s in enumerate(self.benchmark.soft):
                _ = (
                    plot_2d.not_feasible_region(plt, x1, x2, s.fn(x1x2), alpha=0.275)
                    if s is not None
                    else None
                )
            for i, s in enumerate(self.benchmark.soft):
                if s is not None:
                    cs = plot_2d.contour(x1, x2, s.fn(x1x2), color="#0277B6")
                    string = rf"$\mathbf{{\mathfrak{{f_{i + 1}}}}}$"
                    plt.gca().clabel(cs, inline=True, fontsize=6.5, fmt=string)
        labels = self.benchmark.label_X if augmented_space else self.benchmark.label_d
        if augmented_space:
            # The linear stack-up designs are tolerances: there is no design box to outline in the
            # plane of the dimensions.
            if "linear_stackup" in self.benchmark.id:
                pass
            else:
                self.__plot_augmented_space()
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        set_square_ratio(plt.gca())
        plt.tight_layout()

    def __plot_augmented_space(self) -> None:
        """Outline the design box inside a figure drawn over the augmented bounds.

        The margin between the design bounds and the augmented bounds is shaded, the design box is
        outlined and labelled with the design variables along its edges, and the axes limits are
        set to the augmented bounds.
        """
        origin = (
            float(self.benchmark.bounds[0][0]),
            float(self.benchmark.bounds[1][0]),
        )
        d1_len = float(abs(self.benchmark.bounds[0][1] - self.benchmark.bounds[0][0]))
        d2_len = float(abs(self.benchmark.bounds[1][1] - self.benchmark.bounds[1][0]))
        rect = [
            box(
                float(self.benchmark.aug_bounds[0][0]),
                float(self.benchmark.aug_bounds[1][0]),
                float(self.benchmark.aug_bounds[0][1]),
                float(self.benchmark.aug_bounds[1][1]),
            ),
            box(
                float(self.benchmark.bounds[0][0]),
                float(self.benchmark.bounds[1][0]),
                float(self.benchmark.bounds[0][1]),
                float(self.benchmark.bounds[1][1]),
            ),
        ]
        # Augmented rectangle with the design box as a hole: only the margin gets shaded.
        polygon = Polygon(
            list(rect[0].exterior.coords),
            [list(rect[1].exterior.coords)[::-1]],
        )
        plot_polygon(
            polygon,
            ax=plt.gca(),
            add_points=False,
            color="#687E95",
            alpha=0.5,
            zorder=1e11,
        )
        if "linear_stackup" not in self.benchmark.id:
            plt.gca().add_patch(
                Rectangle(
                    origin,
                    d1_len,
                    d2_len,
                    edgecolor="#457AA8",
                    lw=0.5,
                    fill=False,
                    zorder=1e10,
                ),
            )
            offset_0 = 0.15 if self.benchmark.nominal_tolerance_design else 0.085
            offset_1 = 0.15 if self.benchmark.nominal_tolerance_design else 0.085
            if "truss" in self.benchmark.id:
                offset_0 = 0.05

            plt.gca().text(
                origin[0] + d1_len / 2,
                origin[1] - offset_0 * (d2_len),
                self.benchmark.label_d[0],
                color="#4C6E95",
                zorder=1e10,
            )
            plt.gca().text(
                origin[0] - offset_1 * (d1_len),
                origin[1] + (d2_len) / 2,
                self.benchmark.label_d[1],
                rotation=90,
                color="#4C6E95",
                zorder=1e12,
            )
        plt.gca().set_xlim(
            float(self.benchmark.aug_bounds[0][0]),
            float(self.benchmark.aug_bounds[0][1]),
        )
        plt.gca().set_ylim(
            float(self.benchmark.aug_bounds[1][0]),
            float(self.benchmark.aug_bounds[1][1]),
        )
