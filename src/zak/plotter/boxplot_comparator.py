"""Box plots that compare two analyses, by default ZAK-RBDO and ISORAK-RBDO, on one benchmark.

For every quantity of interest the boxes of the two analyses are drawn side by side; each box
summarises the final state of all the runs (seeds) of one analysis.
"""

from typing import Any

from attrs import field, define
from torch import Tensor
from polars import DataFrame
import torch
import polars as pl
import matplotlib
import matplotlib.patches as mpatches

from zak.benchmarks import register_benchmark

from .. import utils, plotter


def load_benchmark_data(analysis: str, benchmark_id: str, base_path: utils.Path) -> DataFrame:
    """Collect the final state of every run of an analysis on a benchmark.

    The runs are the sub-directories of "<base_path>/results/<analysis>/<benchmark_id>".

    Args:
        analysis: Name of the analysis, e.g. "zakrbdo".
        benchmark_id: Benchmark id.
        base_path: Directory that contains the "results" directory.

    Returns:
        One row per run: the last generation logged in its "results.json".
    """
    base_path = base_path / "results" / analysis / benchmark_id
    timestamps = [ts_dir.name for ts_dir in base_path.iterdir() if ts_dir.is_dir()]
    dfs = []
    for ts in timestamps:
        result_path = base_path / ts / "results.json"
        dfi = pl.read_ndjson(result_path)[-1]
        # The columns are sorted so that every DataFrame has the same column order.
        dfs.append(dfi.select(sorted(dfi.columns)))
    return pl.concat(dfs)


@define(auto_attribs=True)
class ModelComparison:
    """Side-by-side box plots of the results of two analyses on one benchmark.

    Attributes:
        data1: Final state of every run of the first analysis, one row per run.
        data2: Final state of every run of the second analysis, one row per run.
        benchmark_id: Benchmark id.
        output_path: Output directory; the figures are saved in its "<benchmark_id>"
            sub-directory, which is created if needed.
        method_labels: Labels of the two analyses.
        box_colors: Edge and face colours of the boxes of the two analyses.
        benchmark: The benchmark, instantiated from its id.
    """

    data1: DataFrame
    data2: DataFrame
    benchmark_id: str
    output_path: utils.Path
    method_labels: tuple[str, str] = (r"$\mathrm{ZAK}$", r"$\mathrm{ISORAK}$")
    box_colors: list[tuple[str, str]] = field(
        factory=lambda: [("k", "#C2D7EB"), ("k", "#DBE5D2")],
    )
    benchmark: Any = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Instantiate the benchmark and create the output directory."""
        self.benchmark = register_benchmark[self.benchmark_id]()
        self.output_path = self.output_path / self.benchmark_id
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _setup_plot(self, figsize="squareSmall") -> None:
        """Create a new figure with the grid shared by all the box plots.

        Args:
            figsize: Name of the figure format (see `plotter.fig_size`).
        """
        plotter.plt.figure(figsize=plotter.fig_size(figsize))
        plotter.plt.grid(visible=True, axis="y", linestyle=":", linewidth=0.15, alpha=0.2)
        plotter.plt.grid(visible=True, axis="x", linestyle=":", linewidth=0.25, alpha=0.3)
        plotter.plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())  # pyright: ignore
        plotter.plt.grid(
            visible=True,
            axis="y",
            which="minor",
            linestyle=":",
            linewidth=0.15,
            alpha=0.2,
        )

    def _style_boxplot(self, boxplot, all_data) -> None:
        """Apply the common style to a box plot, alternating the colours of the two analyses.

        Args:
            boxplot: The dictionary of artists returned by `matplotlib.pyplot.boxplot`.
            all_data: The plotted data sets, alternating between the two analyses.
        """
        for key in boxplot:
            plotter.plt.setp(boxplot[key], linewidth=0.5)

        plotter.plt.setp(
            boxplot["fliers"],
            markersize=2.5,
            marker="+",
            markeredgewidth=0.15,
            markeredgecolor="black",
        )

        plotter.plt.setp(boxplot["medians"], linewidth=0.75, color="#CC4C00")

        for i in range(len(all_data)):
            method_idx = i % 2

            # Every box owns two whiskers and two caps.
            whisker_start = i * 2
            whisker_end = whisker_start + 2

            plotter.plt.setp(
                boxplot["whiskers"][whisker_start:whisker_end],
                linewidth=0.15,
                color=self.box_colors[method_idx][0],
            )
            plotter.plt.setp(
                boxplot["caps"][whisker_start:whisker_end],
                linewidth=0.15,
                color=self.box_colors[method_idx][0],
            )

        for i, patch in enumerate(boxplot["boxes"]):
            method_idx = i % 2
            patch.set_edgecolor(self.box_colors[method_idx][0])
            patch.set_facecolor(self.box_colors[method_idx][1])
            patch.set_linewidth(0.15)

    def _add_methods_legend(self, threshold_line=None, threshold_label=None) -> None:
        """Add the legend of the two analyses and, optionally, of a threshold line.

        Args:
            threshold_line: Artist of the threshold line.
            threshold_label: Label of the threshold line.
        """
        method1_patch = mpatches.Patch(color=self.box_colors[0][1], label=self.method_labels[0])
        method2_patch = mpatches.Patch(color=self.box_colors[1][1], label=self.method_labels[1])

        if threshold_line is not None and threshold_label is not None:
            plotter.plt.legend(
                [method1_patch, method2_patch, threshold_line],
                [self.method_labels[0], self.method_labels[1], threshold_label],
                loc="best",
                fontsize=6,
            )
        else:
            plotter.plt.legend(
                handles=[method1_patch, method2_patch],
                loc="best",
                fontsize=6,
            )

    def _save_plot(self, filename) -> None:
        """Save the current figure as "<filename>.pdf" in the output directory."""
        plotter.set_square_ratio(plotter.plt.gca())
        ax = plotter.plt.gca()
        ax.set_position((0.3, 0.1, 0.6, 0.8))
        plotter.save(plt=plotter.plt, path=self.output_path, filename=filename, pdf_dir=False)

    def _prepare_grouped_boxplot_data(
        self,
        values1,
        values2,
    ) -> tuple[list[Any], list[Any], list[Any], float, int]:
        """Arrange the data of the two analyses in groups of two boxes, one group per item.

        Args:
            values1: Values of the first analysis, of shape (n_runs, n_items).
            values2: Values of the second analysis, of shape (n_runs, n_items).

        Returns:
            The data sets in plotting order, the positions of the boxes, the positions of the
            group ticks, the width of the boxes and the number of items.
        """
        n_items = values1.shape[-1]

        all_data = []
        positions = []
        tick_positions = []

        box_width = 0.125

        for i in range(n_items):
            all_data.append(values1[:, i])
            all_data.append(values2[:, i])

            # The two boxes of a group sit on either side of the integer position i + 1.
            group_center = i + 1
            positions.append(group_center - 1.5 * box_width / 2)
            positions.append(group_center + 1.5 * box_width / 2)

            tick_positions.append(group_center)

        return all_data, positions, tick_positions, box_width, n_items

    def plot_cost(self) -> tuple[Tensor, Tensor]:
        """Save the box plot that compares the cost of the optimal designs.

        Returns:
            The median cost of each analysis.
        """
        self._setup_plot()

        if self.benchmark.nominal_tolerance_design:
            values1 = self.benchmark.cost.fn(
                torch.tensor(self.data1["pop_best"])[int(self.benchmark.dim / 2) :],
            )
        else:
            values1 = self.benchmark.cost.fn(torch.tensor(self.data1["pop_best"]))

        if self.benchmark.nominal_tolerance_design:
            values2 = self.benchmark.cost.fn(
                torch.tensor(self.data2["pop_best"])[int(self.benchmark.dim / 2) :],
            )
        else:
            values2 = self.benchmark.cost.fn(torch.tensor(self.data2["pop_best"]))

        boxplot = plotter.plt.boxplot(
            [values1, values2],
            tick_labels=self.method_labels,
            patch_artist=True,
        )

        plotter.plt.ylabel(r"$\bm{\mathfrak{c}}(\bm{d}^{*})$")

        self._style_boxplot(boxplot, [values1, values2])

        self._save_plot("box_plot_cost_comparison")

        return values1.median(), values2.median()

    def plot_design(self) -> tuple[Tensor, Tensor]:
        """Save the box plot that compares the optimal design variables.

        Returns:
            The median design of each analysis.
        """
        self._setup_plot()

        values1 = torch.tensor(self.data1["pop_best"])
        values2 = torch.tensor(self.data2["pop_best"])

        n_vars = values1.shape[-1]

        tick_labels = [self.benchmark.label_d[i] for i in range(n_vars)]

        all_data, positions, tick_positions, box_width, _ = self._prepare_grouped_boxplot_data(
            values1,
            values2,
        )

        boxplot = plotter.plt.boxplot(
            all_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
        )

        plotter.plt.xticks(tick_positions, tick_labels)
        plotter.plt.ylabel(r"$\bm{d}^{*}$")

        self._style_boxplot(boxplot, all_data)

        self._add_methods_legend()

        self._save_plot("box_plot_design_comparison")

        return values1.median(dim=0).values, values2.median(dim=0).values

    def plot_pf(
        self,
        pf_thr_list: list[float],
        *,
        hat: bool = True,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Save the box plot that compares the failure probabilities at the optimal designs.

        Args:
            pf_thr_list: Thresholds on the failure probabilities; the largest one is drawn.
            hat: Whether the labels carry a hat, for probabilities estimated with the surrogates.

        Returns:
            The median failure probabilities of each analysis, and the threshold drawn.
        """
        self._setup_plot()

        values1 = torch.tensor(self.data1["Pf"])
        values2 = torch.tensor(self.data2["Pf"])

        pf_thr = torch.tensor(pf_thr_list).max()

        if hat:
            tick_labels = [
                rf"$\widehat{{\mathcal{{P}}}}_{{f_{{{i + 1}}}}}(\bm{{d}}^{{*}})$"
                for i in range(values1.shape[-1])
            ]
            y_label = r"$\widehat{{\mathcal{{P}}}}_{{f_{{k}}}}(\bm{{d}}^{{*}})$"
        else:
            tick_labels = [
                rf"$\mathcal{{P}}_{{f_{{{i + 1}}}}}(\bm{{d}}^{{*}})$"
                for i in range(values1.shape[-1])
            ]
            y_label = r"$\mathcal{{P}}_{{f_{{k}}}}(\bm{{d}}^{{*}})$"

        all_data, positions, tick_positions, box_width, n_constraints = (
            self._prepare_grouped_boxplot_data(values1, values2)
        )

        boxplot = plotter.plt.boxplot(
            all_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
        )

        plotter.plt.xticks(tick_positions, tick_labels)
        plotter.plt.ylabel(y_label)

        self._style_boxplot(boxplot, all_data)

        hl = plotter.plt.hlines(pf_thr, 0.5, n_constraints + 0.5, **plotter.plot_style["Pf_thr"])
        plotter.plt.xlim(0.5, n_constraints + 0.5)

        self._add_methods_legend(hl, r"$\overline{\mathcal{P}}_{f_{k}}$")

        plotter.set_scientific_notation(ax=plotter.plt.gca(), axis="y")
        self._save_plot("box_plot_pf_comparison")

        return (values1.median(dim=0).values, values2.median(dim=0).values), pf_thr

    def plot_beta(
        self,
        pf_thr_list: list[float],
        *,
        hat: bool = True,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Save the box plot that compares the reliability indices at the optimal designs.

        Args:
            pf_thr_list: Thresholds on the failure probabilities; the largest one is converted
                to a reliability index and drawn.
            hat: Whether the labels carry a hat, for indices estimated with the surrogates.

        Returns:
            The median reliability indices of each analysis, and the threshold drawn.
        """
        self._setup_plot()

        values1 = torch.tensor(self.data1["Beta"])
        values2 = torch.tensor(self.data2["Beta"])

        pf_thr = torch.tensor(pf_thr_list).max()
        beta_thr = -torch.distributions.Normal(0, 1).icdf(pf_thr)

        if hat:
            tick_labels = [
                rf"$\widehat{{\beta}}_{{{{{i + 1}}}}}(\bm{{d}}^{{*}})$"
                for i in range(values1.shape[-1])
            ]
            y_label = r"$\widehat{{\beta}}_{{{{k}}}}(\bm{{d}}^{{*}})$"
        else:
            tick_labels = [
                rf"$\beta_{{{{{i + 1}}}}}(\bm{{d}}^{{*}})$" for i in range(values1.shape[-1])
            ]
            y_label = r"$\beta_{{{{k}}}}(\bm{{d}}^{{*}})$"

        all_data, positions, tick_positions, box_width, n_constraints = (
            self._prepare_grouped_boxplot_data(values1, values2)
        )

        boxplot = plotter.plt.boxplot(
            all_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
        )

        plotter.plt.xticks(tick_positions, tick_labels)
        plotter.plt.ylabel(y_label)

        self._style_boxplot(boxplot, all_data)

        hl = plotter.plt.hlines(beta_thr, 0.5, n_constraints + 0.5, **plotter.plot_style["Pf_thr"])
        plotter.plt.xlim(0.5, n_constraints + 0.5)

        self._add_methods_legend(hl, r"$\overline{\beta}_{{k}}$")

        self._save_plot("box_plot_beta_comparison")

        return (values1.median(dim=0).values, values2.median(dim=0).values), beta_thr

    def plot_g_calls(self) -> tuple[Tensor, Tensor]:
        """Save the box plot that compares the number of calls to the limit-state functions.

        Returns:
            The median number of calls of each analysis, per limit state.
        """
        self._setup_plot()

        values1 = torch.tensor(self.data1["g_calls"])
        values2 = torch.tensor(self.data2["g_calls"])

        tick_labels = [rf"$N_{{{{\mathfrak{{g}}}}_{{{i + 1}}}}}$" for i in range(values1.shape[-1])]

        all_data, positions, tick_positions, box_width, _ = self._prepare_grouped_boxplot_data(
            values1,
            values2,
        )

        boxplot = plotter.plt.boxplot(
            all_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
        )

        plotter.plt.xticks(tick_positions, tick_labels)
        plotter.plt.ylabel(r"Number of $\mathfrak{{g}}$-calls")

        self._style_boxplot(boxplot, all_data)

        self._add_methods_legend()

        self._save_plot("box_plot_gk_calls_comparison")

        return values1.median(dim=0).values, values2.median(dim=0).values

    def plot_all(
        self,
        pf_thr: list[float],
        *,
        plot_g_calls: bool = True,
        show_plots: bool = True,
    ) -> None:
        """Save all the comparison box plots.

        Args:
            pf_thr: Thresholds on the failure probabilities.
            plot_g_calls: Whether to plot the number of calls to the limit-state functions.
            show_plots: Whether to display the figures once they are saved.
        """
        self.plot_cost()
        self.plot_design()
        self.plot_pf(pf_thr)
        self.plot_beta(pf_thr)
        if plot_g_calls:
            self.plot_g_calls()
        if show_plots:
            plotter.plt.show()
