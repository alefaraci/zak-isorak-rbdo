"""Box plots and median-results tables that summarise the runs of one analysis on one benchmark.

Each row of the input DataFrame is the final state of one run (one seed). A box plot is saved for
every quantity of interest and the medians over the runs are printed in a table and saved to
"median_results.log".
"""

from torch import Tensor
from polars import DataFrame
import torch
import matplotlib

from zak import utils, plotter
from zak.benchmarks import register_benchmark

# The recording console is used so that `utils.log_console` can save the tables to a file.
console = utils.console_direct


def _box_plot_alpha(dfs: DataFrame, path: utils.Path) -> Tensor:
    """Save the box plot of the final isocost level alpha (ISORAK-RBDO).

    Args:
        dfs: Final state of every run, with the column "alpha".
        path: Output directory; the figure is saved as "box_plot_alpha.pdf".

    Returns:
        The median level over the runs.
    """
    values = torch.tensor(dfs["alpha"])
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
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
    boxplot = plotter.plt.boxplot(values, tick_labels=["Iso-cost level"], patch_artist=True)
    plotter.plt.ylabel(r"$\mathrm{\alpha}$")
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
    plotter.plt.setp(boxplot["whiskers"], linewidth=0.15, color="k")
    plotter.plt.setp(boxplot["caps"], linewidth=0.15, color="k")
    for patch in boxplot["boxes"]:
        patch.set_edgecolor("k")
        patch.set_facecolor("#C2D7EB")
        patch.set_linewidth(0.15)
    plotter.set_square_ratio(plotter.plt.gca())
    plotter.plt.tight_layout()
    plotter.save(plt=plotter.plt, path=path, filename="box_plot_alpha", pdf_dir=False)
    plotter.plt.close()
    return values.median()


def _box_plot_cost(dfs: DataFrame, bench_id: str, path: utils.Path) -> Tensor:
    """Save the box plot of the cost of the optimal designs.

    Args:
        dfs: Final state of every run, with the column "pop_best" (the optimal design).
        bench_id: Benchmark id, used to evaluate the cost.
        path: Output directory; the figure is saved as "box_plot_cost.pdf".

    Returns:
        The median cost over the runs.
    """
    benchmark = register_benchmark[bench_id]()
    # For nominal-tolerance designs the cost depends on the tolerances only, which are stored in
    # the second half of the design vector.
    if benchmark.nominal_tolerance_design:
        values = benchmark.cost.fn(torch.tensor(dfs["pop_best"])[:, int(benchmark.dim / 2) :])
    else:
        values = benchmark.cost.fn(torch.tensor(dfs["pop_best"]))
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
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
    boxplot = plotter.plt.boxplot(values, tick_labels=[r"$\bm{d}^{*}$"], patch_artist=True)
    plotter.plt.ylabel(r"$\bm{\mathfrak{c}}(\bm{d}^{*})$")
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
    plotter.plt.setp(boxplot["whiskers"], linewidth=0.15, color="k")
    plotter.plt.setp(boxplot["caps"], linewidth=0.15, color="k")
    for patch in boxplot["boxes"]:
        patch.set_edgecolor("k")
        patch.set_facecolor("#C2D7EB")
        patch.set_linewidth(0.15)
    plotter.set_square_ratio(plotter.plt.gca())
    plotter.plt.tight_layout()
    plotter.save(plt=plotter.plt, path=path, filename="box_plot_cost", pdf_dir=False)
    plotter.plt.close()
    return values.median()


def _box_plot_design(dfs: DataFrame, path: utils.Path, bench_id: str) -> Tensor:
    """Save the box plot of the optimal design variables.

    Args:
        dfs: Final state of every run, with the column "pop_best" (the optimal design).
        path: Output directory; the figure is saved as "box_plot_design.pdf".
        bench_id: Benchmark id, used for the labels of the design variables.

    Returns:
        The median of every design variable over the runs.
    """
    benchmark = register_benchmark[bench_id]()
    values = torch.tensor(dfs["pop_best"])
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
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
    boxplot = plotter.plt.boxplot(
        values,
        tick_labels=benchmark.label_d,
        patch_artist=True,
    )
    plotter.plt.ylabel(r"$\bm{d}^{*}$")
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
    plotter.plt.setp(boxplot["whiskers"], linewidth=0.15, color="k")
    plotter.plt.setp(boxplot["caps"], linewidth=0.15, color="k")
    for patch in boxplot["boxes"]:
        patch.set_edgecolor("k")
        patch.set_facecolor("#C2D7EB")
        patch.set_linewidth(0.15)
    plotter.set_square_ratio(plotter.plt.gca())
    plotter.plt.tight_layout()
    plotter.save(plt=plotter.plt, path=path, filename="box_plot_design", pdf_dir=False)
    plotter.plt.close()
    return values.median(dim=0).values


def _box_plot_pf(
    dfs: DataFrame,
    Pf_thr: list[float],
    path: utils.Path,
    *,
    hat: bool = False,
) -> tuple[Tensor, Tensor]:
    """Save the box plot of the failure probabilities at the optimal designs.

    Args:
        dfs: Final state of every run, with the column "Pf" (one probability per limit state).
        Pf_thr: Thresholds on the failure probabilities.
        path: Output directory; the figure is saved as "box_plot_pf.pdf".
        hat: Whether the labels carry a hat, for probabilities estimated with the surrogates.

    Returns:
        The median failure probability of every limit state over the runs, and the threshold
        drawn on the figure.
    """
    values = torch.tensor(dfs["Pf"])
    # A single threshold line is drawn: the largest of the thresholds.
    pf_thr = torch.tensor(Pf_thr).max()
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
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
    if hat:
        tick_labels = [
            rf"$\widehat{{\mathcal{{P}}}}_{{f_{{{i + 1}}}}}(\bm{{d}}^{{*}})$"
            for i in range(values.shape[-1])
        ]
        y_label = r"$\widehat{{\mathcal{{P}}}}_{{f_{{k}}}}(\bm{{d}}^{{*}})$"
    else:
        tick_labels = [
            rf"$\mathcal{{P}}_{{f_{{{i + 1}}}}}(\bm{{d}}^{{*}})$" for i in range(values.shape[-1])
        ]
        y_label = r"$\mathcal{{P}}_{{f_{{k}}}}(\bm{{d}}^{{*}})$"
    boxplot = plotter.plt.boxplot(values, tick_labels=tick_labels, patch_artist=True)
    plotter.plt.ylabel(y_label)
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
    plotter.plt.setp(boxplot["whiskers"], linewidth=0.15, color="k")
    plotter.plt.setp(boxplot["caps"], linewidth=0.15, color="k")
    for patch in boxplot["boxes"]:
        patch.set_edgecolor("k")
        patch.set_facecolor("#C2D7EB")
        patch.set_linewidth(0.15)
    _ = plotter.plt.hlines(pf_thr, 0.5, values.shape[-1] + 0.5, **plotter.plot_style["Pf_thr"])
    plotter.set_square_ratio(plotter.plt.gca())
    plotter.plt.tight_layout()
    plotter.save(plt=plotter.plt, path=path, filename="box_plot_pf", pdf_dir=False)
    plotter.plt.close()
    return values.median(dim=0).values, pf_thr


def _box_plot_beta(
    dfs: DataFrame,
    Pf_thr: list[float],
    path: utils.Path,
    *,
    hat: bool = False,
) -> tuple[Tensor, Tensor]:
    """Save the box plot of the reliability indices at the optimal designs.

    Args:
        dfs: Final state of every run, with the column "Beta" (one index per limit state).
        Pf_thr: Thresholds on the failure probabilities.
        path: Output directory; the figure is saved as "box_plot_beta.pdf".
        hat: Whether the labels carry a hat, for indices estimated with the surrogates.

    Returns:
        The median reliability index of every limit state over the runs, and the threshold drawn
        on the figure.
    """
    values = torch.tensor(dfs["Beta"])
    pf_thr = torch.tensor(Pf_thr).max()
    # Reliability index that corresponds to the threshold probability: beta = -Phi^-1(Pf).
    beta_thr = -torch.distributions.Normal(0, 1).icdf(pf_thr)
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
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
    if hat:
        tick_labels = [
            rf"$\widehat{{\beta}}_{{{{{i + 1}}}}}(\bm{{d}}^{{*}})$" for i in range(values.shape[-1])
        ]
        y_label = r"$\widehat{{\beta}}_{{{{k}}}}(\bm{{d}}^{{*}})$"
    else:
        tick_labels = [rf"$\beta_{{{{{i + 1}}}}}(\bm{{d}}^{{*}})$" for i in range(values.shape[-1])]
        y_label = r"$\beta_{{{{k}}}}(\bm{{d}}^{{*}})$"
    boxplot = plotter.plt.boxplot(
        values,
        tick_labels=tick_labels,
        patch_artist=True,
    )
    plotter.plt.ylabel(y_label)
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
    plotter.plt.setp(boxplot["whiskers"], linewidth=0.15, color="k")
    plotter.plt.setp(boxplot["caps"], linewidth=0.15, color="k")
    for patch in boxplot["boxes"]:
        patch.set_edgecolor("k")
        patch.set_facecolor("#C2D7EB")
        patch.set_linewidth(0.15)
    _ = plotter.plt.hlines(beta_thr, 0.5, values.shape[-1] + 0.5, **plotter.plot_style["Pf_thr"])
    plotter.set_square_ratio(plotter.plt.gca())
    plotter.plt.tight_layout()
    plotter.save(plt=plotter.plt, path=path, filename="box_plot_beta", pdf_dir=False)
    plotter.plt.close()
    return values.median(dim=0).values, beta_thr


def _box_plot_g_calls(dfs: DataFrame, path: utils.Path) -> Tensor:
    """Save the box plot of the number of calls to the limit-state functions.

    Args:
        dfs: Final state of every run, with the column "g_calls" (one count per limit state).
        path: Output directory; the figure is saved as "box_plot_gk_calls.pdf".

    Returns:
        The median number of calls of every limit state over the runs.
    """
    values = torch.tensor(dfs["g_calls"])
    plotter.plt.figure(figsize=plotter.fig_size("squareSmall"))
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
    boxplot = plotter.plt.boxplot(
        values,
        tick_labels=[rf"$N_{{{{\mathfrak{{g}}}}_{{{i + 1}}}}}$" for i in range(values.shape[-1])],
        patch_artist=True,
    )
    plotter.plt.ylabel(r"Number of $\mathfrak{{g}}$-calls")
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
    plotter.plt.setp(boxplot["whiskers"], linewidth=0.15, color="k")
    plotter.plt.setp(boxplot["caps"], linewidth=0.15, color="k")
    for patch in boxplot["boxes"]:
        patch.set_edgecolor("k")
        patch.set_facecolor("#C2D7EB")
        patch.set_linewidth(0.15)
    plotter.set_square_ratio(plotter.plt.gca())
    plotter.plt.tight_layout()
    plotter.save(plt=plotter.plt, path=path, filename="box_plot_gk_calls", pdf_dir=False)
    plotter.plt.close()
    return values.median(dim=0).values


def _median_results_ddo(cost, design, bench_id, path) -> None:
    """Print the table of the median results of the DDO runs and save the console log.

    Args:
        cost: Median cost.
        design: Median design.
        bench_id: Benchmark id, shown in the title.
        path: Directory where "median_results.log" is written.
    """
    tab = utils.table(
        ["cost", "design"],
        title=f"Median Results DDO\n{bench_id}\n",
    )
    design_str = ", ".join([f"{di.item():.4f}" for di in design])
    formatted_row = [f"{cost.item():.4f}", design_str]
    tab.add_row(*formatted_row)
    console.log(tab)
    console.log("\n")
    utils.log_console(path, "median_results.log")


def _median_results_rbdo(cost, design, bench_id, pf, beta, pf_thr, beta_thr, path) -> None:
    """Print the tables of the median results of the RBDO runs and save the console log.

    The second table checks the medians against the thresholds.

    Args:
        cost: Median cost.
        design: Median design.
        bench_id: Benchmark id, shown in the title.
        pf: Median failure probability of every limit state.
        beta: Median reliability index of every limit state.
        pf_thr: Threshold on the failure probabilities.
        beta_thr: Corresponding threshold on the reliability indices.
        path: Directory where "median_results.log" is written.
    """
    tab = utils.table(
        ["cost", "design", "Beta_k", "Pf_k"],
        title=f"Median Results RBDO ~ Benchmark: {bench_id}",
    )
    design_str = ", ".join([f"{di.item():.4f}" for di in design])
    beta_str = ", ".join([f"{betak.item():.4f}" for betak in beta])
    pf_str = ", ".join([f"{pfk.item():.4e}" for pfk in pf])

    formatted_row = [
        f"{cost.item():.4f}",
        design_str,
        beta_str,
        pf_str,
    ]
    tab.add_row(*formatted_row)
    console.log(tab)

    tab = utils.table(
        ["Beta_k min", "Beta_thr", "Beta", "Pf_k max", "Pfk_thr", "Pf"],
        title=None,
    )

    beta_check = bool(beta.min() >= beta_thr)
    pf_check = bool(pf.max() <= pf_thr)
    beta_check_str = f"[green]{beta_check}[/green]" if beta_check else f"[red]{beta_check}[/red]"
    pf_check_str = f"[green]{pf_check}[/green]" if pf_check else f"[red]{pf_check}[/red]"

    formatted_row = [
        f"{beta.min().item():.4f}",
        f"{beta_thr.item():.4f}",
        beta_check_str,
        f"{pf.max().item():.4e}",
        f"{pf_thr.item():.4e}",
        pf_check_str,
    ]
    tab.add_row(*formatted_row)
    console.log(tab)
    console.log("\n")
    utils.log_console(path, "median_results.log")


def _median_results_zakrbdo(
    cost,
    design,
    bench_id,
    pf,
    beta,
    pf_thr,
    beta_thr,
    g_calls,
    path,
) -> None:
    """Print the tables of the median results of the ZAK-RBDO runs and save the console log.

    The second table checks the medians against the thresholds.

    Args:
        cost: Median cost.
        design: Median design.
        bench_id: Benchmark id, shown in the title.
        pf: Median failure probability of every limit state.
        beta: Median reliability index of every limit state.
        pf_thr: Threshold on the failure probabilities.
        beta_thr: Corresponding threshold on the reliability indices.
        g_calls: Median number of calls of every limit-state function.
        path: Directory where "median_results.log" is written.
    """
    tab = utils.table(
        ["cost", "design", "Beta_k", "Pf_k", "gk-calls"],
        title=f"Median Results ZAKRBDO ~ Benchmark: {bench_id}",
    )
    design_str = ", ".join([f"{di.item():.4f}" for di in design])
    beta_str = ", ".join([f"{betak.item():.4f}" for betak in beta])
    pf_str = ", ".join([f"{pfk.item():.4e}" for pfk in pf])
    g_calls_str = ", ".join([f"{int(nk.item())}" for nk in g_calls])

    formatted_row = [
        f"{cost.item():.4f}",
        design_str,
        beta_str,
        pf_str,
        g_calls_str,
    ]
    tab.add_row(*formatted_row)
    console.log(tab)

    tab = utils.table(
        ["Beta_k min", "Beta_thr", "Beta", "Pf_k max", "Pfk_thr", "Pf"],
        title=None,
    )

    beta_check = bool(beta.min() >= beta_thr)
    pf_check = bool(pf.max() <= pf_thr)
    beta_check_str = f"[green]{beta_check}[/green]" if beta_check else f"[red]{beta_check}[/red]"
    pf_check_str = f"[green]{pf_check}[/green]" if pf_check else f"[red]{pf_check}[/red]"

    formatted_row = [
        f"{beta.min().item():.4f}",
        f"{beta_thr.item():.4f}",
        beta_check_str,
        f"{pf.max().item():.4e}",
        f"{pf_thr.item():.4e}",
        pf_check_str,
    ]
    tab.add_row(*formatted_row)
    console.log(tab)
    console.log("\n")
    utils.log_console(path, "median_results.log")


def _median_results_isorakrbdo(
    alpha,
    cost,
    design,
    bench_id,
    pf,
    beta,
    pf_thr,
    beta_thr,
    g_calls,
    path,
) -> None:
    """Print the tables of the median results of the ISORAK-RBDO runs and save the console log.

    The second table checks the medians against the thresholds.

    Args:
        alpha: Median level of the final isocost region.
        cost: Median cost.
        design: Median design.
        bench_id: Benchmark id, shown in the title.
        pf: Median failure probability of every limit state.
        beta: Median reliability index of every limit state.
        pf_thr: Threshold on the failure probabilities.
        beta_thr: Corresponding threshold on the reliability indices.
        g_calls: Median number of calls of every limit-state function.
        path: Directory where "median_results.log" is written.
    """
    tab = utils.table(
        ["alpha", "cost", "design", "Beta_k", "Pf_k", "gk-calls"],
        title=f"Median Results ISORAKRBDO ~ Benchmark: {bench_id}",
    )
    design_str = ", ".join([f"{di.item():.4f}" for di in design])
    beta_str = ", ".join([f"{betak.item():.4f}" for betak in beta])
    pf_str = ", ".join([f"{pfk.item():.4e}" for pfk in pf])
    g_calls_str = ", ".join([f"{int(nk.item())}" for nk in g_calls])
    formatted_row = [
        f"{alpha.item():.4f}",
        f"{cost.item():.4f}",
        design_str,
        beta_str,
        pf_str,
        g_calls_str,
    ]
    tab.add_row(*formatted_row)
    console.log(tab)

    tab = utils.table(
        ["Beta_k min", "Beta_thr", "Beta", "Pf_k max", "Pfk_thr", "Pf"],
        title=None,
    )

    beta_check = bool(beta.min() >= beta_thr)
    pf_check = bool(pf.max() <= pf_thr)
    beta_check_str = f"[green]{beta_check}[/green]" if beta_check else f"[red]{beta_check}[/red]"
    pf_check_str = f"[green]{pf_check}[/green]" if pf_check else f"[red]{pf_check}[/red]"

    formatted_row = [
        f"{beta.min().item():.4f}",
        f"{beta_thr.item():.4f}",
        beta_check_str,
        f"{pf.max().item():.4e}",
        f"{pf_thr.item():.4e}",
        pf_check_str,
    ]
    tab.add_row(*formatted_row)
    console.log(tab)
    console.log("\n")
    utils.log_console(path, "median_results.log")


def box_plot(analysis: str, dfs: DataFrame, bench_id: str, path: utils.Path) -> None:
    """Save the box plots of an analysis over its runs and print the median results.

    Args:
        analysis: "ddo", "rbdo", "zakrbdo" or "isorakrbdo".
        dfs: Final state of every run, one row per run.
        bench_id: Benchmark id.
        path: Output directory, expected to be "<root>/results/<analysis>/<bench_id>": the
            benchmark settings are read from "<root>/<bench_id>.yaml".

    Raises:
        ValueError: If the analysis is unknown.
    """
    config = utils.load_yaml(path=path.parent.parent.parent, filename=f"{bench_id}.yaml")
    Pf_thr = config["RBDO"]["rbdo"]["Pf_thr"]
    match analysis:
        case "ddo":
            dfs = dfs.select(["pop_best"])
            cost = _box_plot_cost(dfs=dfs, bench_id=bench_id, path=path)
            design = _box_plot_design(dfs=dfs, path=path, bench_id=bench_id)
            _median_results_ddo(cost, design, bench_id, path)
        case "rbdo":
            dfs = dfs.select(["Beta", "Pf", "pop_best"])
            cost = _box_plot_cost(dfs=dfs, bench_id=bench_id, path=path)
            design = _box_plot_design(dfs=dfs, path=path, bench_id=bench_id)
            pf, pf_thr = _box_plot_pf(dfs=dfs, Pf_thr=Pf_thr, path=path)
            beta, beta_thr = _box_plot_beta(dfs=dfs, Pf_thr=Pf_thr, path=path)
            _median_results_rbdo(cost, design, bench_id, pf, beta, pf_thr, beta_thr, path)
        case "zakrbdo":
            dfs = dfs.select(["Beta", "Pf", "pop_best", "g_calls"])
            g_calls = _box_plot_g_calls(dfs=dfs, path=path)
            cost = _box_plot_cost(dfs=dfs, bench_id=bench_id, path=path)
            design = _box_plot_design(dfs=dfs, path=path, bench_id=bench_id)
            pf, pf_thr = _box_plot_pf(dfs=dfs, Pf_thr=Pf_thr, path=path, hat=True)
            beta, beta_thr = _box_plot_beta(dfs=dfs, Pf_thr=Pf_thr, path=path, hat=True)
            _median_results_zakrbdo(
                cost,
                design,
                bench_id,
                pf,
                beta,
                pf_thr,
                beta_thr,
                g_calls,
                path,
            )
        case "isorakrbdo":
            dfs = dfs.select(["Beta", "Pf", "pop_best", "g_calls", "alpha"])
            g_calls = _box_plot_g_calls(dfs=dfs, path=path)
            cost = _box_plot_cost(dfs=dfs, bench_id=bench_id, path=path)
            alpha = _box_plot_alpha(dfs=dfs, path=path)
            design = _box_plot_design(dfs=dfs, path=path, bench_id=bench_id)
            pf, pf_thr = _box_plot_pf(dfs=dfs, Pf_thr=Pf_thr, path=path, hat=True)
            beta, beta_thr = _box_plot_beta(dfs=dfs, Pf_thr=Pf_thr, path=path, hat=True)
            _median_results_isorakrbdo(
                alpha,
                cost,
                design,
                bench_id,
                pf,
                beta,
                pf_thr,
                beta_thr,
                g_calls,
                path,
            )
        case _:
            error_msg = f"Analysis {analysis} not found"
            raise ValueError(error_msg)
