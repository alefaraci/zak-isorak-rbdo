"""Post-processing of a finished run: result figures, evolution frames and videos.

:class:`PostProcessing` turns the per-generation results of one run into

* evolution curves (cost, design variables, fitness, failure probabilities, reliability indices,
  maximum ESC error and number of limit-state calls);
* the optimal design drawn over the 2D contours of the benchmark, alone, with the history of the
  best design, and with its Monte Carlo and quasi-Monte Carlo sample clouds;
* per-generation frames of the population, assembled into videos by ``run_animator.sh``.

The evolution curves exist for every benchmark; the 2D figures only for the benchmarks listed in
:data:`plotting`.
"""

import subprocess

from attrs import field, define
from torch import Tensor
from polars import DataFrame
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import KernelDensity
import numpy as np
import torch

from .. import utils, plotter
from ..utils import Table
from ..loggers import log_soft_constr
from ..plotter import register_zoom
from ..problem import OptimizationProblem
from .animations import (
    animation_evolution_g_space_frame,
    animation_evolution_tol_space_frame,
    animation_evolution_cost_space_frame,
    animation_evolution_tol_g_space_frame,
    animation_evolution_nom_tol_adim_space_frame,
    animation_evolution_tol_mean_shift_adim_space_frame,
)
from ..benchmarks import BenchmarkBase
from ..plotter.benchmark_2d import BaseModel2D

# Benchmark ids for which each 2D figure is produced. A figure is skipped for the benchmarks that
# are not in its set, because it has no 2D representation there or would not be informative.
#
# - "optimal_design": optimal design, and history of the best design, over the benchmark contours
#   (two design variables, or two nominal values).
# - "optimal_design_mcs": Monte Carlo and quasi-Monte Carlo clouds at the optimal design, in the
#   augmented space (two random variables).
# - "animation_g_space": frames of the population in the design space.
# - "animation_g_aug_space": same frames with the sample clouds of the individuals, for the
#   benchmarks whose random variables are the perturbed design variables.
# - "animation_cost_space": frames of the population over the cost contours.
# - "animation_tol_space": frames of the population in the plane of the two tolerances and frames
#   of the tolerance intervals around the nominal values. `_run_all_base_plots` also uses this set
#   to trigger the history of the best design in the tolerance plane.
# - "animation_nom_tol_adim_space": frames in the dimensionless nominal/tolerance plane and in the
#   dimensionless mean-shift plane, which can show any number of toleranced dimensions.
# - "optimal_design_history_tol_space": history of the best design in the plane of the two
#   tolerances.
plotting = {
    "optimal_design": {
        "highly_nonlinear",
        "three_nonlinear",
        "four_branches",
        "column_under_compression",
        "truss_bridge",
        "cantilever_2d",
        "cantilever_2d_mean_shift",
    },
    "optimal_design_mcs": {
        "highly_nonlinear",
        "three_nonlinear",
        "four_branches",
        "linear_stackup",
        "linear_stackup_mean_shift",
        "cantilever_2d",
        "cantilever_2d_mean_shift",
    },
    "animation_g_space": {
        "highly_nonlinear",
        "three_nonlinear",
        "four_branches",
        "column_under_compression",
        "truss_bridge",
    },
    "animation_g_aug_space": {
        "highly_nonlinear",
        "three_nonlinear",
        "four_branches",
    },
    "animation_cost_space": {
        "column_under_compression",
        "truss_bridge",
    },
    "animation_tol_space": {
        "linear_stackup",
        "linear_stackup_mean_shift",
        "cantilever_2d",
        "cantilever_2d_mean_shift",
    },
    "animation_nom_tol_adim_space": {
        "cantilever_2d",
        "cantilever_2d_mean_shift",
        "cantilever_4d",
        "cantilever_4d_mean_shift",
    },
    "optimal_design_history_tol_space": {
        "linear_stackup",
        "linear_stackup_mean_shift",
        "cantilever_2d",
        "cantilever_2d_mean_shift",
    },
}


@define(
    slots=False,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class PostProcessing:
    """Figures, evolution frames and videos of one finished run.

    Attributes:
        path: Run directory. Figures are saved in ``<path>/pdf`` and frames in
            ``<path>/evolution``.
        problem: Optimization problem of the run.
        results: Results of the run, one row per generation.
        benchmark_2d: 2D view of the benchmark with its true limit states.
        surrogate_2d: 2D view with the final Kriging surrogates (adaptive-Kriging analyses only).
        design_opt: Best design of the last generation.
        evo_cost: Cost of the best design of each generation.
        evo_design: Best design of each generation.
        evo_prob_fail: Failure probabilities of the best design of each generation (RBDO only).
        evo_beta: Reliability indices of the best design of each generation (RBDO only).
        evo_eps_max: Maximum ESC error of each generation (adaptive-Kriging analyses only).
        benchmark_zoom: Axis limits of the best-design history figure, by analysis and benchmark.
    """

    path: utils.Path
    problem: OptimizationProblem
    results: DataFrame
    benchmark_2d: BaseModel2D = field(init=False)
    surrogate_2d: BaseModel2D = field(init=False)
    design_opt: Tensor = field(init=False)
    evo_cost: Tensor = field(init=False)
    evo_design: Tensor = field(init=False)
    evo_prob_fail: Tensor = field(init=False)
    evo_beta: Tensor = field(init=False)
    evo_eps_max: Tensor = field(init=False)
    benchmark_zoom: dict[str, dict[str, dict[str, tuple[float, float]]]] = field(
        default=register_zoom,
    )

    def __attrs_post_init__(self) -> None:
        """Build the 2D views and extract the evolution histories from the results."""
        self.benchmark_2d = BaseModel2D(benchmark=self.benchmark)
        if "ak" in self.problem.analysis:
            self.surrogate_2d = BaseModel2D(
                benchmark=self.benchmark,
                hard=self.problem.metamodels.hard,
            )
        self.design_opt = torch.tensor(self.results["pop_best"][-1])
        self.evo_design = torch.tensor(self.results["pop_best"])
        if self.benchmark.nominal_tolerance_design:
            # The cost of a [nominal values, tolerances] design depends on its tolerances only.
            tolerance = torch.tensor(self.results["pop_best"])[:, int(self.benchmark.dim / 2) :]
            self.evo_cost = self.benchmark.evaluate_cost(tolerance)
        else:
            self.evo_cost = self.benchmark.evaluate_cost(self.evo_design)
        self.evo_cost = self.benchmark.evaluate_cost(self.evo_design)
        if "rbdo" in self.problem.analysis:
            self.evo_prob_fail = torch.tensor(self.results["Pf"])
            self.evo_beta = torch.tensor(self.results["Beta"])
        if "ak" in self.problem.analysis:
            self.evo_eps_max = torch.tensor(self.results["eps_max"])

    @property
    def benchmark(self) -> BenchmarkBase:
        """Return the benchmark."""
        return self.problem.benchmark

    @property
    def n_gen(self) -> int:
        """Return the number of generations."""
        return len(self.results)

    def optimal_design(self) -> str:
        """Return a console summary of the optimal design.

        Returns:
            A rich-markup string with the design variables, the cost, the final iso-cost level
            ``alpha`` for ISORAK-RBDO and, for [nominal values, tolerances] designs, the state of
            the soft constraints.
        """
        log_str = "[bold green]Optimal design: "
        log_str += f"{', '.join(f'[bold yellow]d{i + 1}: [cyan]{self.design_opt[i].numpy():.6f}' for i in range(len(self.design_opt)))} "
        if self.benchmark.nominal_tolerance_design:
            # The cost of a [nominal values, tolerances] design depends on its tolerances only.
            tolerance = torch.tensor(self.results["pop_best"])[:, int(self.benchmark.dim / 2) :][-1]
            log_str += f"[white]- [bold red]Cost: [cyan]{self.benchmark.evaluate_cost(tolerance).numpy():.6f}"
        else:
            log_str += f"[white]- [bold red]Cost: [cyan]{self.benchmark.evaluate_cost(self.design_opt).numpy():.6f}"
        log_str += (
            f" [blue] alpha: [cyan]{self.results['alpha'][-1]:.4f}"
            if self.problem.analysis == "isorakrbdo"
            else ""
        )
        if self.benchmark.nominal_tolerance_design:
            log_str += log_soft_constr(self.benchmark, self.design_opt)
        return log_str

    def generate_plots(self, *, show: bool) -> tuple[Table, Table] | None:
        """Produce every result figure of the run and save it in ``<path>/pdf``.

        Args:
            show: Display the figures instead of closing them.

        Returns:
            For the RBDO analyses, the quasi-Monte Carlo and Monte Carlo reliability tables at the
            optimal design; ``None`` otherwise.
        """
        tables = self._run_all_base_plots()
        plotter.plt.show() if show else plotter.plt.close("all")
        return tables

    def generate_animations(self, *, show: bool) -> None:
        """Render the evolution frames of every animation available for the benchmark.

        Frames are saved in ``<path>/evolution/<animation name>``, one or more per generation.

        Args:
            show: Render the frames one after the other and display them, instead of rendering
                them in a process pool.
        """
        self._animation_evolution_g_space(show=show)
        self._animation_evolution_g_aug_space(show=show)
        self._animation_evolution_cost_space(show=show)
        self._animation_evolution_tol_g_space(show=show)
        self._animation_evolution_tol_space(show=show)
        self._animation_evolution_nom_tol_adim_space(show=show)
        self.animation_evolution_tol_mean_shift_adim_space(show=show)

    @staticmethod
    def generate_animations_videos(analysis: str, benchmark: str, timestamp: str) -> None:
        """Assemble the evolution frames of a run into one video per animation.

        For every sub-directory of ``results/<analysis>/<benchmark>/<timestamp>/evolution`` the
        external script is run as ``bash ./run_animator.sh --analysis=... --benchmark=...
        --timestamp=... --type=<sub-directory>``. Both ``results`` and ``run_animator.sh`` are
        resolved from the current working directory, which must therefore be the directory that
        contains them. The script requires ImageMagick and ffmpeg.

        Args:
            analysis: Analysis of the run (``ddo``, ``rbdo``, ``zakrbdo``, ``isorakrbdo``).
            benchmark: Benchmark id of the run.
            timestamp: Name of the run directory.

        Raises:
            subprocess.CalledProcessError: If the script exits with a non-zero status.
        """
        base_path = utils.Path().cwd() / "results" / analysis / benchmark / timestamp / "evolution"
        anim_types = [f.name for f in base_path.iterdir() if f.is_dir()]
        for anim_type in anim_types:
            command = [
                "bash",
                "./run_animator.sh",
                f"--analysis={analysis}",
                f"--benchmark={benchmark}",
                f"--timestamp={timestamp}",
                f"--type={anim_type}",
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                utils.console_direct.log("Command executed successfully")
                utils.console_direct.log("Output:", result.stdout)
                utils.console_direct.log("\n\n\n")
            except subprocess.CalledProcessError as e:
                utils.console_direct.log(f"Command failed with exit code {e.returncode}")
                utils.console_direct.log(f"Error output: {e.stderr}")
                raise
            except Exception as e:
                utils.console_direct.log(f"An error occurred: {e!s}")
                raise

    def _run_mcs(self) -> tuple[Tensor, Tensor, Table, Table]:
        """Estimate the reliability of the optimal design by quasi-Monte Carlo and Monte Carlo.

        Returns:
            The quasi-Monte Carlo samples, the Monte Carlo samples, the reliability table of the
            former (model used by the optimization) and that of the latter (true limit states).
        """
        samples_qmc = self.benchmark.get_W(
            N_SAMPLES=int(self.problem.batch_qmc),
            design=self.design_opt.unsqueeze(0),
        ).squeeze()
        tab_qmc = self.problem.run_mcs(samples_qmc)
        samples_mcs = self.benchmark.get_W_MCS(
            N_SAMPLES=int(self.problem.size_mc),
            design=self.design_opt.unsqueeze(0),
        )
        tab_mcs = self.problem.run_mcs_reference(samples_mcs)
        return samples_qmc, samples_mcs, tab_qmc, tab_mcs

    def _plot_evo_cost(self) -> None:
        """Plot the cost of the best design against the generation (``evo_cost``)."""
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        plotter.plt.plot(range(1, self.n_gen + 1), self.evo_cost, **plotter.plot_style["1"])
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        ylabel = (
            r"$\mathfrak{c}(\bm{t})$"
            if self.benchmark.nominal_tolerance_design or "linear_stackup" in self.benchmark.id
            else r"$\mathfrak{c}(\mathbf{d})$"
        )
        plotter.plt.ylabel(ylabel)
        plotter.plt.xlim(1, self.n_gen)
        plotter.plt.xscale("log")
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_cost")

    def _plot_evo_design_vars(self) -> None:
        """Plot the best design variables against the generation (``evo_design_vars``)."""
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        for i, d in enumerate(self.evo_design.t()):
            plotter.plt.plot(range(1, self.n_gen + 1), d, **plotter.plot_style[f"{i}"])
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        plotter.plt.ylabel(r"$\bm{d}$")
        plotter.plt.xlim(1, self.n_gen)
        plotter.plt.legend(self.benchmark.label_d, loc="upper left")
        plotter.plt.xscale("log")
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_design_vars")

    def _plot_evo_fit_pop(self) -> None:
        """Plot the median and best fitness of the population per generation (``evo_fitness``)."""
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        plotter.plt.plot(
            range(1, self.n_gen + 1),
            self.results["median_eval"],
            **plotter.plot_style["1"],
        )
        plotter.plt.plot(
            range(1, self.n_gen + 1),
            self.results["pop_best_eval"],
            **plotter.plot_style["0"],
        )
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        plotter.plt.ylabel("Fitness")
        plotter.plt.legend(
            [
                r"$\mathrm{med}\big[f\big(P^{(i)}\big)\big]$",
                r"$\min_{P_m^{(i)} \in P^{(i)}} f\big(P_m^{(i)}\big)$",
            ],
            loc="upper right",
        )
        plotter.plt.xscale("log")
        plotter.plt.xlim(1, self.n_gen)
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_fitness")

    def _plot_evo_fit_best_pop(self) -> None:
        """Plot the best fitness of the population per generation (``evo_fitness_best``)."""
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        plotter.plt.plot(
            range(1, self.n_gen + 1),
            self.results["pop_best_eval"],
            **plotter.plot_style["0"],
        )
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        plotter.plt.ylabel("Fitness")
        plotter.plt.legend(
            [r"$\min_{P_m^{(i)} \in P^{(i)}} f\big(P_m^{(i)}\big)$"],
            loc="upper right",
        )
        plotter.plt.xscale("log")
        plotter.plt.xlim(1, self.n_gen)
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_fitness_best")

    def _plot_evo_prob_fail(self) -> None:
        """Plot the failure probability of each limit state per generation (``evo_Pfk``).

        The dashed line is the threshold of the first limit state.
        """
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        leg = []
        for k, Pfk in enumerate(self.evo_prob_fail.t()):
            plotter.plt.plot(range(1, self.n_gen + 1), Pfk, **plotter.plot_style[f"{k}"])
            leg_label = (
                rf"$\widehat{{\mathcal{{P}}}}_{{f_{k + 1}}}$"
                if "ak" in self.problem.analysis
                else rf"$\mathcal{{P}}_{{f_{k + 1}}}$"
            )
            leg.append(leg_label)
        plotter.plt.hlines(self.problem.Pf_thr[0], 1, self.n_gen, **plotter.plot_style["Pf_thr"])
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        y_label = (
            r"$\widehat{\mathcal{P}}_{f_k}$"
            if "ak" in self.problem.analysis
            else r"$\mathcal{P}_{f_k}$"
        )
        plotter.plt.ylabel(y_label)
        leg.append(r"$\overline{\mathcal{P}}_{f_k}$")
        plotter.plt.legend(leg, loc="upper left")
        plotter.plt.xlim(1, self.n_gen)
        plotter.plt.xscale("log")
        plotter.set_scientific_notation(ax=plotter.plt.gca(), axis="y")
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_Pfk")

    def _plot_evo_beta(self) -> None:
        """Plot the reliability index of each limit state per generation (``evo_Beta``).

        The dashed line is the reliability index of the first failure-probability threshold.
        """
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        leg = []
        for k, betak in enumerate(self.evo_beta.t()):
            plotter.plt.plot(range(1, self.n_gen + 1), betak, **plotter.plot_style[f"{k}"])
            leg_label = (
                rf"$\widehat{{\beta}}_{k + 1}$"
                if "ak" in self.problem.analysis
                else rf"$\beta_{k + 1}$"
            )
            leg.append(leg_label)
        beta_thr = -torch.distributions.Normal(0, 1).icdf(self.problem.Pf_thr[0])
        plotter.plt.hlines(beta_thr, 1, self.n_gen, **plotter.plot_style["Pf_thr"])
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        y_label = r"$\widehat{\beta}_k$" if "ak" in self.problem.analysis else r"$\beta_k$"
        plotter.plt.ylabel(y_label)
        leg.append(r"$\overline{\beta}_k$")
        plotter.plt.legend(leg, loc="upper left")
        plotter.plt.xlim(1, self.n_gen)
        plotter.plt.xscale("log")
        plotter.set_scientific_notation(ax=plotter.plt.gca(), axis="y")
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_Beta")

    def _plot_evo_eps_max(self) -> None:
        """Plot the maximum ESC error per generation with its threshold (``evo_eps_max``)."""
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        leg = []
        plotter.plt.plot(range(1, self.n_gen + 1), self.evo_eps_max, **plotter.plot_style["0"])
        leg_label = r"$\hat{\varepsilon}_{\max}$"
        leg.append(leg_label)
        plotter.plt.hlines(self.problem.eps_max_thr, 1, self.n_gen, **plotter.plot_style["Pf_thr"])
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        y_label = r"$\hat{\varepsilon}_{\max}$"
        plotter.plt.ylabel(y_label)
        leg.append(r"${\varepsilon}_{\text{thr}}$")
        plotter.plt.legend(leg, loc="upper left")
        plotter.plt.xlim(1, self.n_gen)
        plotter.plt.xscale("log")
        plotter.set_scientific_notation(ax=plotter.plt.gca(), axis="y")
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_eps_max")

    def _plot_evo_g_calls(self) -> None:
        """Plot the cumulative number of calls to each limit state (``evo_g_calls``)."""
        g_calls = torch.tensor(self.results["g_calls"])
        plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        for k in range(g_calls.shape[1]):
            plotter.plt.step(
                range(1, self.n_gen + 1),
                g_calls[: self.n_gen, k],
                where="post",
                **plotter.plot_style[str(k)],
            )
        plotter.plt.xlabel(r"$n_{\textrm{gen}}$")
        plotter.plt.ylabel(r"Number of $\mathfrak{g}$-calls")
        legend_labels = [rf"$\mathfrak{{g}}_{k + 1}$-calls" for k in range(g_calls.shape[1])]
        plotter.plt.legend(legend_labels, loc="upper left")
        plotter.plt.xscale("log")
        plotter.plt.xlim(1, self.n_gen)
        plotter.plt.ylim(0, g_calls.max() + 2)
        plotter.plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        self._set_ax_position()
        plotter.utils.save(plotter.plt, path=self.path, filename="evo_g_calls")

    def _plot_optimal_design(self) -> None:
        """Plot the optimal design over the 2D contours of the benchmark (``optimal_design``).

        For the adaptive-Kriging analyses the true limit states are drawn in a light colour, the
        final surrogates are superimposed as dashed lines, and the initial design of experiments
        and the enrichment points are added.
        """
        if self.benchmark.id not in plotting["optimal_design"]:
            return
        augmented = "rbdo" in self.problem.analysis
        nominal_tolerance = self.benchmark.nominal_tolerance_design
        if "ak" in self.problem.analysis:
            self.benchmark_2d.visualize(
                augmented_space=augmented,
                nominal_tolerance=nominal_tolerance,
                color="#FE978E",
            )
            self.surrogate_2d.true_model(color="#D62519")
            train_x = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in self.results["doe"][0]]
            doe = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in self.results["doe"][-1]]
            # Enrichment points: final design of experiments minus the initial one.
            points_added = [utils.tensor_difference(doe[k], train_x[k]) for k in range(len(doe))]
            plotter.plt.scatter(
                train_x[0][:, 0],
                train_x[0][:, 1],
                **plotter.scatter_style["train"],
            )
            for i, pi in enumerate(points_added):
                for pj in pi:
                    plotter.plt.scatter(
                        pj[0],
                        pj[1],
                        **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                    )
                    plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train"])
        else:
            self.benchmark_2d.visualize(
                augmented_space=augmented,
                nominal_tolerance=nominal_tolerance,
            )
        # The figure plane is that of the nominal values for [nominal values, tolerances] designs.
        design_opt = (
            self.design_opt[: int(self.benchmark.dim / 2)]
            if self.benchmark.nominal_tolerance_design
            else self.design_opt
        )
        plotter.plt.scatter(design_opt[0], design_opt[1], **plotter.scatter_style["best"])
        plotter.utils.save(plotter.plt, path=self.path, filename="optimal_design")

    def _plot_optimal_design_nom_tol_adim_space(self) -> None:
        """Plot the optimal design in the dimensionless nominal/tolerance plane.

        Cantilever beam benchmarks only: one point per toleranced dimension, at its nominal value
        and tolerance both scaled to [0, 1] by their bounds. Saved as
        ``optimal_design_nom_tol_adim_space``.
        """
        if "cantilever_4d" in self.benchmark.id:
            labels = self.benchmark.label_X
            points_lab = ["nom_tol_1", "nom_tol_2", "nom_tol_3", "nom_tol_4"]
        elif "cantilever_2d" in self.benchmark.id:
            labels = self.benchmark.label_X
            points_lab = ["nom_tol_1", "nom_tol_2"]
        else:
            return
        mean_shift = "mean_shift" in self.benchmark.id
        plotter.plot_tolerance_2d.plot_nom_tol_adim_space(mean_shift=mean_shift)
        nominal = self.design_opt[: int(self.benchmark.dim / 2)]
        tolerance = self.design_opt[int(self.benchmark.dim / 2) :]
        bounds_nom = self.benchmark.bounds[: len(labels), :]
        bounds_tol = self.benchmark.bounds[len(labels) :, :]
        nominal_norm = (nominal - bounds_nom.min(dim=1).values) / (
            bounds_nom.max(dim=1).values - bounds_nom.min(dim=1).values
        )
        tolerance_norm = (tolerance - bounds_tol.min(dim=1).values) / (
            bounds_tol.max(dim=1).values - bounds_tol.min(dim=1).values
        )
        for k, nom, tol in zip(points_lab, nominal_norm, tolerance_norm, strict=False):
            plotter.plt.scatter(nom, tol, **plotter.scatter_style[k])
        plotter.plt.legend(labels)
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()
        plotter.utils.save(
            plotter.plt,
            path=self.path,
            filename="optimal_design_nom_tol_adim_space",
        )

    def _plot_optimal_design_tol_mean_shift_adim_space(self) -> None:
        """Plot the optimal design in the dimensionless mean-shift plane.

        One point per toleranced dimension at ``(t* / 2, t*)``, where ``t*`` is its tolerance
        scaled to [0, 1] by its bounds. Saved as ``optimal_design_tol_mean_shift_adim_space``.
        """
        if "cantilever_4d" in self.benchmark.id:
            points_lab = ["nom_tol_1", "nom_tol_2", "nom_tol_3", "nom_tol_4"]
        elif "cantilever_2d" in self.benchmark.id or "linear_stackup" in self.benchmark.id:
            points_lab = ["nom_tol_1", "nom_tol_2"]
        else:
            return
        labels = self.benchmark.label_X
        plotter.plot_tolerance_2d.plot_tol_mean_shift_adim_space()

        tolerance = (
            self.design_opt[int(self.benchmark.dim / 2) :]
            if self.benchmark.nominal_tolerance_design
            else self.design_opt
        )
        bounds_tol = (
            self.benchmark.bounds[len(labels) :, :]
            if self.benchmark.nominal_tolerance_design
            else self.benchmark.bounds
        )
        tolerance_norm = (tolerance - bounds_tol.min(dim=1).values) / (
            bounds_tol.max(dim=1).values - bounds_tol.min(dim=1).values
        )
        delta_norm = tolerance_norm / 2
        for k, nom, tol in zip(points_lab, delta_norm, tolerance_norm, strict=False):
            plotter.plt.scatter(nom, tol, **plotter.scatter_style[k])
        plotter.plt.legend(labels)
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()
        plotter.utils.save(
            plotter.plt,
            path=self.path,
            filename="optimal_design_tol_mean_shift_adim_space",
        )

    def _plot_optimal_design_history(self) -> None:
        """Plot the path of the best design over the generations (``optimal_design_history``).

        Same background as :meth:`_plot_optimal_design`, always in the design space, with the axes
        zoomed on the region registered in ``benchmark_zoom`` when there is one.
        """
        if self.benchmark.id not in plotting["optimal_design"]:
            return
        nominal_tolerance = self.benchmark.nominal_tolerance_design
        if "ak" in self.problem.analysis:
            self.benchmark_2d.visualize(
                augmented_space=False,
                nominal_tolerance=nominal_tolerance,
                color="#FE978E",
            )
            self.surrogate_2d.true_model(color="#D62519")
            train_x = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in self.results["doe"][0]]
            doe = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in self.results["doe"][-1]]
            points_added = [utils.tensor_difference(doe[k], train_x[k]) for k in range(len(doe))]
            plotter.plt.scatter(
                train_x[0][:, 0],
                train_x[0][:, 1],
                **plotter.scatter_style["train"],
            )
            for i, pi in enumerate(points_added):
                for pj in pi:
                    plotter.plt.scatter(
                        pj[0],
                        pj[1],
                        **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                    )
                    plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train"])
        else:
            self.benchmark_2d.visualize(augmented_space=False, nominal_tolerance=nominal_tolerance)
        evo_design = (
            self.evo_design[:, : int(self.benchmark.dim / 2)]
            if self.benchmark.nominal_tolerance_design
            else self.evo_design
        )
        plotter.plt.plot(evo_design[:, 0], evo_design[:, 1], **plotter.plot_style["trace"])
        plotter.plt.scatter(evo_design[:, 0], evo_design[:, 1], **plotter.scatter_style["trace"])
        design_opt = (
            self.design_opt[: int(self.benchmark.dim / 2)]
            if self.benchmark.nominal_tolerance_design
            else self.design_opt
        )
        plotter.plt.scatter(design_opt[0], design_opt[1], **plotter.scatter_style["best"])
        zoom_config = self.benchmark_zoom[self.problem.analysis].get(self.benchmark.id, {})
        if zoom_config:
            plotter.plt.xlim(zoom_config["zoom_xlim"])
            plotter.plt.ylim(zoom_config["zoom_ylim"])
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.utils.save(plotter.plt, path=self.path, filename="optimal_design_history")

    def _plot_optimal_design_history_nom_tol_adim_space(self) -> None:
        """Plot the path of the best design in the dimensionless nominal/tolerance plane.

        Cantilever beam benchmarks only: one path per toleranced dimension, ending on the optimal
        design. Saved as ``optimal_design_nom_tol_adim_space_history``.
        """
        if "cantilever_4d" in self.benchmark.id:
            labels = self.benchmark.label_X
            points_lab = ["nom_tol_1", "nom_tol_2", "nom_tol_3", "nom_tol_4"]
        elif "cantilever_2d" in self.benchmark.id:
            labels = self.benchmark.label_X
            points_lab = ["nom_tol_1", "nom_tol_2"]
        else:
            return
        mean_shift = "mean_shift" in self.benchmark.id
        plotter.plot_tolerance_2d.plot_nom_tol_adim_space(mean_shift=mean_shift)
        nominal = self.evo_design[:, : int(self.benchmark.dim / 2)]
        tolerance = self.evo_design[:, int(self.benchmark.dim / 2) :]
        bounds_nom = self.benchmark.bounds[: len(labels), :]
        bounds_tol = self.benchmark.bounds[len(labels) :, :]
        nominal_norm = (nominal - bounds_nom.min(dim=1).values) / (
            bounds_nom.max(dim=1).values - bounds_nom.min(dim=1).values
        )
        tolerance_norm = (tolerance - bounds_tol.min(dim=1).values) / (
            bounds_tol.max(dim=1).values - bounds_tol.min(dim=1).values
        )
        for nom, tol in zip(nominal_norm.t(), tolerance_norm.t(), strict=False):
            plotter.plt.scatter(nom, tol, **plotter.scatter_style["trace"])
        leg_list = []
        for k in range(len(points_lab)):
            plotter.plt.plot(
                nominal_norm[:, k],
                tolerance_norm[:, k],
                **plotter.plot_style["trace"],
            )
            leg_list.append(
                plotter.plt.scatter(
                    nominal_norm[-1, k],
                    tolerance_norm[-1, k],
                    **plotter.scatter_style[points_lab[k]],
                ),
            )
        plotter.plt.legend(leg_list, labels)
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()
        plotter.utils.save(
            plotter.plt,
            path=self.path,
            filename="optimal_design_nom_tol_adim_space_history",
        )

    def _plot_optimal_design_history_tol_space(self) -> None:
        """Plot the path of the best design over the cost contours in the tolerance plane.

        The iso-cost mask is added for ISORAK-RBDO. Saved as ``optimal_design_history_tol_space``.
        """
        if self.benchmark.id not in plotting["optimal_design_history_tol_space"]:
            return
        # Without a fixed nominal point the design vector is [nominal values, tolerances] and the
        # tolerances are its second half; otherwise the design variables are the tolerances.
        if self.benchmark.nominal.numel() <= 0:
            self.benchmark.cost.visualize(
                design_bounds=self.benchmark.bounds[int(self.benchmark.dim / 2) :],
                bench_id=self.benchmark.id,
            )
            if "iso" in self.problem.analysis:
                self.benchmark.cost.visualize_mask_iso(
                    design_bounds=self.benchmark.bounds[int(self.benchmark.dim / 2) :],
                )
            plotter.plt.plot(
                self.evo_design[:, 2],
                self.evo_design[:, 3],
                **plotter.plot_style["trace"],
            )
            plotter.plt.scatter(
                self.evo_design[:, 2],
                self.evo_design[:, 3],
                **plotter.scatter_style["trace"],
            )
            plotter.plt.scatter(
                self.design_opt[2],
                self.design_opt[3],
                **plotter.scatter_style["best"],
            )
        else:
            self.benchmark.cost.visualize(
                design_bounds=self.benchmark.bounds,
                bench_id=self.benchmark.id,
            )
            if "iso" in self.problem.analysis:
                self.benchmark.cost.visualize_mask_iso(
                    design_bounds=self.benchmark.bounds,
                )
            plotter.plt.plot(
                self.evo_design[:, 0],
                self.evo_design[:, 1],
                **plotter.plot_style["trace"],
            )
            plotter.plt.scatter(
                self.evo_design[:, 0],
                self.evo_design[:, 1],
                **plotter.scatter_style["trace"],
            )
            plotter.plt.scatter(
                self.design_opt[0],
                self.design_opt[1],
                **plotter.scatter_style["best"],
            )
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()
        plotter.utils.save(
            plotter.plt,
            path=self.path,
            filename="optimal_design_history_tol_space",
        )

    def _plot_optimal_design_mcs(self) -> tuple[Table, Table] | None:
        """Plot the sample clouds of the optimal design in the augmented space.

        Two figures are saved, ``optimal_design_qmc`` and ``optimal_design_mcs``, with the failing
        samples in red and the safe ones in green. The reliability estimates are computed for every
        benchmark; the figures only for those listed in ``plotting["optimal_design_mcs"]``.

        Returns:
            The quasi-Monte Carlo and Monte Carlo reliability tables at the optimal design.
        """
        samples_qmc, samples_mcs, table_qmc, table_mcs = self._run_mcs()
        if self.benchmark.id not in plotting["optimal_design_mcs"]:
            return table_qmc, table_mcs

        def __plot(samples: Tensor, method: str) -> None:
            """Draw one sample cloud, classified with the model used by the optimization."""
            g, _ = self.problem.evaluate_hard(samples)
            neg = (g <= 0).any(dim=1)
            pos = (g > 0).all(dim=1)
            nominal_tolerance = self.benchmark.nominal_tolerance_design
            if "ak" in self.problem.analysis:
                self.benchmark_2d.visualize(
                    augmented_space=True,
                    nominal_tolerance=nominal_tolerance,
                    color="#FE978E",
                )
                self.surrogate_2d.true_model(color="#D62519")
            else:
                self.benchmark_2d.visualize(
                    augmented_space=True,
                    nominal_tolerance=nominal_tolerance,
                )
            plotter.plt.scatter(
                samples[neg, 0],
                samples[neg, 1],
                **plotter.scatter_style["mcs_fail"],
            )
            plotter.plt.scatter(
                samples[pos, 0],
                samples[pos, 1],
                **plotter.scatter_style["mcs_safe"],
            )
            # The linear stack-up designs are tolerances: the cloud is centred on the fixed
            # nominal point, not on the design.
            if "linear_stackup" in self.benchmark.id:
                plotter.plt.scatter(
                    self.benchmark.nominal[0, 0],
                    self.benchmark.nominal[0, 1],
                    **plotter.scatter_style["mcs_best"],
                )
            else:
                plotter.plt.scatter(
                    self.design_opt[0],
                    self.design_opt[1],
                    **plotter.scatter_style["mcs_best"],
                )
            plotter.utils.save(plotter.plt, path=self.path, filename=f"optimal_design_{method}")

        __plot(samples_qmc, "qmc")
        __plot(samples_mcs, "mcs")

        return table_qmc, table_mcs

    def _plot_forces_stress(self) -> None:
        """Plot the densities of the force and of the stress at the optimal design.

        Cantilever beam benchmarks only. The two responses are recovered from the limit states of
        a Monte Carlo sample at the optimal design, and their kernel density estimates are saved as
        ``force`` and ``sigma``, with the admissible bounds as dashed vertical lines.
        """
        samples_mcs = self.benchmark.get_W_MCS(
            N_SAMPLES=int(self.problem.size_mc),
            design=self.design_opt.unsqueeze(0),
        )
        g_mcs, _ = self.problem.evaluate_hard(samples_mcs)
        fig1 = plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        fig2 = plotter.plt.figure(figsize=plotter.fig_size("rectangleQuasi"))
        col = ["#7CB5D3", "#BB6162", "#BB6162"]
        # Invert the limit states g1 = F - F_min, g2 = F_max - F and g3 = sigma_max - sigma to get
        # the force F (twice) and the stress sigma back.
        lsf_thresh = [5.0, 25.0, 100.0]
        g_mcs[:, 0] = g_mcs[:, 0] + lsf_thresh[0]
        g_mcs[:, 1] = lsf_thresh[1] - g_mcs[:, 1]
        g_mcs[:, 2] = lsf_thresh[2] - g_mcs[:, 2]
        for k in range(g_mcs.shape[1]):
            # Columns 0 and 1 both hold the force: its density is drawn once.
            if k == 1:
                continue
            if k in (0, 1):
                plotter.plt.figure(fig1)
            else:
                plotter.plt.figure(fig2)
            xx = torch.linspace(
                g_mcs[:, k].cpu().min(),
                g_mcs[:, k].cpu().max(),
                200,
            )
            kde = KernelDensity(kernel="gaussian", bandwidth=1.10).fit(
                g_mcs[:, k].cpu().reshape(-1, 1).numpy(),
            )
            log_dens = np.array(kde.score_samples(xx.reshape(-1, 1).numpy()))
            plotter.plt.fill_between(
                xx,
                torch.exp(torch.tensor(log_dens)).squeeze().numpy(),
                0,
                color=col[k],
                alpha=0.4,
            )
            plotter.plt.plot(
                xx,
                torch.exp(torch.tensor(log_dens)).squeeze(),
                color=col[k],
                linewidth=0.5,
                label="_nolegend_",
            )
            plotter.plt.gca().set_ylim(
                bottom=float(torch.exp(torch.tensor(log_dens)).squeeze().min()),
            )
        for k in range(g_mcs.shape[1]):
            if k in (0, 1):
                plotter.plt.figure(fig1)
            else:
                plotter.plt.figure(fig2)
            plotter.plt.axvline(x=lsf_thresh[k], color="k", linestyle="--", linewidth=0.55)
        leg = [
            [r"$F$", r"$\overline{F}_{\min}$", r"$\overline{F}_{\max}$"],
            [r"$\sigma$", r"$\overline{\sigma}_{\max}$"],
        ]
        xlabel = [r"$F$", r"$\sigma$"]
        fig_name = ["force", "sigma"]
        for k, ki in enumerate([fig1, fig2]):
            plotter.plt.figure(ki)
            plotter.plt.xlabel(f"{xlabel[k]}")
            plotter.plt.ylabel("Frequency")
            plotter.plt.legend(leg[k], loc="upper right")
            plotter.plt.tight_layout()
            plotter.utils.save(plotter.plt, path=self.path, filename=f"{fig_name[k]}")

    def _run_all_base_plots(self) -> tuple[Table, Table] | None:
        """Produce every figure that applies to the analysis and the benchmark of the run.

        Returns:
            For the RBDO analyses, the quasi-Monte Carlo and Monte Carlo reliability tables at the
            optimal design; ``None`` otherwise.
        """
        tables = None
        self._plot_evo_cost()
        self._plot_evo_design_vars()
        self._plot_evo_fit_pop()
        self._plot_evo_fit_best_pop()
        self._plot_optimal_design()
        self._plot_optimal_design_history()
        if self.benchmark.id in plotting["animation_tol_space"]:
            self._plot_optimal_design_history_tol_space()
        if self.benchmark.nominal_tolerance_design:
            self._plot_optimal_design_nom_tol_adim_space()
            self._plot_optimal_design_history_nom_tol_adim_space()
            self._plot_forces_stress()
        if "mean_shift" in self.benchmark.id:
            self._plot_optimal_design_tol_mean_shift_adim_space()
        if "rbdo" in self.problem.analysis:
            self._plot_evo_prob_fail()
            self._plot_evo_beta()
            tables = self._plot_optimal_design_mcs()
        if "ak" in self.problem.analysis:
            self._plot_evo_g_calls()
            self._plot_evo_eps_max()
        return tables

    def _animation_evolution_g_space(self, *, show: bool) -> None:
        """Render the design-space frames of every generation (``evolution/evo_g_space``)."""
        if self.benchmark.id not in plotting["animation_g_space"]:
            return
        aug_w = False
        if show:
            for gen in range(self.n_gen):
                animation_evolution_g_space_frame(self.path, gen, show, aug_w)
            return
        funtoeval = [
            (animation_evolution_g_space_frame, self.path, gen, show, aug_w)
            for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    def _animation_evolution_g_aug_space(self, *, show: bool) -> None:
        """Render the augmented-space frames of every generation (``evolution/evo_g_aug_space``).

        They are the design-space frames with, in addition, the sample clouds of the random
        variables around the feasible individuals and the best design.
        """
        if self.benchmark.id not in plotting["animation_g_aug_space"]:
            return
        aug_w = True
        if show:
            for gen in range(self.n_gen):
                animation_evolution_g_space_frame(self.path, gen, show, aug_w)
            return
        funtoeval = [
            (animation_evolution_g_space_frame, self.path, gen, show, aug_w)
            for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    def _animation_evolution_tol_space(self, *, show: bool) -> None:
        """Render the tolerance-space frames of every generation (``evolution/evo_tol_space``)."""
        if self.benchmark.id not in plotting["animation_tol_space"]:
            return
        if show:
            for gen in range(self.n_gen):
                animation_evolution_tol_space_frame(self.path, gen, show)
            return
        funtoeval = [
            (animation_evolution_tol_space_frame, self.path, gen, show) for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    def _animation_evolution_cost_space(self, *, show: bool) -> None:
        """Render the cost-space frames of every generation (``evolution/evo_cost_space``)."""
        if self.benchmark.id not in plotting["animation_cost_space"]:
            return
        if show:
            for gen in range(self.n_gen):
                animation_evolution_cost_space_frame(self.path, gen, show)
            return
        funtoeval = [
            (animation_evolution_cost_space_frame, self.path, gen, show)
            for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    def _animation_evolution_tol_g_space(self, *, show: bool) -> None:
        """Render the tolerance-interval frames of every generation (``evolution/evo_tol_g_space``).

        Produced for the same benchmarks as the tolerance-space frames.
        """
        if self.benchmark.id not in plotting["animation_tol_space"]:
            return
        if show:
            for gen in range(self.n_gen):
                animation_evolution_tol_g_space_frame(self.path, gen, show)
            return
        funtoeval = [
            (animation_evolution_tol_g_space_frame, self.path, gen, show)
            for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    def _animation_evolution_nom_tol_adim_space(self, *, show: bool) -> None:
        """Render the dimensionless nominal/tolerance frames of every generation.

        Frames are saved in ``evolution/evo_nom_tol_adim_space``.
        """
        if self.benchmark.id not in plotting["animation_nom_tol_adim_space"]:
            return
        if show:
            for gen in range(self.n_gen):
                animation_evolution_nom_tol_adim_space_frame(self.path, gen, show)
            return
        funtoeval = [
            (animation_evolution_nom_tol_adim_space_frame, self.path, gen, show)
            for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    def animation_evolution_tol_mean_shift_adim_space(self, *, show: bool) -> None:
        """Render the dimensionless mean-shift frames of every generation.

        Frames are saved in ``evolution/evo_tol_mean_shift_adim_space``, for the same benchmarks as
        the dimensionless nominal/tolerance frames.

        Args:
            show: Render the frames one after the other and display them, instead of rendering
                them in a process pool.
        """
        if self.benchmark.id not in plotting["animation_nom_tol_adim_space"]:
            return
        if show:
            for gen in range(self.n_gen):
                animation_evolution_tol_mean_shift_adim_space_frame(self.path, gen, show)
            return
        funtoeval = [
            (animation_evolution_tol_mean_shift_adim_space_frame, self.path, gen, show)
            for gen in range(self.n_gen)
        ]
        utils.parallelize(funtoeval)  # pyright: ignore

    @staticmethod
    def _set_ax_position() -> None:
        """Pin the axes box to a fixed position, so that all the evolution curves share a frame."""
        ax = plotter.plt.gca()
        ax.set_position((0.22, 0.15, 0.76, 0.78))
