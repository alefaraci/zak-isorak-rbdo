"""Run one analysis on one benchmark.

Usage (from any directory)::

    python examples/run.py --analysis zakrbdo --benchmark four_branches --seed 27

``--analysis`` is one of ``ddo``, ``rbdo``, ``zakrbdo`` or ``isorakrbdo``; ``--benchmark`` is the
name of one of the ``<benchmark>.yaml`` files stored next to this script.

Results are written to ``examples/results/<analysis>/<benchmark>/<timestamp>_seed<seed>/``:

* ``results.json``: one NDJSON row per generation (best design, cost, failure probabilities, ...);
* ``opt_problem.pickle``: the optimization problem, including the trained surrogates;
* ``analysis.log``: a copy of the console output.
"""

import os

from zak import utils, console
from zak.loggers import PolarsLogger, save_loggers_pl
from zak.problem import OptimizationProblem
from zak.searcher.ga import GeneticAlgorithmSearcher

HERE = utils.Path(__file__).parent


def results_path(analysis: str, benchmark_id: str, seed: int) -> utils.Path:
    """Create and return the output directory of a run."""
    path = HERE / "results" / analysis / benchmark_id / f"{utils.time_stamp()}_seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def classic_analysis(opt_problem: OptimizationProblem, analysis: str, benchmark_id: str) -> None:
    """Run a single genetic-algorithm search (DDO, RBDO and ZAK-RBDO)."""
    searcher = GeneticAlgorithmSearcher(opt_problem)
    logger_pl = PolarsLogger(searcher)
    path = results_path(analysis, benchmark_id, opt_problem.seed)

    with utils.live:
        searcher.run(logger=logger_pl, path=path)

    logger_pl.save(path / "results.json")
    utils.save(path=path, variables={"opt_problem": opt_problem}, filename="opt_problem")
    console.log(f"[bold green]Results saved in: [bold yellow]{path}")


def isocost_analysis(opt_problem: OptimizationProblem, analysis: str, benchmark_id: str) -> None:
    """Run ISORAK-RBDO: a sequence of searches over growing isocost regions.

    Each search is restricted to the isocost region of level ``alpha``. When the search ends
    without a reliable design, the region is enlarged by bisection between the current isocost
    level and the upper bound estimated by the searcher, and a new search starts from the design
    of experiments accumulated so far.
    """
    path = results_path(analysis, benchmark_id, opt_problem.seed)
    config = utils.load_yaml(path=HERE, filename=f"{benchmark_id}.yaml")[analysis.upper()]
    alpha_init = config["ga"]["alpha_init"]

    with utils.live:
        loggers_pl, doe, n_gen_alpha = [], None, 0

        k = 0
        stop_alpha = False
        alpha_i = alpha_init

        while not stop_alpha:
            # Reset the progress bar for the new isocost region.
            task_ids_to_clear = [task.id for task in utils.progressbar.tasks]
            for task_id in task_ids_to_clear:
                utils.progressbar.remove_task(task_id)
            utils.live.update(utils.Group(utils.progressbar, utils.Text("")))  # pyright: ignore

            # Search inside the current isocost region, reusing the points already evaluated.
            opt_problem.increment_alpha(alpha_i)
            searcher = GeneticAlgorithmSearcher(opt_problem)
            if doe is not None:
                searcher.set_doe(doe)
            doe = searcher.doe
            searcher.set_n_gen_alpha(n_gen_alpha)
            loggers_pl.append(PolarsLogger(searcher))
            stop_alpha = searcher.run(logger=loggers_pl[k], path=path)
            if stop_alpha:
                console.log(f"[green bold]Alpha found : [cyan]{alpha_i:.2f}[/cyan] !")
                break
            n_gen_alpha += searcher.step_count

            # Bisection between the current isocost level and its estimated upper bound.
            alpha_ub = searcher.benchmark.cost.alpha_ub
            if alpha_ub > 1.0:
                alpha_ub = 1.0
                searcher.benchmark.cost._alpha_ub = alpha_ub

            isolevels = searcher.benchmark.cost.isolevels
            c_delta_alpha = isolevels[alpha_i] + 0.5 * (isolevels[alpha_ub] - isolevels[alpha_i])
            alpha_i = next(key for key, value in isolevels.items() if value >= c_delta_alpha)

            del searcher
            utils.clear_gpu_cache()
            k += 1

    save_loggers_pl(loggers_pl, path / "results.json")
    utils.save(path=path, variables={"opt_problem": opt_problem}, filename="opt_problem")
    console.log(f"[bold green]Results saved in: [bold yellow]{path}")


def main() -> None:
    """Parse the command line and run the requested analysis."""
    # `OptimizationProblem` reads `<benchmark>.yaml` and `settings.yaml` from the working directory.
    os.chdir(HERE)
    analysis, benchmark_id, seed = utils.args_parser()
    opt_problem = OptimizationProblem(analysis=analysis, benchmark_id=benchmark_id, seed=seed)
    analyzer = isocost_analysis if "iso" in analysis else classic_analysis
    analyzer(opt_problem, analysis, benchmark_id)


if __name__ == "__main__":
    main()
