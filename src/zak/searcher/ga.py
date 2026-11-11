"""Genetic-algorithm searcher for the DDO, RBDO, ZAK-RBDO and ISORAK-RBDO analyses.

`GeneticAlgorithmSearcher` extends the EvoTorch genetic algorithm. For ZAK-RBDO and ISORAK-RBDO
it also trains the Kriging surrogates of the hard constraints and enriches their designs of
experiments between two generations: globally with the learning function A, locally around the
best design with the U learning function and, once the search stagnates, with AK-MCS.
"""

from copy import deepcopy

from attrs import field, define
from torch import Tensor
from evotorch.core import Problem, SolutionBatch
from evotorch.operators import TwoPointCrossOver
from evotorch.algorithms import GeneticAlgorithm
from torch.distributions import Normal
from evotorch.operators.base import CopyingOperator
import torch

from .. import utils
from ..utils import TaskID
from ..loggers import PolarsLogger, log_ddo, log_rbdo, log_zakrbdo, log_isorakrbdo
from ..problem import OptimizationProblem
from ..problem import stopping_criteria as sc
from .clustering import not_in_cluster
from ..benchmarks import BenchmarkBase
from .convergence import check_convergence
from ..random_vars import SobolTrainingPoints
from ..gaussianprocess.gp import ListGPRBoTorch
from ..problem.optimization import BETA_MAX

# Value of the U learning function, U = |mean| / standard deviation of a surrogate prediction,
# above which the sign of the prediction is considered reliable: the probability of a wrong sign
# is then below Phi(-2), about 2.3 %.
U_THR = 2


class GaussianMutation(CopyingOperator):
    """Gaussian mutation operator with one standard deviation per decision variable.

    Follows the algorithm description in:

        Sean Luke, 2013, Essentials of Metaheuristics, Lulu, second edition
        available for free at http://cs.gmu.edu/~sean/book/metaheuristics/

    The standard deviation is stored as a tensor, which the searcher updates between two
    generations to shrink the mutation step.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        stdev: Tensor,
        mutation_probability: float | None,
    ) -> None:
        """Initialize the Gaussian mutation.

        Args:
            problem: The problem object to work with.
            stdev: The standard deviation of the Gaussian noise to apply on
                each decision variable.
            mutation_probability: The probability of mutation, for each
                decision variable.
                If None, the value of this argument becomes 1.0, which means
                that all of the decision variables will be affected by the
                mutation.
        """
        super().__init__(problem)
        self._mutation_probability = (
            1.0 if mutation_probability is None else float(mutation_probability)
        )
        self._stdev = stdev

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        """Return a mutated copy of the batch, kept within the bounds of the problem."""
        result = deepcopy(batch)
        data = result.access_values()
        mutation_matrix = self.problem.make_uniform_shaped_like(data) <= self._mutation_probability
        stdev = torch.ones(mutation_matrix.size()) * self._stdev
        data[mutation_matrix] += stdev[mutation_matrix] * self.problem.make_gaussian_shaped_like(
            data[mutation_matrix],
        )
        data[:] = self._respect_bounds(data)
        return result


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=False,
    eq=False,
)
class GeneticAlgorithmSearcher(GeneticAlgorithm):
    """Genetic algorithm running the DDO, RBDO, ZAK-RBDO and ISORAK-RBDO analyses.

    The searcher is an elitist EvoTorch genetic algorithm with two-point cross-over and Gaussian
    mutation, whose parameters are read from the configuration of the problem. For ZAK-RBDO and
    ISORAK-RBDO it also builds the initial Kriging surrogates of the hard constraints and
    enriches them during the search, see `_run_zakrbdo` and `_run_isorakrbdo`.

    Attributes:
        _problem: The optimization problem.
        _pop_init: Initial population of the genetic algorithm.
        _mc_global: Global candidate points for the enrichment of the surrogates, in the
            augmented space (design variables and random variables).
        _mc_global_design: Designs corresponding to the points of ``_mc_global``.
        _doe: Design of experiments (training points) of each surrogate.
        _n_gen_alpha: Number of generations spent in the previous isocost regions
            (ISORAK-RBDO).
    """

    _problem: OptimizationProblem
    _pop_init: Tensor = field(init=False)
    _mc_global: Tensor = field(init=False)
    _mc_global_design: Tensor = field(init=False)
    _doe: list[Tensor] = field(default=None)
    _n_gen_alpha: int = field(default=0)

    @property
    def covar_module(self) -> list[str]:
        """Return the name of the covariance kernel of each surrogate."""
        return self._problem.config["gp"]["covar_module"]

    @property
    def batch_train_set(self) -> int:
        """Return the number of points of the initial design of experiments."""
        return self._problem.config["gp"]["batch_train_set"]

    @property
    def size_candidate_global(self) -> int:
        """Return the number of global candidate points for the enrichment of the surrogates."""
        return int(self.problem.config["gp"]["size_candidate_global"])

    @property
    def rho_thr(self) -> float:
        """Return the threshold on the rho indicator above which the global enrichment stops."""
        return self._problem.config["gp"]["rho_thr"]

    @property
    def eps_max_thr(self) -> float:
        """Return the threshold on the maximum relative error of the ESC stopping criterion."""
        return self._problem.config["gp"]["eps_max_thr"]

    @property
    def min_samples(self) -> int:
        """Return the ``min_samples`` parameter of the DBSCAN filter."""
        return self._problem.config["DBSCAN"]["min_samples"]

    @property
    def eps(self) -> float:
        """Return the neighborhood radius of the DBSCAN filter, measured in the U-space."""
        return self._problem.config["DBSCAN"]["eps"]

    @property
    def size_mc(self) -> int:
        """Return the size of the Monte Carlo population of the AK-MCS refinement."""
        return int(self._problem.config["rbdo"]["size_mc"])

    @property
    def n_gen(self) -> int:
        """Return the maximum number of generations."""
        return self._problem.config["ga"]["n_gen"]

    @property
    def n_ind(self) -> int:
        """Return the number of individuals of the population."""
        return int(self._problem.config["ga"]["n_ind"])

    @property
    def tournament_size(self) -> int:
        """Return the size of the tournaments selecting the parents."""
        return int(self._problem.config["ga"]["tournament_size"])

    @property
    def stdev(self) -> Tensor:
        """Return the initial standard deviation of the Gaussian mutation, per variable."""
        return torch.tensor(self._problem.config["ga"]["stdev"])

    @property
    def mutation_probability(self) -> float:
        """Return the probability of mutating each decision variable."""
        return self._problem.config["ga"]["mutation_probability"]

    @property
    def cross_over_rate(self) -> float:
        """Return the cross-over rate."""
        return self._problem.config["ga"]["cross_over_rate"]

    @property
    def window(self) -> int:
        """Return the number of generations of the stagnation window."""
        return self._problem.config["ga"]["window"]

    @property
    def improvement_max(self) -> float:
        """Return the threshold on the normalized improvement of the convergence check."""
        return self._problem.config["ga"]["improvement_max"]

    @property
    def problem(self) -> OptimizationProblem:
        """Return the optimization problem."""
        return self._problem

    @problem.setter
    def problem(self, value) -> None:
        self._problem = value

    @property
    def num_actors(self) -> int:
        """Return the number of actors evaluating the population in parallel."""
        return self.problem.num_actors

    @property
    def mc_global(self) -> Tensor:
        """Return the global candidate points still available for the enrichment."""
        return self._mc_global

    @property
    def doe(self) -> list[Tensor]:
        """Return the design of experiments of each surrogate."""
        return self._doe

    @property
    def benchmark(self) -> BenchmarkBase:
        """Return the benchmark."""
        return self._problem.benchmark

    @property
    def ng(self) -> int:
        """Return the number of hard constraints, i.e. of surrogates."""
        return len(self.benchmark.hard)

    @property
    def g_calls(self) -> Tensor:
        """Return the number of evaluations of each hard constraint (size of each DoE)."""
        return torch.tensor([doe_k.shape[0] for doe_k in self._doe])

    @staticmethod
    def Pr_feas(mean, var) -> Tensor:
        """Return the probability of feasibility ``Phi(mean / sqrt(var))`` of each prediction.

        Called with ``-mean``, it returns the predicted probability of failure instead.
        """
        return Normal(0, 1).cdf(mean.cpu() / var.sqrt().cpu())

    def Pr_cost(self, mc_pop_global: Tensor) -> Tensor:
        """Return the cost benefit of each candidate: one minus its normalized cost."""
        # The cost of a [nominal values, tolerances] design depends on the tolerances only.
        mc_pop_global = (
            mc_pop_global[..., int(self.benchmark.dim / 2) :]
            if self.problem.benchmark.nominal_tolerance_design
            else mc_pop_global
        )
        cost_norm = self.benchmark.cost.fn_norm(mc_pop_global)
        return 1 - cost_norm

    @staticmethod
    def desiderability(Pc, Pfeas) -> Tensor:
        """Return the cost benefit times the probability of feasibility for all constraints."""
        return Pc * Pfeas.prod(dim=1)

    def set_doe(self, doe: list[Tensor]) -> None:
        """Set the design of experiments of each surrogate."""
        self._doe = doe

    def set_n_gen_alpha(self, n_gen: int) -> None:
        """Set the number of generations spent in the previous isocost regions."""
        self._n_gen_alpha = n_gen

    def run(
        self,
        logger: PolarsLogger,
        path: utils.Path,
        *,
        stop_alpha: bool = False,
    ) -> None | bool:
        """Run the analysis selected in the problem.

        Args:
            logger: Logger attached to the searcher; the convergence checks read its history.
            path: Directory of the run, where the console log is saved.
            stop_alpha: ISORAK-RBDO only, value returned when the search ends without having
                converged in the current isocost region.

        Returns:
            None for DDO, RBDO and ZAK-RBDO. For ISORAK-RBDO, True if the search has converged
            in the current isocost region, else ``stop_alpha``.

        Raises:
            ValueError: If the analysis is not supported.
        """
        match self.problem.analysis:
            case "ddo" | "rbdo":
                return self._run_ddo_rbdo(logger, path)
            case "zakrbdo":
                return self._run_zakrbdo(logger, path)
            case "isorakrbdo":
                return self._run_isorakrbdo(logger, path, stop_alpha=stop_alpha)
            case _:
                msg = f"""Analysis not supported: {self.problem.analysis}
                    Supported analysis: ['ddo', 'rbdo', 'zakrbdo', 'isorakrbdo']"""
                raise ValueError(msg)

    def __attrs_post_init__(self) -> None:
        """Train the initial surrogates when needed and initialize the genetic algorithm."""
        # The surrogate-based analyses ("zakrbdo" and "isorakrbdo") need initial surrogates,
        # unless the problem already holds the ones trained in a previous isocost region.
        if "ak" in self.problem.analysis and self.problem.metamodels is None:
            self.__init_metamodels()
        super().__init__(
            problem=self.problem,
            popsize=self.n_ind,
            operators=[
                TwoPointCrossOver(
                    self.problem,
                    tournament_size=self.tournament_size,
                    cross_over_rate=self.cross_over_rate,
                ),
                GaussianMutation(
                    self.problem,
                    stdev=self.stdev,
                    mutation_probability=self.mutation_probability,
                ),
            ],
            # The parents are re-evaluated at every generation: the fitness of a design depends
            # on the surrogates, which are enriched during the search, and on the other designs
            # evaluated with it (rank-based penalties).
            re_evaluate=True,
            elitist=True,
            re_evaluate_parents_first=True,
        )
        self._pop_init = torch.tensor(self.population.values.cpu().detach().tolist())  # pyright: ignore

    def __init_metamodels(self) -> None:
        """Build the initial design of experiments and train the first surrogates."""
        sobol = SobolTrainingPoints(
            dim=self.benchmark.dim,
            batch_size=self.batch_train_set,
            bounds=self.benchmark.bounds,
            seed=self.problem.seed,
        )
        with torch.no_grad():
            # The first Sobol points train the surrogates. Completed with the next points of the
            # sequence, they also form the initial population of the genetic algorithm.
            train_x = sobol.get_initial_points()
            additional_points = sobol.get_additional_points(self.n_ind - self.batch_train_set)
            init_pop = torch.vstack([train_x, additional_points])
            # Move from the design space to the augmented space: d -> w = {d, mean of z}.
            train_x = self.benchmark.get_W(N_SAMPLES=1, design=train_x)
            # All the surrogates start from the same design of experiments.
            doe = [train_x.clone() for _ in range(self.ng)]
        metamodels = ListGPRBoTorch(
            covar_module=self.covar_module,
            meta_bounds=self.benchmark.aug_bounds,
            ng=self.ng,
        )
        with torch.no_grad():
            train_y = [gk.fn(doe[k]) for k, gk in enumerate(self.benchmark.hard)]
        metamodels.train(DoE=doe, train_y=train_y)
        self._problem.metamodels = metamodels
        self._problem.init_pop = init_pop
        self._doe = doe

    def _set_getters(self) -> None:
        """Register the functions adding the entries of each analysis to the status.

        The functions are called by EvoTorch after each generation; the entries they return are
        merged into the status of the searcher and recorded by the logger.

        Raises:
            ValueError: If the analysis is not supported.
        """

        def get_individuals() -> dict[str, list[Tensor]]:
            """Return the individuals of the current population."""
            return {"individuals": self.population.values.cpu().detach().clone().tolist()}  # pyright: ignore

        def get_rho_alpha() -> dict[str, bool]:
            """Return whether the isocost region is well explored and holds an admissible point."""
            mu_in_isoregion, var_in_isoregion = self.problem.evaluate_hard(self._mc_global)
            rho_alpha = self._get_rho_alpha(mu_in_isoregion, var_in_isoregion)
            is_region_well_explored = bool(rho_alpha >= self.rho_thr)
            any_admissible_point = bool((mu_in_isoregion > 0).all(dim=1).any())
            return {
                "rho_alpha": is_region_well_explored,
                "admissibility": any_admissible_point,
            }

        def get_Pf_Beta_feasibility() -> dict[str, bool | float]:
            """Return the reliability of the best individual.

            The failure probabilities are estimated by quasi-Monte Carlo and converted into
            reliability indices. For the surrogate-based analyses, the maximum relative error
            ``eps_max`` of the ESC stopping criterion is returned as well.
            """
            best_pop_d: Tensor = self.population.values[0].detach().clone()  # pyright: ignore
            best_pop_w = self.problem.benchmark.get_W(
                design=best_pop_d.unsqueeze(0),
                N_SAMPLES=self.problem.batch_qmc,
            )
            hard_w, var_w = self.problem.evaluate_hard(best_pop_w)
            Nf = (hard_w <= 0).sum(dim=1)
            Pf = Nf / self.problem.batch_qmc
            Beta = -Normal(0, 1).icdf(Pf).clamp(min=-BETA_MAX, max=BETA_MAX)
            if self.problem.analysis == "rbdo":
                return {
                    "Pf": Pf.tolist()[0],
                    "Beta": Beta.tolist()[0],
                    "feasibility": bool((Pf <= self.problem.Pf_thr).all()),
                }
            # `esc` expects tensors of shape (n_samples, n_hard): with a single hard constraint,
            # `squeeze` would also remove the constraint dimension.
            if hard_w.shape[-1] == 1:
                _, _, _, _, eps_max = sc.esc(
                    hard_w.reshape(-1, 1),
                    var_w.reshape(-1, 1),
                    self.eps_max_thr,
                )
            else:
                _, _, _, _, eps_max = sc.esc(hard_w.squeeze(), var_w.squeeze(), self.eps_max_thr)
            return {
                "Pf": Pf.tolist()[0],
                "Beta": Beta.tolist()[0],
                "feasibility": bool((Pf <= self.problem.Pf_thr).all()),
                "eps_max": float(eps_max),
            }

        def get_stdev_pop_evals() -> dict[str, float]:
            """Return the standard deviation of the fitness over the population."""
            return {"pop_evals_stdev": float(torch.std(self.population.evals[:, 0]))}

        def get_cost_pop_best() -> dict[str, float]:
            """Return the cost of the best individual."""
            pop = (
                self.population.values[0, int(self.benchmark.dim / 2) :]  # pyright: ignore
                if self.problem.benchmark.nominal_tolerance_design
                else self.population.values[0]  # pyright: ignore
            )
            return {"cost": float(self.benchmark.cost.fn(pop).item())}

        def get_alpha() -> dict[str, float]:
            """Return the level of the current isocost region."""
            return {"alpha": self.problem.alpha}

        def get_g_calls() -> dict[str, Tensor]:
            """Return the number of evaluations of each hard constraint."""
            return {"g_calls": self.g_calls}

        def get_doe() -> dict[str, list[list[float]]]:
            """Return the design of experiments of each surrogate."""
            return {"doe": [doe.tolist() for doe in self._doe]}

        def get_pop_init_alpha() -> dict[str, Tensor]:
            """Return the initial population of the current isocost region."""
            return {"pop_init_alpha": self._pop_init}

        match self.problem.analysis:
            case "ddo":
                self.after_step_hook.append(get_individuals)
                self.after_step_hook.append(get_stdev_pop_evals)
                self.after_step_hook.append(get_cost_pop_best)
            case "rbdo":
                self.after_step_hook.append(get_individuals)
                self.after_step_hook.append(get_Pf_Beta_feasibility)
                self.after_step_hook.append(get_stdev_pop_evals)
                self.after_step_hook.append(get_cost_pop_best)
            case "zakrbdo":
                self.after_step_hook.append(get_individuals)
                self.after_step_hook.append(get_Pf_Beta_feasibility)
                self.after_step_hook.append(get_stdev_pop_evals)
                self.after_step_hook.append(get_g_calls)
                self.after_step_hook.append(get_doe)
                self.after_step_hook.append(get_cost_pop_best)
            case "isorakrbdo":
                self.after_step_hook.append(get_individuals)
                self.after_step_hook.append(get_Pf_Beta_feasibility)
                self.after_step_hook.append(get_stdev_pop_evals)
                self.after_step_hook.append(get_cost_pop_best)
                self.after_step_hook.append(get_g_calls)
                self.after_step_hook.append(get_doe)
                self.after_step_hook.append(get_alpha)
                self.after_step_hook.append(get_pop_init_alpha)
                self.after_step_hook.append(get_rho_alpha)
            case _:
                msg = f"""Analysis not supported: {self.problem.analysis} -
                 Supported analysis: ['ddo', 'rbdo', 'zakrbdo', 'isorakrbdo']"""
                raise ValueError(msg)

    def _update_stdev_mutation(self, shrink: float = 1.0) -> None:
        """Shrink the standard deviation of the Gaussian mutation.

        The current standard deviation is multiplied by ``1 - shrink * step_count / n_gen``.
        The factor applies to the value left by the previous generations, so the shrinkage
        compounds along the search.

        Args:
            shrink: Shrinkage rate.
        """
        self._operators[1]._stdev = self._operators[1]._stdev * (
            1 - shrink * self.step_count / self.n_gen
        )

    def _reset_stdev_mutation(self, *, stdev: Tensor) -> None:
        """Set the standard deviation of the Gaussian mutation to ``stdev``."""
        self._operators[1]._stdev = stdev

    def _final_step(self) -> None:
        """Store the initial population in the problem and call the end-of-run hooks."""
        if self._problem.init_pop is None:
            self._problem.init_pop = self._pop_init
        if len(self._end_of_run_hook) >= 1:
            self._end_of_run_hook(dict(self.status))

    def _init_progressbar(self) -> TaskID:
        """Add the task of the run to the progress bar and return its identifier."""
        task_msg = "[bold light_coral]🛠️  Running Optimization"
        task_msg += (
            f" - [blue]isoline[/blue] [green]{self.problem.alpha:.3f}[/green]: \n"
            if self.problem.analysis == "isorakrbdo"
            else ": \n"
        )
        return utils.progressbar.add_task(task_msg, total=self.n_gen)

    def _set_MC_pop_global(self) -> tuple[Tensor, Tensor]:
        """Sample the global candidate points for the enrichment of the surrogates.

        The candidates are drawn from a scrambled Sobol sequence over the design domain.

        Returns:
            The candidates in the augmented space, each design being completed to its
            one-sample point ``get_W(N_SAMPLES=1)``, and the corresponding designs.
        """
        with torch.no_grad():
            sobol_engine = torch.quasirandom.SobolEngine(
                dimension=self.benchmark.dim,
                scramble=True,
                seed=self.problem.seed,
            )
            sobol_samples = sobol_engine.draw(self.size_candidate_global)
            lower_bounds = self.benchmark.bounds.t()[0]
            upper_bounds = self.benchmark.bounds.t()[1]
            mc_pop_design = lower_bounds + (upper_bounds - lower_bounds) * sobol_samples
            soft_evals = self.benchmark.evaluate_soft(mc_pop_design)
            if soft_evals is not None:
                soft_violations = (soft_evals > 0).any(dim=1)
                mc_pop = mc_pop_design[~soft_violations]
            mc_pop = self.benchmark.get_W(N_SAMPLES=1, design=mc_pop_design)
            return mc_pop, mc_pop_design

    def _local_metamodels_update(self) -> str:
        """Enrich the surrogates around the current best design (local enrichment).

        The ESC stopping criterion is evaluated on the quasi-Monte Carlo population of the best
        design, with a threshold three times looser than the one of the AK-MCS refinement. When
        it is not met, the sample with the smallest U of the active constraint is added to the
        design of experiments of this constraint, unless the DBSCAN filter finds it too close to
        the existing training points.

        Returns:
            The message describing the outcome, for the console.
        """
        pop_best = self.population.values[0].clone()  # pyright: ignore
        w = self.benchmark.get_W(
            design=pop_best.unsqueeze(0),
            N_SAMPLES=self.problem.batch_qmc,
        ).squeeze()
        mean, var = self.problem.evaluate_hard(w)
        stopping_criteria_met, pos, activated_gk, _, eps_max = sc.esc(
            mean,
            var,
            3 * self.eps_max_thr,
        )
        msg = ""
        if not stopping_criteria_met:
            x_new = w[pos]
            # DBSCAN filter: the candidate is evaluated only if it is not clustered with the
            # training points already available.
            if not_in_cluster(
                DoE_physical=self.doe[activated_gk],
                x_new_physical=x_new,
                design=pop_best.unsqueeze(0),
                benchmark=self.benchmark,
                min_samples=self.min_samples,
                eps=self.eps,
            ):
                self._doe[activated_gk] = torch.vstack([self.doe[activated_gk], x_new])
                self.problem.metamodels.hard[activated_gk].fit(
                    train_x=self.doe[activated_gk],
                    train_y=self.benchmark.hard[activated_gk].fn(self.doe[activated_gk]),
                )
                msg += "\n📢[yellow bold] LOCALLY ~ NO CLUSTER DETECTED[/]"
                msg += f"\n   eps_max_{activated_gk + 1} = {float(eps_max):.4f} > {3 * self.eps_max_thr}"
                msg += f"\n[yellow]   New point added to DoE {activated_gk + 1}![/]"
            else:
                msg += "\n✅[green bold] LOCALLY ~ CLUSTER DETECTED![/]"
                msg += f"\n   eps_max_{activated_gk + 1} = {float(eps_max):.4f} > {3 * self.eps_max_thr}"
                msg += f"\n[green]   New point not added to DoE {activated_gk + 1}![/]"
        else:
            msg += f"\n✅[green bold] LOCALLY ~ eps_max_{activated_gk + 1} = {float(eps_max):.4f} <= {3 * self.eps_max_thr} found![/]"
        msg += "\n"
        return msg

    def _global_metamodels_update(self) -> str:
        """Enrich the surrogates over the global candidate points (global enrichment).

        The learning function A is evaluated at the global candidates for each hard constraint;
        the constraint reaching the largest value is the active one. The rho indicator is the
        fraction of the candidates whose sign is reliably predicted (U above the threshold) for
        the active constraint. While rho does not exceed ``rho_thr``, one candidate is selected
        (the maximizer of A for ZAK-RBDO, the minimizer of U for ISORAK-RBDO) and added to the
        design of experiments of the active constraint, unless it violates a soft constraint or
        the DBSCAN filter finds it too close to the existing training points.

        Returns:
            The message describing the outcome, for the console.

        Raises:
            ValueError: If the analysis is neither ZAK-RBDO nor ISORAK-RBDO.
        """
        mean, var = self.problem.evaluate_hard(self.mc_global)
        U = mean.abs().cpu() / var.sqrt().cpu()
        # Learning function A, one column per hard constraint: the predicted probability of
        # failure Pm times the desirability Ps (cost benefit times the probability of
        # feasibility for all the constraints). For a given constraint, this is the cost
        # benefit, times the Bernoulli variance of its predicted failure indicator, times the
        # probability of feasibility for the other constraints.
        Pm = self.Pr_feas(-mean, var)
        Pfeas = self.Pr_feas(mean, var)
        Pc = self.Pr_cost(self.mc_global)
        Ps = self.desiderability(Pc, Pfeas)
        A = Pm * Ps.reshape(-1, 1)
        activated_gk = int(A.max(dim=0).values.argmax().detach().clone())
        rho = (U[..., activated_gk] > U_THR).sum() / U.shape[0]
        if rho <= self.rho_thr:
            match self.problem.analysis:
                case "zakrbdo":
                    best_idx = A[..., activated_gk].argmax()
                case "isorakrbdo":
                    best_idx = U[..., activated_gk].argmin()
                case _:
                    err_str = "Invalid analysis type"
                    raise ValueError(err_str)
            x_new = self.mc_global[best_idx]
            x_new_soft = self._mc_global_design[best_idx]
            del mean, var, U, A, Pm, Pfeas, Pc, Ps
            check_soft = self.benchmark.evaluate_soft(x_new_soft)
            if check_soft is not None and (check_soft > 0).any():
                return "✅[green bold] GLOBALLY ~ x_new not in soft admissible space"
            design_new = self._mc_global_design[best_idx]
            # DBSCAN filter: the candidate is evaluated only if it is not clustered with the
            # training points already available.
            if not_in_cluster(
                DoE_physical=self.doe[activated_gk],
                x_new_physical=x_new,
                design=design_new.unsqueeze(0),
                benchmark=self.benchmark,
                min_samples=self.min_samples,
                eps=self.eps,
            ):
                self._doe[activated_gk] = torch.vstack([self.doe[activated_gk], x_new])
                self.problem.metamodels.hard[activated_gk].fit(
                    train_x=self.doe[activated_gk],
                    train_y=self.benchmark.hard[activated_gk].fn(self.doe[activated_gk]),
                )
                msg = "📢[yellow bold] GLOBALLY ~ NO CLUSTER DETECTED[/]"
                msg += f"\n   rho[U_{activated_gk + 1}] = {rho:.4f} <= {self.rho_thr}"
                msg += f"\n[yellow bold]   New point added to DoE {activated_gk + 1}![/]"
            else:
                msg = "✅[green bold] GLOBALLY ~ CLUSTER DETECTED[/]"
                msg += f"\n   rho[U_{activated_gk + 1}] = {rho:.4f} <= {self.rho_thr}"
                msg += f"\n[green]   New point not added to DoE {activated_gk + 1}![/]"
            # Added or not, the candidate is removed so that it is not selected again.
            mask = ~torch.all(torch.eq(self.mc_global, x_new), dim=1)
            self._mc_global = self.mc_global[mask]
            return msg
        return f"✅[green bold] GLOBALLY ~ rho[U_{activated_gk + 1}] = {rho:.4f} >= {self.rho_thr} found!\n[/]"

    def _local_akmcs_update(self, path: utils.Path) -> bool:
        """Refine the surrogates at the best design with AK-MCS.

        A Monte Carlo population is sampled at the best design. Until the ESC stopping criterion
        is met, the sample with the smallest U of the active constraint is added to the design
        of experiments of this constraint, without DBSCAN filter, and removed from the
        population.

        Args:
            path: Directory of the run, where the console log is saved.

        Returns:
            True, once the stopping criterion is met.
        """
        best_solution = self.status["pop_best"].values
        mc_pop_best = self.benchmark.get_W_MCS(
            N_SAMPLES=int(self.size_mc),
            design=best_solution.unsqueeze(0),
        ).squeeze()
        akmcs_log = "[bold red]📢 RUNNING AK-MCS!"
        utils.update_live(utils.progressbar, [{"content": akmcs_log}])
        utils.log_live_state(path, "analysis.log")
        stopping_criteria_met = False
        while not stopping_criteria_met:
            mean, var = self.problem.evaluate_hard(mc_pop_best.to(best_solution.device))
            stopping_criteria_met, pos, activated_gk, log_str, _ = sc.esc(
                mean,
                var,
                self.eps_max_thr,
            )
            if not stopping_criteria_met:
                x_new = mc_pop_best[pos]
                self._doe[activated_gk] = torch.vstack([self.doe[activated_gk], x_new])
                self.problem.metamodels.hard[activated_gk].fit(
                    train_x=self.doe[activated_gk],
                    train_y=self.benchmark.hard[activated_gk].fn(self.doe[activated_gk]),
                )
                # Remove the new training point from the Monte Carlo population.
                mask = ~torch.all(
                    torch.eq(mc_pop_best, x_new),
                    dim=1,
                )
                mc_pop_best = mc_pop_best[mask]
                utils.update_live(utils.progressbar, [{"content": log_str, "style": "bold green"}])
                utils.log_live_state(path, "analysis.log")
            else:
                log_str += f" ~ Total g calls DoE: {[int(gc_i) for gc_i in self.g_calls]}\n"
                text_items = [{"content": log_str, "style": "bold red"}]
                utils.update_live(utils.progressbar, text_items)
                utils.log_live_state(path, "analysis.log")
        del mc_pop_best
        utils.clear_gpu_cache()
        return stopping_criteria_met

    def _run_ddo_rbdo(self, logger: PolarsLogger, path: utils.Path) -> None:
        """Run the genetic algorithm for the DDO and RBDO analyses.

        The search stops after ``n_gen`` generations, or as soon as the fitness has stagnated
        over a window of ``2 * window`` generations.

        Args:
            logger: Logger attached to the searcher, read by the convergence check.
            path: Directory of the run, where the console log is saved.

        Raises:
            ValueError: If the analysis is neither DDO nor RBDO.
        """
        task_id = self._init_progressbar()
        self._set_getters()
        for n_iter in range(self.n_gen):
            self.step()
            self._update_stdev_mutation()
            has_converged, msg, converg_log = check_convergence(
                logger,
                n_iter,
                2 * self.window,
                self.improvement_max,
            )
            if not has_converged:
                utils.progressbar.advance(task_id)
                if self.problem.analysis == "ddo":
                    evo_torch_log = log_ddo(self.status)
                elif self.problem.analysis == "rbdo":
                    evo_torch_log = log_rbdo(self.status)
                else:
                    msg = f"Unknown analysis type: {self.problem.analysis}"
                    raise ValueError(msg)
                text_items = [
                    {"content": "[cyan]".join(evo_torch_log)},
                    {"content": "".join(converg_log), "style": "magenta"},
                ]
                utils.update_live(utils.progressbar, text_items)
                utils.log_live_state(path, "analysis.log")
            else:
                utils.update_live(
                    utils.progressbar,
                    [{"content": f"[magenta bold]🛑 {msg}\n"}],
                )
                utils.log_live_state(path, "analysis.log")
                break
        self._final_step()

    def _run_zakrbdo(self, logger: PolarsLogger, path: utils.Path) -> None:
        """Run the ZAK-RBDO analysis.

        The search goes through three stages:

        1. Enrichment. Before each generation, the surrogates are enriched locally around the
           best design (U learning function, ESC stopping criterion) and globally over the
           design domain (learning function A, rho indicator), until the fitness stagnates over
           a window of ``window`` generations.
        2. AK-MCS refinement. At stagnation, the surrogates are refined at the best design
           until the ESC stopping criterion is met. The mutation step is then reduced and the
           search goes on for a waiting period of ``2 * window`` generations.
        3. Convergence. After the waiting period, the search stops as soon as the fitness has
           stagnated over a window of ``2 * window`` generations, or after ``n_gen``
           generations.

        Args:
            logger: Logger attached to the searcher, read by the convergence checks.
            path: Directory of the run, where the console log is saved.
        """
        self._mc_global, self._mc_global_design = self._set_MC_pop_global()

        task_id = self._init_progressbar()

        self._set_getters()
        stagnation, stop_learn, has_converged, wait_time = False, False, False, 0

        for n_iter in range(self.n_gen):
            text_items = []
            # Stage 1: local and global enrichment, until the search stagnates. The stagnation
            # check is five times looser than the final convergence check.
            if not stagnation and n_iter > 0:
                local_log = self._local_metamodels_update()
                global_log = self._global_metamodels_update()
                stagnation, msg, converg_log = check_convergence(
                    logger=logger,
                    n_iter=n_iter,
                    window=self.window,
                    improvement_max=5 * self.improvement_max,
                )
                text_items += [{"content": "".join(local_log)}]
                text_items += [{"content": "".join(global_log)}]
                text_items += [{"content": "".join(converg_log), "style": "magenta"}]
                if stagnation:
                    utils.update_live(utils.progressbar, [{"content": f"[magenta bold]🛑 {msg}\n"}])
                    utils.log_live_state(path, "analysis.log")

            # Stage 2: AK-MCS refinement at the best design. The search then restarts with a
            # smaller mutation step, and convergence is not checked before the fitness history
            # has been renewed with the refined surrogates.
            if stagnation and not stop_learn:
                stop_learn = self._local_akmcs_update(path)
                if stop_learn:
                    self._reset_stdev_mutation(stdev=0.25 * self.stdev)
                    wait_time = n_iter + round(2 * self.window)
                    msg = (
                        f"[blue bold]🕒 Entering waiting period for {2 * self.window} iterations\n"
                    )
                    utils.update_live(utils.progressbar, [{"content": msg}])
                    utils.log_live_state(path, "analysis.log")

            doe_log = [f"DoE{k + 1}: {int(self.g_calls[k])} " for k in range(len(self.doe))]
            # Send the updated surrogates to the remote copies of the problem.
            if self.num_actors > 1:
                self._problem.all_remote_problems()._set_hard(self.problem.metamodels)
            self.step()
            self._update_stdev_mutation()

            # Stage 3: final convergence check, once the waiting period is over.
            if stagnation and stop_learn:
                if n_iter <= wait_time:
                    converg_log = f"🕒 WAITING: {wait_time - n_iter} iterations before checking convergence..."
                else:
                    has_converged, msg, converg_log = check_convergence(
                        logger=logger,
                        n_iter=n_iter,
                        window=2 * self.window,
                        improvement_max=self.improvement_max,
                    )
                    if has_converged:
                        utils.update_live(
                            utils.progressbar,
                            [{"content": f"[magenta bold]🛑 {msg}\n"}],
                        )
                        utils.log_live_state(path, "analysis.log")
                        break
                text_items += [{"content": "[magenta]".join(converg_log)}]

            utils.progressbar.advance(task_id)
            evo_torch_log = log_zakrbdo(self.status, doe_log)

            text_items[:0] = [{"content": "[cyan]".join(evo_torch_log)}]

            utils.update_live(utils.progressbar, text_items)
            utils.log_live_state(path, "analysis.log")

            # Free the GPU cache every 5 generations.
            if n_iter % 5 == 0:
                utils.clear_gpu_cache()

        self._final_step()

        # Bring the surrogates back to the CPU at the end of the run.
        for meta in self._problem.metamodels.hard:
            if meta.device.type != "cpu":  # pyright: ignore
                meta.device = torch.device("cpu")
                meta.model.to(meta.device)
                meta.model.train().to(meta.device)

    def _get_rho_alpha(self, mu_in_isoregion, var_in_isoregion) -> bool:
        """Tell whether the current isocost region is well explored.

        The rho indicator is computed as in `_global_metamodels_update`, on the global
        candidates of the isocost region.

        Args:
            mu_in_isoregion: Surrogate means at the candidates of the region, shape
                ``(n, n_hard)``.
            var_in_isoregion: Surrogate variances at the candidates of the region, shape
                ``(n, n_hard)``.

        Returns:
            Whether the rho indicator reaches the threshold ``rho_thr``.
        """
        U_in_isoregion = mu_in_isoregion.abs().cpu() / var_in_isoregion.sqrt().cpu()
        Pm = self.Pr_feas(-mu_in_isoregion, var_in_isoregion)
        Pfeas = self.Pr_feas(mu_in_isoregion, var_in_isoregion)
        Pc = self.Pr_cost(self.mc_global)
        Ps = self.desiderability(Pc, Pfeas)
        A = Pm * Ps.reshape(-1, 1)
        activated_gk = int(A.max(dim=0).values.argmax().detach().clone())
        rho_alpha = (U_in_isoregion[..., activated_gk] > U_THR).sum() / U_in_isoregion.shape[0]
        return rho_alpha >= self.rho_thr

    def _update_alpha_ub(self, mc_global) -> None:
        """Update the upper bound of the isocost level from the candidates outside the region.

        Among the global candidates outside the current isocost region, the ones predicted
        admissible for all the hard constraints with a reliable sign (U above the threshold)
        are retained. The upper bound ``alpha_ub`` is the first tabulated isocost level whose
        region contains the cheapest of them.

        Args:
            mc_global: All the global candidates, in the augmented space.
        """
        c_norm = self.benchmark.cost.fn_norm(mc_global)
        mask_not_isoline = c_norm > self.benchmark.cost.isolevels[self.problem.alpha]
        S_not_in_isoregion = mc_global[mask_not_isoline]
        mu_not_isoregion, var_not_isoregion = self.problem.evaluate_hard(S_not_in_isoregion)
        U_not_isoregion = mu_not_isoregion.abs() / var_not_isoregion.sqrt()
        idx_admissible = (mu_not_isoregion > 0).all(dim=1)
        S_not_in_isoregion_feas = S_not_in_isoregion[idx_admissible]
        S_F = S_not_in_isoregion_feas[(U_not_isoregion[idx_admissible] > U_THR).all(dim=1)]
        c_alpha_ub = self.benchmark.cost.fn_norm(S_F).min()
        alpha_ub = next(
            key for key, value in self.benchmark.cost.isolevels.items() if value >= c_alpha_ub
        )
        self.benchmark.cost._alpha_ub = alpha_ub

    def _adaptive_alpha_expansion(self) -> None:
        """Update the upper bound of the isocost level from the reliability violation.

        The isocost level is increased by a step between 0.001 and 0.05, proportional to the
        largest normalized excess of the failure probabilities of the best design over their
        thresholds. The upper bound ``alpha_ub`` is 1.25 times the increased level.
        """
        zita = torch.tensor(self.status["Pf"]) - self.problem.Pf_thr
        zita_norm = zita.clamp(min=0) / (1 - self.problem.Pf_thr)
        max_violation = zita_norm.max()
        D_alpha_min, D_alpha_max = 0.001, 0.05
        D_alpha = D_alpha_min + (D_alpha_max - D_alpha_min) * max_violation
        alpha_ub = 1.25 * (D_alpha + self.benchmark.cost.alpha)
        self.benchmark.cost._alpha_ub = round(alpha_ub.item(), 3)

    def _run_isorakrbdo(self, logger: PolarsLogger, path: utils.Path, *, stop_alpha: bool) -> bool:
        """Run the ISORAK-RBDO analysis inside the current isocost region.

        The search follows the stages of `_run_zakrbdo`, with the global candidates restricted
        to the isocost region of level ``alpha``. It returns early, after updating the upper
        bound ``alpha_ub`` of the isocost level, when the region has to be expanded:

        * the region is well explored (rho indicator) and holds no admissible candidate;
        * the fitness stagnates while the region holds no admissible candidate;
        * the region holds admissible candidates, but the best design has violated the
          reliability constraints at 10 generations.

        Args:
            logger: Logger attached to the searcher, read by the convergence checks.
            path: Directory of the run, where the console log is saved.
            stop_alpha: Value returned when the search ends without having converged.

        Returns:
            True if the search has converged inside the region, else ``stop_alpha``.
        """
        # Global candidates: the Sobol points of the design domain that lie inside the isocost
        # region. The whole set is kept to update the upper bound of the isocost level.
        mc_global, mc_global_design = self._set_MC_pop_global()
        indicator = self.benchmark.cost.pop_in_isolevels(mc_global, self.benchmark.cost.alpha)
        self._mc_global = mc_global[indicator].detach().clone()
        self._mc_global_design = mc_global_design[indicator].detach().clone()

        task_id = self._init_progressbar()

        self._set_getters()
        stagnation, stop_learn, has_converged, wait_time = False, False, False, 0
        count_unfeasibility = 0

        for n_iter in range(self.n_gen):
            text_items = []

            # A candidate is admissible when the surrogates predict that it satisfies all the
            # hard constraints. The region is well explored when its rho indicator reaches the
            # threshold.
            mu_in_isoregion, var_in_isoregion = self.problem.evaluate_hard(self._mc_global)
            is_any_admissible = (mu_in_isoregion > 0).all(dim=1).any()
            is_region_well_explored = self._get_rho_alpha(mu_in_isoregion, var_in_isoregion)

            # Well explored region without any admissible candidate: expand the region.
            if not is_any_admissible and is_region_well_explored:
                self._update_alpha_ub(mc_global)
                text_items += [{"content": "".join("Skip alpha"), "style": "magenta"}]
                utils.update_live(utils.progressbar, text_items)
                utils.log_live_state(path, "analysis.log")
                return stop_alpha

            # Stage 1: enrichment, until the search stagnates. The local enrichment needs an
            # admissible candidate in the region.
            if not stagnation and n_iter > 0:
                if not is_region_well_explored:
                    if is_any_admissible:
                        local_log = self._local_metamodels_update()
                        text_items += [{"content": "".join(local_log)}]
                    global_log = self._global_metamodels_update()
                    text_items += [{"content": "".join(global_log)}]
                stagnation, msg, converg_log = check_convergence(
                    logger=logger,
                    n_iter=n_iter,
                    window=self.window,
                    improvement_max=5 * self.improvement_max,
                )
                text_items += [{"content": "".join(converg_log), "style": "magenta"}]
                if stagnation:
                    utils.update_live(utils.progressbar, [{"content": f"[magenta bold]🛑 {msg}\n"}])
                    utils.log_live_state(path, "analysis.log")

            # Stage 2: AK-MCS refinement at the best design if the region holds an admissible
            # candidate, expansion of the region otherwise.
            if stagnation and not stop_learn:
                if is_any_admissible:
                    stop_learn = self._local_akmcs_update(path)
                    if stop_learn:
                        self._reset_stdev_mutation(stdev=0.25 * self.stdev)
                        wait_time = n_iter + round(2 * self.window)
                        msg = f"[blue bold]🕒 Entering waiting period for {2 * self.window} iterations\n"
                        utils.update_live(utils.progressbar, [{"content": msg}])
                        utils.log_live_state(path, "analysis.log")
                else:
                    self._update_alpha_ub(mc_global)
                    text_items += [{"content": "".join("Skip alpha"), "style": "magenta"}]
                    utils.update_live(utils.progressbar, text_items)
                    utils.log_live_state(path, "analysis.log")
                    return stop_alpha

            doe_log = [f"DoE{k + 1}: {int(self.g_calls[k])} " for k in range(len(self.doe))]
            # Send the updated surrogates and the isocost level to the remote copies of the
            # problem.
            if self.num_actors > 1:
                self._problem.all_remote_problems()._set_hard(self.problem.metamodels)
                self._problem.all_remote_problems()._set_alpha(self.problem.alpha)
            self.step()
            self._update_stdev_mutation()

            # The best design keeps violating the reliability constraints although the region
            # holds admissible candidates: expand the region according to the violation.
            if is_any_admissible and not self.status["feasibility"]:
                count_unfeasibility += 1
                if count_unfeasibility >= 10:
                    self._adaptive_alpha_expansion()
                    text_items += [
                        {"content": "\n[magenta bold]: Adaptive alpha Expansion"},
                    ]
                    utils.update_live(utils.progressbar, text_items)
                    utils.log_live_state(path, "analysis.log")
                    return stop_alpha

            # Stage 3: final convergence check, once the waiting period is over.
            if stagnation and stop_learn:
                if n_iter <= wait_time:
                    converg_log = f"🕒 WAITING: {wait_time - n_iter} iterations before checking convergence..."
                else:
                    has_converged, msg, converg_log = check_convergence(
                        logger=logger,
                        n_iter=n_iter,
                        window=2 * self.window,
                        improvement_max=self.improvement_max,
                    )
                    if has_converged:
                        utils.update_live(
                            utils.progressbar,
                            [{"content": f"[magenta bold]🛑 {msg}\n"}],
                        )
                        utils.log_live_state(path, "analysis.log")
                        stop_alpha = True
                        break
                text_items += [{"content": "[magenta]".join(converg_log)}]

            utils.progressbar.advance(task_id)
            evo_torch_log = log_isorakrbdo(self.status, doe_log)

            text_items[:0] = [{"content": "[cyan]".join(evo_torch_log)}]

            utils.update_live(utils.progressbar, text_items)
            utils.log_live_state(path, "analysis.log")

            # Free the GPU cache every 5 generations.
            if n_iter % 5 == 0:
                utils.clear_gpu_cache()

        self._final_step()

        # Bring the surrogates back to the CPU at the end of the run.
        for meta in self._problem.metamodels.hard:
            if meta.device.type != "cpu":  # pyright: ignore
                meta.device = torch.device("cpu")
                meta.model.to(meta.device)
                meta.model.train().to(meta.device)
        return stop_alpha
