"""EvoTorch problem for the DDO, RBDO, ZAK-RBDO and ISORAK-RBDO analyses.

The module defines `OptimizationProblem`, which evaluates the fitness of a population of
designs: the reliability violation of each design, then a tiered penalty that ranks the
constraint violations above the cost.

Throughout the module, the *mean point* of a design is the single point of the augmented space
(design variables and random variables) returned by ``benchmark.get_W(N_SAMPLES=1, design=...)``.
"""

from typing import Any
from collections.abc import Callable

from attrs import field, define
from torch import Tensor
from evotorch import Problem, SolutionBatch
from torch.distributions import Normal
import torch

from .. import utils
from ..benchmarks import BenchmarkBase, register_benchmark
from ..gaussianprocess.gp import ListGPRBoTorch

# Base values of the fitness penalty bands, from the most to the least severe violation.
# Within a band the fitness is `base + base * penalty`, with a rank-based penalty in
# [0.01, 1.01]: the bands cannot overlap, and they all lie above the normalized cost used for
# the designs without violation.
FITNESS_BAND_SOFT = 10000
FITNESS_BAND_HARD = 1000
FITNESS_BAND_RELIABILITY = 100
# Value given to every design before it is assigned to a band or to its cost.
FITNESS_UNASSIGNED = 100000
# Reliability indices are clamped to [-BETA_MAX, BETA_MAX]: a failure probability estimated
# as 0 or 1 would otherwise give an infinite index.
BETA_MAX = 8.0


@define(
    slots=False,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class OptimizationProblem(Problem):
    """Design optimization problem of a benchmark, with or without reliability constraints.

    The problem minimizes the cost ``c(d)`` of the benchmark subject to its deterministic soft
    constraints ``f_j(d) <= 0`` and to constraints on its hard limit-state functions ``g_k``.
    The accepted analyses are:

    * ``"ddo"``: deterministic design optimization, ``g_k > 0`` is required at the mean point
      of each design only;
    * ``"rbdo"``: reliability-based design optimization, the failure probabilities
      ``P(g_k <= 0)`` are estimated by quasi-Monte Carlo on the true limit-state functions and
      must not exceed the thresholds ``Pf_thr``;
    * ``"zakrbdo"``: ZAK-RBDO, same constraints as ``"rbdo"`` with the limit-state functions
      replaced by the Kriging surrogates stored in ``metamodels``;
    * ``"isorakrbdo"``: ISORAK-RBDO, as ``"zakrbdo"`` with the search restricted to the isocost
      region of level ``alpha``.

    Two configuration files are read from the current working directory when the problem is
    created:

    * ``<benchmark_id>.yaml``: parameters of the benchmark, one section per analysis. The
      section named after the analysis in upper case (e.g. ``ZAKRBDO``) is stored in ``config``;
    * ``settings.yaml``: numerical settings (``dtype``, ``device`` and, optionally, the
      EvoTorch parallelization options ``num_actors``, ``subbatch_size`` and
      ``num_gpus_per_actor``), stored in ``settings``.

    Attributes:
        analysis: Name of the analysis, see above.
        benchmark_id: Name under which the benchmark is registered; also the stem of its YAML
            file.
        benchmark: Benchmark instance, created at initialization.
        seed: Seed of the random number generators.
        config: Section of ``<benchmark_id>.yaml`` for the analysis, loaded at initialization.
        settings: Content of ``settings.yaml``, loaded at initialization.
        init_pop: Initial population of the genetic algorithm. When None, the population is
            drawn uniformly within the design bounds.
        metamodels: Kriging surrogates of the hard constraints (ZAK-RBDO and ISORAK-RBDO).
        alpha: Level of the current isocost region (ISORAK-RBDO): the search is restricted to
            the designs whose cost does not exceed the ``alpha``-quantile of the cost over the
            design domain.
        _multiactor: True on the remote copies of the problem used for parallel evaluation,
            which only compute the reliability violation.
    """

    analysis: str
    benchmark_id: str
    benchmark: BenchmarkBase = field(init=False)
    seed: int = field(default=27)
    config: dict[str, Any] = field(default=None)
    settings: dict[str, Any] = field(default=None)
    init_pop: Tensor = field(default=None)
    metamodels: ListGPRBoTorch = field(default=None)
    alpha: float = field(default=1.0)
    _multiactor: bool = field(default=False)

    def increment_alpha(self, isoline: float | Tensor) -> None:
        """Set the level of the isocost region, on the problem and on the benchmark cost.

        Args:
            isoline: New isocost level ``alpha``, rounded to three decimals (the resolution of
                the tabulated isocost levels).
        """
        self.alpha = round(float(isoline), 3)
        self.benchmark.cost._alpha = self.alpha

    def __check_analysis(self) -> None:
        """Raise a ValueError if the name of the analysis is not recognized."""
        # "zakddo" passes this check but is not implemented: `evaluate_hard` and the searcher
        # reject it.
        if self.analysis not in ("ddo", "zakddo", "rbdo", "zakrbdo", "isorakrbdo"):
            msg = """Invalid analysis type.
            Must be one of 'ddo', 'zakddo', 'rbdo', 'zakrbdo', 'isorakrbdo'."""
            raise ValueError(msg)

    def __load_config(self) -> None:
        """Load the two configuration files from the current working directory."""
        self.config = utils.load_yaml(
            path=utils.Path.cwd(),
            filename=f"{self.benchmark_id}.yaml",
        )[self.analysis.upper()]

        self.settings = utils.load_yaml(
            path=utils.Path.cwd(),
            filename="settings.yaml",
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point type set in ``settings.yaml`` (float64 by default)."""
        match self.settings["dtype"]:
            case "float16":
                return torch.float16
            case "float32":
                return torch.float32
            case "float64":
                return torch.float64
            case _:
                return torch.float64

    @property
    def device(self) -> torch.device:
        """Return the device set in ``settings.yaml`` (CPU by default)."""
        match self.settings["device"]:
            case "cuda":
                return torch.device("cuda")
            case "mps":
                return torch.device("mps")
            case _:
                return torch.device("cpu")

    def __attrs_post_init__(self) -> None:
        """Load the configuration, create the benchmark and initialize the EvoTorch problem."""
        self.__check_analysis()
        self.__load_config()
        torch.manual_seed(self.seed)
        torch.set_default_dtype(self.dtype)
        self.benchmark = register_benchmark[self.benchmark_id](
            seed=self.seed,
            dtype=self.dtype,
        )
        self.benchmark.cost._alpha = self.alpha
        super().__init__(
            objective_sense="min",
            bounds=self.benchmark.bounds.t(),
            solution_length=self.benchmark.bounds.t().shape[1],
            store_solution_stats=False,
            device=self.device,
            dtype=self.dtype,
            num_actors=self.settings.get("num_actors", 1),
            subbatch_size=self.settings.get("subbatch_size", None),
            num_gpus_per_actor=self.settings.get("num_gpus_per_actor", None),
        )

    def _fill(self, values: Tensor) -> Tensor:
        """Fill the initial population (EvoTorch hook).

        For ISORAK-RBDO the population is drawn uniformly inside the current isocost region.
        Otherwise it is ``init_pop`` when available, else a uniform sample within the design
        bounds.

        Args:
            values: Empty tensor to fill, of shape ``(n_solutions, solution_length)``.

        Returns:
            The filled tensor.
        """
        if self.analysis == "isorakrbdo":
            # Rejection sampling: draw uniform candidates in the design domain and keep the
            # first ones that fall inside the isocost region.
            mc_pop = (
                torch.distributions.Uniform(
                    low=self.benchmark.bounds[:, 0],
                    high=self.benchmark.bounds[:, 1],
                )
                .sample((int(1e6),))
                .to(dtype=torch.float64)
            )
            # The cost of a [nominal values, tolerances] design depends on the tolerances only.
            mc_pop_cost = (
                mc_pop
                if not self.benchmark.nominal_tolerance_design
                else mc_pop[:, int(self.benchmark.dim / 2) :]
            )
            mask = self.benchmark.cost.pop_in_isolevels(
                pop=mc_pop_cost,
                alpha=self.alpha,
            )
            pop_iso = mc_pop[mask]
            assert pop_iso.shape[0] >= values.shape[0]
            pop_selected = pop_iso[: values.shape[0], :]
            values[:] = pop_selected.clone()
            self.init_pop = values.clone()
            return values
        if self.init_pop is None:
            return self.make_uniform(
                out=values,
                lb=self.initial_lower_bounds,
                ub=self.initial_upper_bounds,
            )
        values[:] = self.init_pop.clone()
        return values

    def _violation_soft(self, design: Tensor) -> Tensor:
        """Return the normalized violation of the soft constraints.

        The violation ``max(f_j, 0)`` of each constraint is divided by its largest value over
        the batch, then the violations are summed over the constraints.

        Args:
            design: Designs to evaluate, shape ``(n, dim)``.

        Returns:
            The violation of each design, shape ``(n, 1)``. It is zero for a design that
            satisfies every soft constraint, and for a benchmark without soft constraints.
        """
        # The soft constraints are evaluated at the mean point, except for the
        # [nominal values, tolerances] designs, where they apply to the design itself.
        if not self.benchmark.nominal_tolerance_design:
            design_w = self.benchmark.get_W(design=design, N_SAMPLES=1).to(design)
        else:
            design_w = design
        SOFT = self.evaluate_soft(design_w)
        if SOFT is not None:
            SOFT = SOFT.clamp(min=0)
            SOFT_NORM = SOFT.abs() / (SOFT.abs().max(dim=0, keepdim=True).values + 1e-10)
            return SOFT_NORM.sum(dim=1, keepdim=True)
        return torch.zeros(
            (design.shape[0], 1),
            device=design.device,
            dtype=design.dtype,
        )

    def _violation_hard(self, design: Tensor) -> Tensor:
        """Return the normalized violation of the hard constraints at the mean point.

        The limit states (true functions or surrogate means, depending on the analysis) are
        evaluated at the mean point of each design. The violation ``max(-g_k, 0)`` of each
        constraint is divided by its largest value over the batch, then the violations are
        summed over the constraints.

        Args:
            design: Designs to evaluate, shape ``(n, dim)``.

        Returns:
            The violation of each design, shape ``(n, 1)``. It is zero for a design whose mean
            point satisfies every hard constraint.
        """
        design_w = self.benchmark.get_W(design=design, N_SAMPLES=1).to(design)
        HARD, _ = self.evaluate_hard(design_w)
        HARD = HARD.clamp(max=0)
        HARD_NORM = HARD.abs() / (HARD.abs().max(dim=0, keepdim=True).values + 1e-10)
        return HARD_NORM.sum(dim=1, keepdim=True)

    def _evaluate_fitness(self, design: Tensor, zita: Tensor) -> Tensor:
        """Compute the tiered-penalty fitness of a batch of designs.

        Each design is assigned to the first of the following cases that applies:

        1. it violates a soft constraint;
        2. its mean point violates a hard constraint;
        3. it violates a reliability constraint (``zita > 0``);
        4. it violates nothing, and its fitness is its normalized cost.

        Cases 1 to 3 are penalty bands of decreasing magnitude, all above the normalized cost.
        Inside a band, the designs are ordered by a rank-based penalty of their violation. For
        ISORAK-RBDO, a further penalty is added to the designs outside the current isocost
        region.

        Args:
            design: Designs to evaluate, shape ``(n, dim)``.
            zita: Reliability violation of each design, i.e. the excess of the failure
                probabilities over their thresholds summed over the hard constraints, shape
                ``(n, 1)``. Empty for ``"ddo"``.

        Returns:
            The fitness to minimize, shape ``(n, 1)``.
        """
        # The cost of a [nominal values, tolerances] design depends on the tolerances only.
        pop = (
            design[..., int(self.benchmark.dim / 2) :]
            if self.benchmark.nominal_tolerance_design
            else design
        )

        cost = self.benchmark.evaluate_cost_norm(pop).reshape(-1, 1).clone()
        fitness = torch.ones_like(cost) * FITNESS_UNASSIGNED

        hard_norm = self._violation_hard(design)
        soft_norm = self._violation_soft(design)
        idx = torch.arange(0, design.shape[0], 1)

        # Band 1: soft-constraint violations.
        is_soft_violated = (soft_norm > 0).squeeze()
        idx_soft_violated = idx[is_soft_violated]
        if idx_soft_violated.numel() > 0:
            penalty = self._linear_rank_penalty(soft_norm[idx_soft_violated])
            fitness[idx_soft_violated] = FITNESS_BAND_SOFT + FITNESS_BAND_SOFT * penalty

        # Band 2: hard-constraint violations at the mean point, soft constraints satisfied.
        is_hard_violated = (hard_norm > 0).squeeze()
        idx_hard_violated = idx[is_hard_violated * ~is_soft_violated]
        if idx_hard_violated.numel() > 0:
            penalty = self._linear_rank_penalty(hard_norm[idx_hard_violated])
            fitness[idx_hard_violated] = FITNESS_BAND_HARD + FITNESS_BAND_HARD * penalty

        # Band 3: reliability violations (zita), soft and hard constraints satisfied.
        is_zita_violated = (zita > 0).squeeze()
        idx_zita_violated = idx[is_zita_violated * ~is_hard_violated * ~is_soft_violated]
        if idx_zita_violated.numel() > 0:
            penalty = self._linear_rank_penalty(zita[idx_zita_violated])
            fitness[idx_zita_violated] = (
                FITNESS_BAND_RELIABILITY + FITNESS_BAND_RELIABILITY * penalty
            )

        # No violation: the fitness is the normalized cost.
        are_all_violated = torch.hstack([soft_norm > 0, hard_norm > 0, zita > 0]).any(dim=1)
        idx_not_all_violated = idx[~are_all_violated]
        fitness[idx_not_all_violated] = cost[idx_not_all_violated]

        # Isocost region (ISORAK-RBDO): the designs outside the region are penalized with a
        # term proportional to the worst fitness of the batch, which ranks them after all the
        # designs inside the region.
        if self.analysis == "isorakrbdo":
            pop = (
                design[..., int(self.benchmark.dim / 2) :]
                if self.benchmark.nominal_tolerance_design
                else design
            )
            I_alpha = self.benchmark.cost.pop_in_isolevels(pop, self.alpha).reshape(-1, 1)
            if not I_alpha.all():
                new_fitness = fitness.clone()
                new_fitness[~I_alpha] = new_fitness[~I_alpha] + 5 + 5 * fitness.max().item()
                fitness = new_fitness
        return fitness

    @staticmethod
    def _linear_rank_penalty(raw_violation: Tensor) -> Tensor:
        """Convert violations into a rank-based penalty between 0.01 and 1.01.

        The squared rank of each violation (0 for the smallest one) is divided by the largest
        squared rank, and 0.01 is added so that every penalized design stays strictly above the
        base value of its band. Using ranks rather than raw violations makes the penalty
        insensitive to the scale of the violations.

        Args:
            raw_violation: Violations of the designs of one band, shape ``(m, 1)``.

        Returns:
            The penalties, shape ``(m, 1)``. A single violation gets the penalty 1.01.
        """
        if raw_violation.numel() == 1:
            return torch.ones_like(raw_violation) + 0.01
        ranks = raw_violation.argsort(dim=0).argsort(dim=0).float().pow(2).to(raw_violation.dtype)
        max_ranks = ranks.max()
        if max_ranks == 0:
            return torch.zeros_like(ranks)
        return ranks / max_ranks + 0.01

    def _evaluate_batch(self, batch: SolutionBatch) -> None:
        """Evaluate the reliability violation of a batch of designs (EvoTorch hook).

        The failure probabilities ``Pf`` are estimated by quasi-Monte Carlo, with ``batch_qmc``
        samples per design, for the designs selected by `get_index_admissible`; the other
        designs keep ``Pf = 1``. The evaluation stored in the batch is the excess of ``Pf`` over
        the thresholds ``Pf_thr``, summed over the hard constraints. `_evaluate_all` then turns
        it into the fitness.

        Args:
            batch: The batch to evaluate.
        """
        design: Tensor = batch.values  # pyright: ignore

        # Start from the worst case Pf = 1, kept by the designs that are not sampled below.
        zita = torch.ones((design.shape[0], self.Pf_thr.shape[0])).to(
            design.device,
        ) - self.Pf_thr.to(design.device)

        # Constraints at the mean point: hard [n_batch, n_hard], soft [n_batch, n_soft] or None.
        design_w = self.benchmark.get_W(design=design, N_SAMPLES=1).to(design.device)
        hard, _ = self.evaluate_hard(design_w)
        soft = self.benchmark.evaluate_soft(design)
        index_admissible = self.get_index_admissible(hard, soft)

        if index_admissible.any():
            w_pop = self.benchmark.get_W(
                design=design[index_admissible],
                N_SAMPLES=self.batch_qmc,
            ).to(design.device)
            hard_w, _ = self.evaluate_hard(w_pop)
            # Fraction of failed samples per design and per hard constraint.
            Pf = (hard_w <= 0).sum(dim=1) / self.batch_qmc
            zita[index_admissible] = Pf - self.Pf_thr.to(design.device)

        batch.set_evals(zita.clamp(min=0).sum(dim=1, keepdim=True))

    def _evaluate_all(self, batch: "SolutionBatch") -> None:
        """Evaluate the fitness of all the solutions of the batch (EvoTorch hook).

        For the reliability-based analyses, the reliability violation is first computed by
        `_evaluate_batch`, in this process or on the remote actors, with the same dispatch as
        the EvoTorch base class. It is then converted into the tiered-penalty fitness in the
        main process, on the whole batch at once, because the penalties are normalized and
        ranked over the batch. For ``"ddo"`` the fitness is computed directly.

        Args:
            batch: The batch to evaluate.
        """
        if "ddo" not in self.analysis:
            if self._actors is None:
                fitness_device = self._device_of_fitness_function()
                if fitness_device is None:
                    self._evaluate_batch(batch)
                else:
                    original_device = batch.device
                    moved_batch = batch.to(fitness_device)
                    self._evaluate_batch(moved_batch)
                    batch.set_evals(moved_batch.evals.to(original_device))
            else:
                # Flag the remote copies of the problem, so that they return the reliability
                # violation without converting it into the fitness.
                for actor in self._actors:
                    actor.set.remote("_multiactor", True)  # noqa: FBT003
                if self._num_subbatches is not None:
                    pieces = batch.split(self._num_subbatches)
                elif self._subbatch_size is not None:
                    pieces = batch.split(max_size=self._subbatch_size)
                else:
                    pieces = batch.split(len(self._actors))
                mapresult = self._actor_pool.map_unordered(
                    lambda a, v: a.evaluate_batch_piece.remote(v[0], v[1]),
                    list(enumerate(pieces)),
                )
                for i, evals in mapresult:
                    row_begin, row_end = pieces.indices_of(i)
                    batch._evdata[row_begin:row_end, :] = evals
                batch.set_evals(batch._evdata)
            if not self._multiactor:
                _zita_sum = batch.evals.clone()
                fitness = self._evaluate_fitness(batch.values, _zita_sum)  # pyright: ignore
                batch.set_evals(fitness)
        else:
            fitness = self._evaluate_fitness(batch.values, torch.empty([0]))  # pyright: ignore
            batch.set_evals(fitness)

    @property
    def evaluate_soft(self) -> Callable[[Tensor], Tensor | None]:
        """Return the function evaluating the soft constraints of the benchmark."""
        return self.benchmark.evaluate_soft

    @property
    def evaluate_hard(self) -> Callable[[Tensor], list[Tensor]]:
        """Return the function evaluating the hard constraints.

        Returns:
            The true limit-state functions of the benchmark for ``"ddo"`` and ``"rbdo"``, the
            Kriging surrogates for ``"zakrbdo"`` and ``"isorakrbdo"``. Both return the pair
            ``[values, variances]``; the variances are empty for the true functions.
        """
        match self.analysis:
            case "ddo" | "rbdo":
                return self.benchmark.evaluate_hard
            case "zakrbdo" | "isorakrbdo":
                return self.metamodels.evaluate_surrogate
            case _:
                msg = """
                Invalid analysis type.
                Must be one of 'ddo', 'zakddo','rbdo', 'zakrbdo', 'isorakrbdo'.
                """
                raise ValueError(msg)

    @property
    def ng(self) -> int:
        """Return the number of hard constraints of the benchmark."""
        return len(self.benchmark.hard)

    @property
    def batch_qmc(self) -> int:
        """Return the number of quasi-Monte Carlo samples per design used to estimate Pf."""
        return int(self.config["rbdo"]["batch_qmc"])

    @property
    def size_mc(self) -> int:
        """Return the size of the Monte Carlo populations (AK-MCS refinement, validation)."""
        return int(self.config["rbdo"]["size_mc"])

    @property
    def Pf_thr(self) -> Tensor:
        """Return the failure probability threshold of each hard constraint."""
        return torch.tensor(self.config["rbdo"]["Pf_thr"])

    @property
    def CoV_thr(self) -> Tensor:
        """Return the threshold on the coefficient of variation of the Monte Carlo estimates."""
        return torch.tensor(self.config["rbdo"]["CoV_thr"])

    @property
    def eps_max_thr(self) -> Tensor:
        """Return the threshold on the maximum relative error of the ESC stopping criterion."""
        return torch.tensor(self.config["gp"]["eps_max_thr"])

    def _set_alpha(self, alpha: float) -> None:
        """Set the level of the isocost region (called on the remote copies of the problem).

        Args:
            alpha: Isocost level.
        """
        self.alpha = alpha

    def _set_hard(self, metamodels: ListGPRBoTorch) -> None:
        """Set the surrogates of the hard constraints (called on the remote copies).

        Args:
            metamodels: Kriging surrogates, one per hard constraint.
        """
        self.metamodels = metamodels

    def run_mcs(self, samples: Tensor) -> utils.Table:
        """Estimate the reliability at a design by Monte Carlo simulation.

        The hard constraints are evaluated with the model of the analysis (the surrogates for
        ZAK-RBDO and ISORAK-RBDO).

        Args:
            samples: Monte Carlo samples of the augmented space, shape ``(n_samples, n_dims)``.

        Returns:
            A table with, for each hard constraint, the reliability index, the failure
            probability, their thresholds and whether the constraint is satisfied.
        """
        with torch.no_grad():
            g, _ = self.evaluate_hard(samples)
        N_fails = (g <= 0).sum(dim=0).cpu()
        Pf = N_fails / samples.shape[0]
        zita = Pf <= self.Pf_thr.cpu()
        Beta_thr = -Normal(0, 1).icdf(self.Pf_thr.cpu())
        Beta = -Normal(0, 1).icdf(Pf)
        Beta = torch.clamp(Beta, min=-BETA_MAX, max=BETA_MAX)
        tab = utils.table(
            ["Beta", "Beta_thr", "Pf", "Pf_thr", "Pf <= Pf_thr"],
            title="MCS results at Optimal Design",
        )
        for row in zip(Beta, Beta_thr, Pf, self.Pf_thr, zita, strict=False):
            formatted_row = [
                f"{r.item():.3f}"
                if i in [0, 1]
                else f"{r.item():.4e}"
                if i in [2, 3]
                else f"{r.item():.4f}"
                if i == 5
                else f"[green]{r.item()}[/green]"
                if r.item()
                else f"[red]{r.item()}[/red]"
                for i, r in enumerate(row)
            ]
            tab.add_row(*formatted_row)
        return tab

    def run_mcs_reference(self, samples: Tensor) -> utils.Table:
        """Estimate the reliability at a design by Monte Carlo simulation on the true model.

        Same as `run_mcs`, but the true limit-state functions of the benchmark are always
        used, and the coefficient of variation of each estimate is reported as well.

        Args:
            samples: Monte Carlo samples of the augmented space, shape ``(n_samples, n_dims)``.

        Returns:
            A table with, for each hard constraint, the reliability index, the failure
            probability, their thresholds, the coefficient of variation of the estimate and
            whether the thresholds are satisfied.
        """
        g, _ = self.benchmark.evaluate_hard(samples)
        N_fails = (g <= 0).sum(dim=0).cpu()
        Pf = N_fails / samples.shape[0]
        zita = Pf <= self.Pf_thr.cpu()
        Beta_thr = -Normal(0, 1).icdf(self.Pf_thr.cpu())
        Beta = -Normal(0, 1).icdf(Pf)
        Beta = torch.clamp(Beta, min=-BETA_MAX, max=BETA_MAX)
        CoV = ((1 - Pf) / (Pf * samples.shape[0])).sqrt()
        # Without any failed sample the coefficient of variation is infinite: it is replaced
        # by the threshold.
        CoV[torch.isinf(CoV)] = self.CoV_thr[0].cpu()
        eta = CoV <= self.CoV_thr.cpu()
        tab = utils.table(
            [
                "Beta",
                "Beta_thr",
                "Pf",
                "Pf_thr",
                "Pf <= Pf_thr",
                "CoV",
                "CoV <= CoV_thr",
            ],
            title="MCS results at Optimal Design w/ reference model",
        )
        for row in zip(Beta, Beta_thr, Pf, self.Pf_thr, zita, CoV, eta, strict=False):
            formatted_row = [
                f"{r.item():.3f}"
                if i in [0, 1]
                else f"{r.item():.4e}"
                if i in [2, 3]
                else f"{r.item():.4f}"
                if i == 5
                else f"[green]{r.item()}[/green]"
                if r.item()
                else f"[red]{r.item()}[/red]"
                for i, r in enumerate(row)
            ]
            tab.add_row(*formatted_row)
        return tab

    @staticmethod
    def get_index_admissible(hard, soft) -> Tensor:
        """Return the mask of the designs whose failure probability is worth estimating.

        Args:
            hard: Hard constraints at the mean point, shape ``(n, n_hard)`` or ``(n_hard,)``.
            soft: Soft constraints, shape ``(n, n_soft)`` or ``(n_soft,)``, or None.

        Returns:
            A boolean tensor of shape ``(n,)``. Without soft constraints, a design is discarded
            when its mean point violates a hard constraint. With soft constraints, it is
            discarded only when it violates both a hard and a soft constraint.
        """
        # A single design: (n_hard,) -> (1, n_hard).
        if hard.dim() == 1:
            hard = hard.unsqueeze(0)
        idx_hard_violations = torch.any(hard <= 0, dim=1)
        if soft is not None:
            if soft.dim() == 1:
                soft = soft.unsqueeze(0)
            idx_soft_violations = torch.any(soft > 0, dim=1)
            idx_violations = idx_soft_violations & idx_hard_violations
        else:
            idx_violations = idx_hard_violations
        return ~idx_violations
