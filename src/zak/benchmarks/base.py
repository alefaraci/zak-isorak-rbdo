"""Benchmark problems for reliability-based design optimization (RBDO).

A benchmark bundles everything that defines one RBDO problem:

* the cost function ``c(d)`` to minimise over the design variables ``d``;
* the hard constraints, i.e. the limit-state functions ``g_k`` whose failure probability
  ``P(g_k(x(d), z) <= 0)`` is bounded, and the optional soft (deterministic) constraints
  ``f_j(d) <= 0``;
* the bounds of the design space and of the augmented space (design + random variables) in which
  the Kriging surrogates of ``g_k`` are built;
* the probabilistic model of the random design variables ``x`` (centred on ``d``) and of the
  environmental variables ``z``: quasi-Monte Carlo sampling (``get_W``), crude Monte Carlo
  sampling (``get_W_MCS``) and the equivalent OpenTURNS distribution used for the
  iso-probabilistic transform to the standard normal space;
* the labels used in the plots.

The benchmarks are registered by id in :data:`register_benchmark`.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
import math
import warnings

from attrs import field, define
from torch import Tensor
from scipy.stats import mstats
from torch.distributions import Gumbel, Normal, Uniform, Weibull, LogNormal
from botorch.sampling.qmc import NormalQMCEngine
import numpy as np
import torch
import openturns as ot

from ..test_functions.cost import (
    Truss,
    Bracket,
    Stackup,
    EggHolder,
    SimpleSum,
    CostFunction,
    ToleranceVolume,
    CircularParaboloid,
    SimpleMultiplication,
)
from ..test_functions.hard import (
    BracketG1,
    BracketG2,
    StackupG1,
    StackupG2,
    CantileverG1,
    CantileverG2,
    CantileverG3,
    TrussBridgeG,
    FourBranchesG,
    HardConstraint,
    ThreeNonlinearG1,
    ThreeNonlinearG2,
    ThreeNonlinearG3,
    HighlyNonlinearG1,
    HighlyNonlinearG2,
    ColumnUnderCompressionG,
)
from ..test_functions.soft import (
    Greater,
    SoftConstraint,
    Cantilever2dNominalToleranceLower,
    Cantilever4dNominalToleranceLower,
    Cantilever2dNominalToleranceGreater,
    Cantilever4dNominalToleranceGreater,
)

# Euler-Mascheroni constant: the mean of a Gumbel(loc, beta) variable is loc + gamma * beta.
EULER_MASCHERONI = 0.577216


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class BenchmarkBase(ABC):
    """Base class of the RBDO benchmark problems.

    Subclasses set the fields below through their defaults and implement the probabilistic model
    (``get_W``, ``get_W_MCS`` and ``get_probabilistic_model``).

    Attributes:
        id: Identifier of the benchmark, also its key in ``register_benchmark``.
        dim: Number of design variables.
        cost: Cost function ``c`` to minimise.
        hard: Limit-state functions ``g_k`` of the probabilistic (hard) constraints; ``g_k <= 0``
            is the failure event.
        soft: Deterministic (soft) constraints ``f_j <= 0``, or ``None`` if there are none.
        bounds: Bounds of the design variables, shape ``(dim, 2)`` with columns (lower, upper).
        aug_bounds: Bounds of the augmented space in which the surrogates are built, shape
            ``(n_aug, 2)``. Rows follow the column order of the samples returned by ``get_W``.
        min_max: Minimum and maximum of the cost over the design space, used to normalise the
            cost to ``[0, 1]``.
        seed: Seed of the quasi-random engines and of the Monte Carlo sampling.
        dtype: Floating-point type to which every tensor field is converted.
        scale: Standard deviations of the random design variables, shape ``(1, n)``.
        cov: Coefficients of variation of the random design variables, shape ``(1, n)``.
        cp: Process-capability indices Cp of the toleranced variables, shape ``(1, n)``.
        cpk: Process-capability indices Cpk of the toleranced variables, shape ``(1, n)``.
        nominal: Fixed nominal values of the toleranced variables, shape ``(1, n)``, for the
            benchmarks whose design vector contains the tolerances only.
        delta: Mean-shift samples of the last ``get_W`` call (only set by
            ``LinearStackupMeanShift``).
        label_d: LaTeX labels of the design variables.
        label_X: LaTeX labels of the random design variables.
        nominal_tolerance_design: ``True`` when the design vector is the concatenation of the
            nominal values and of the tolerances of the random variables (``dim`` is then twice
            the number of random variables). The cost function then depends on the tolerance
            half of the design vector only.
    """

    id: str = field(default=None)
    dim: int = field(default=None)
    cost: CostFunction = field(default=None)
    hard: Sequence[HardConstraint] = field(default=None)
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.empty(0))
    aug_bounds: Tensor = field(default=torch.empty(0))
    min_max: Tensor = field(default=torch.empty(0))
    seed: int = field(default=30)
    dtype: torch.dtype = field(default=torch.float64)
    scale: Tensor = field(default=torch.empty(0))
    cov: Tensor = field(default=torch.empty(0))
    cp: Tensor = field(default=torch.empty(0))
    cpk: Tensor = field(default=torch.empty(0))
    nominal: Tensor = field(default=torch.empty(0))
    delta: Tensor = field(default=torch.empty(0))
    label_d: list[str] = field(default=[])
    label_X: list[str] = field(default=[])
    nominal_tolerance_design: bool = field(default=False)

    def __attrs_post_init__(self) -> None:
        """Cast the tensor fields to ``dtype`` and attach the normalisation data to the cost."""
        self.__convert_tensors_to_dtype()
        self.cost._min_max = self.min_max
        self.cost.isolevels = self._cost_norm_isolevels()

    def __convert_tensors_to_dtype(self) -> None:
        """Convert every tensor attribute to ``self.dtype``."""
        for field_name in dir(self):
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                setattr(self, field_name, value.to(self.dtype))

    def _cost_norm_isolevels(self) -> dict[float, float]:
        """Compute the isocost levels of the normalised cost over the design space.

        The level associated with a probability ``alpha`` is the ``alpha``-quantile of the
        normalised cost of designs uniformly distributed within the bounds, estimated from
        ``1e7`` scrambled Sobol points.

        Returns:
            Mapping from ``alpha`` (0.001 to 1.0 in steps of 0.001, rounded to three decimals)
            to the corresponding normalised cost level.
        """
        # For a nominal/tolerance design the cost is a function of the tolerances only (it is
        # always evaluated on the second half of the design vector), so the quantiles are taken
        # over the tolerance bounds alone.
        lb = (
            self.bounds[:, 0]
            if not self.nominal_tolerance_design
            else self.bounds[:, 0][int(self.dim / 2) :]
        )
        ub = (
            self.bounds[:, 1]
            if not self.nominal_tolerance_design
            else self.bounds[:, 1][int(self.dim / 2) :]
        )
        dim = self.dim if not self.nominal_tolerance_design else int(self.dim / 2)
        sobol_engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=self.seed)
        sobol_samples = sobol_engine.draw(int(1e7))
        mc_pop = lb + (ub - lb) * sobol_samples
        alpha = 0.001
        iso_lines_prob = torch.arange(alpha, 1 + alpha, alpha, dtype=self.min_max.dtype)
        iso_levels = torch.tensor(mstats.mquantiles(self.cost.fn_norm(mc_pop), prob=iso_lines_prob))
        return {
            round(float(prb), 3): float(iso)
            for prb, iso in zip(iso_lines_prob, iso_levels, strict=False)
        }

    @abstractmethod
    def get_probabilistic_model(self, design: Tensor) -> ot.Distribution:
        """Return the joint distribution of the augmented variables as an OpenTURNS object.

        Args:
            design: The design point on which the distribution depends, shape ``(1, dim)``.

        Returns:
            The joint distribution, with marginals in the column order of ``get_W``.
        """

    def transform_to_u_space(self, w_physical: Tensor, design: Tensor) -> Tensor:
        """Map samples from the physical space to the standard normal space (U-space).

        The iso-probabilistic transformation is the one of the distribution returned by
        ``get_probabilistic_model`` for the given design.

        Args:
            w_physical: Samples in the physical (augmented) space, shape ``(n_samples, n_aug)``.
            design: The design point defining the distribution.

        Returns:
            The samples in the U-space, with the same shape as ``w_physical``.
        """
        ot_distribution = self.get_probabilistic_model(design)
        w_physical_np = w_physical.cpu().numpy()
        ot_sample_physical = ot.Sample(w_physical_np)
        isoprob_transform = ot_distribution.getIsoProbabilisticTransformation()
        ot_sample_u_space = isoprob_transform(ot_sample_physical)
        u_space_np = np.array(ot_sample_u_space)
        return torch.from_numpy(u_space_np).to(self.dtype)

    @staticmethod
    def _get_lognormal_params(loc: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        """Convert the mean and standard deviation of a LogNormal variable to its parameters.

        Args:
            loc: Mean of the LogNormal variable.
            scale: Standard deviation of the LogNormal variable.

        Returns:
            Mean and standard deviation of the underlying Normal variable:
            ``sigma^2 = ln(1 + (scale / loc)^2)`` and ``mu = ln(loc) - sigma^2 / 2``.
        """
        variance_ratio = (scale**2) / (loc**2)
        variance = torch.log(variance_ratio + 1)
        normal_scale = variance.sqrt()
        normal_loc = loc.log() - variance / 2
        return normal_loc, normal_scale

    @abstractmethod
    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample the augmented variables ``w = {x, z}`` around a population of designs.

        The samples are quasi-random with a fixed seed, so every design of the population (and
        every call) sees the same underlying sequence.

        Args:
            N_SAMPLES: Number of samples per design. With ``N_SAMPLES == 1`` a single evaluation
                point per design is returned instead (see each benchmark).
            design: Population of designs, shape ``(POP_SIZE, dim)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, n_aug)``.
        """

    @abstractmethod
    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample the augmented variables ``w = {x, z}`` for a crude Monte Carlo simulation.

        The global torch seed is reset to ``self.seed`` at every call.

        Args:
            N_SAMPLES: Number of Monte Carlo samples.
            design: A single design, shape ``(1, dim)``.

        Returns:
            The samples, shape ``(N_SAMPLES, n_aug)``.
        """

    def _get_normal(
        self,
        N_SAMPLES: int,
        loc: Tensor,
        scale: Tensor,
        engine: NormalQMCEngine,
    ) -> Tensor:
        """Draw quasi-random Normal samples around each row of ``loc``.

        The same standard normal draws are shared by all the rows of ``loc``.

        Args:
            N_SAMPLES: Number of samples per row of ``loc``.
            loc: Means, shape ``(POP_SIZE, n)``.
            scale: Standard deviations, shape ``(POP_SIZE, n)`` or ``(1, n)``.
            engine: Quasi-random standard normal engine of dimension ``n``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, n)``.
        """
        std_norm_samples = engine.draw(N_SAMPLES).to(loc.device)  # pyright: ignore
        return loc.unsqueeze(1) + scale.unsqueeze(1).to(loc.device) * std_norm_samples.unsqueeze(0)

    def evaluate_hard(self, X: Tensor) -> list[Tensor]:
        """Evaluate the true limit-state functions.

        Args:
            X: Points of the augmented space, shape ``(..., n_aug)``.

        Returns:
            A two-item list: the values of the limit states stacked on the last dimension, shape
            ``(..., n_hard)``, and an empty tensor standing for the predictive variance, so that
            the output has the same layout as a surrogate evaluation.
        """
        return [torch.stack([gk.fn(X) for gk in self.hard], dim=-1), torch.empty(0)]

    def evaluate_cost(self, X: Tensor) -> Tensor:
        """Evaluate the cost function."""
        return self.cost.fn(X)

    def evaluate_cost_norm(self, X: Tensor) -> Tensor:
        """Evaluate the cost function normalised to ``[0, 1]`` with ``min_max``."""
        return self.cost.fn_norm(X)

    def evaluate_soft(self, X: Tensor) -> Tensor | None:
        """Evaluate the soft constraints.

        Args:
            X: Design points, shape ``(..., dim)``.

        Returns:
            The values of the soft constraints stacked on the last dimension, shape
            ``(..., n_soft)``, or ``None`` if the benchmark has no soft constraint.
        """
        return (
            torch.stack([fj.fn(X) for fj in self.soft], dim=-1) if self.soft is not None else None
        )


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class HighlyNonlinear(BenchmarkBase):
    """Two-dimensional highly nonlinear benchmark (the "Haupt problem").

    Design variables ``d = (d1, d2)`` are the means of the random variables ``X1, X2``, which are
    independent Normal with standard deviation 0.1. The cost is a circular paraboloid. There are
    two hard constraints (one highly nonlinear, one linear) and no soft constraint.
    """

    id: str = "highly_nonlinear"
    dim: int = 2
    cost: CostFunction = field(default=CircularParaboloid())
    hard: Sequence[HardConstraint] = field(default=[HighlyNonlinearG1(), HighlyNonlinearG2()])
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[0.0, 3.7], [0.0, 4.0]]))
    aug_bounds: Tensor = field(default=torch.tensor([[-0.6, 4.3], [-0.6, 4.6]]))
    min_max: Tensor = field(default=torch.tensor([0.0, 29.69]))
    scale: Tensor = field(default=torch.tensor([[0.1, 0.1]]))
    label_d: list[str] = field(default=[r"$d_1$", r"$d_2$"])
    label_X: list[str] = field(default=[r"$X_1$", r"$X_2$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(design, scale)`` with a quasi-random sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs, shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``. With ``N_SAMPLES == 1`` the designs
            themselves (the means) are returned unchanged.
        """
        if N_SAMPLES == 1:
            return design
        engine = NormalQMCEngine(d=self.dim, seed=self.seed)
        return self._get_normal(N_SAMPLES=N_SAMPLES, loc=design, scale=self.scale, engine=engine)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(design, scale)`` for a crude Monte Carlo simulation."""
        torch.manual_seed(self.seed)
        return Normal(loc=design, scale=self.scale).sample((N_SAMPLES,)).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``x``: independent ``Normal(design_i, scale_i)``."""
        design_params = design.squeeze().cpu().numpy()
        scale_params = self.scale.squeeze().cpu().numpy()
        marginals = [
            ot.Normal(float(design_params[i]), float(scale_params[i])) for i in range(self.dim)
        ]
        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ThreeNonlinear(BenchmarkBase):
    """Two-dimensional benchmark with three nonlinear limit states (the "Choi problem").

    Design variables ``d = (d1, d2)`` are the means of the random variables ``X1, X2``, which are
    independent Normal with standard deviation 0.3. The cost is ``d1 + d2``. There are three
    hard constraints and no soft constraint.
    """

    id: str = "three_nonlinear"
    dim: int = 2
    cost: CostFunction = field(default=SimpleSum())
    hard: Sequence[HardConstraint] = field(
        default=[ThreeNonlinearG1(), ThreeNonlinearG2(), ThreeNonlinearG3()],
    )
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[0.0, 10.0], [0.0, 10.0]]))
    aug_bounds: Tensor = field(default=torch.tensor([[-1.8, 11.8], [-1.8, 11.8]]))
    min_max: Tensor = field(default=torch.tensor([0.0, 20.0]))
    scale: Tensor = field(default=torch.tensor([[0.3, 0.3]]))
    label_d: list[str] = field(default=[r"$d_1$", r"$d_2$"])
    label_X: list[str] = field(default=[r"$X_1$", r"$X_2$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(design, scale)`` with a quasi-random sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs, shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``. With ``N_SAMPLES == 1`` the designs
            themselves (the means) are returned unchanged.
        """
        if N_SAMPLES == 1:
            return design
        engine = NormalQMCEngine(d=self.dim, seed=self.seed)
        return self._get_normal(N_SAMPLES=N_SAMPLES, loc=design, scale=self.scale, engine=engine)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(design, scale)`` for a crude Monte Carlo simulation."""
        torch.manual_seed(self.seed)
        return Normal(loc=design, scale=self.scale).sample((N_SAMPLES,)).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``x``: independent ``Normal(design_i, scale_i)``."""
        design_params = design.squeeze().cpu().numpy()
        scale_params = self.scale.squeeze().cpu().numpy()
        marginals = [
            ot.Normal(float(design_params[i]), float(scale_params[i])) for i in range(self.dim)
        ]
        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class FourBranches(BenchmarkBase):
    """Two-dimensional benchmark with a four-branch series-system limit state.

    Design variables ``d = (d1, d2)`` are the means of the random variables ``X1, X2``, which are
    independent Normal with standard deviation 0.3. The cost is a rescaled Eggholder function.
    There is one hard constraint (the minimum of four branches) and no soft constraint.
    """

    id: str = "four_branches"
    dim: int = 2
    cost: CostFunction = field(default=EggHolder())
    hard: Sequence[HardConstraint] = field(default=[FourBranchesG()])
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[-6.0, 6.0], [-6.0, 6.0]]))
    aug_bounds: Tensor = field(default=torch.tensor([[-7.8, 7.8], [-7.8, 7.8]]))
    min_max: Tensor = field(default=torch.tensor([-127.8, 126.5]))
    scale: Tensor = field(default=torch.tensor([[0.3, 0.3]]))
    label_d: list[str] = field(default=[r"$d_1$", r"$d_2$"])
    label_X: list[str] = field(default=[r"$X_1$", r"$X_2$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(design, scale)`` with a quasi-random sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs, shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``. With ``N_SAMPLES == 1`` the designs
            themselves (the means) are returned unchanged.
        """
        if N_SAMPLES == 1:
            return design
        engine = NormalQMCEngine(d=self.dim, seed=self.seed)
        return self._get_normal(N_SAMPLES=N_SAMPLES, loc=design, scale=self.scale, engine=engine)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(design, scale)`` for a crude Monte Carlo simulation."""
        torch.manual_seed(self.seed)
        return Normal(loc=design, scale=self.scale).sample((N_SAMPLES,)).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``x``: independent ``Normal(design_i, scale_i)``."""
        design_params = design.squeeze().cpu().numpy()
        scale_params = self.scale.squeeze().cpu().numpy()
        marginals = [
            ot.Normal(float(design_params[i]), float(scale_params[i])) for i in range(self.dim)
        ]
        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ColumnUnderCompression(BenchmarkBase):
    """Column under compression with a rectangular cross-section.

    Design variables are the width ``b`` and the height ``h`` of the cross-section; they are
    deterministic (``x = d``). The environmental variables ``z = (k, E, L)`` are independent
    LogNormal with means (0.6, 1e4, 3e3) and coefficients of variation (10 %, 5 %, 1 %). The cost
    is the area ``b * h``. There is one hard constraint (buckling load against the service load)
    and one soft constraint (``h <= b``).
    """

    id: str = "column_under_compression"
    dim: int = 2
    cost: CostFunction = field(default=SimpleMultiplication())
    hard: Sequence[HardConstraint] = field(default=[ColumnUnderCompressionG()])
    soft: Sequence[SoftConstraint] | None = field(default=[Greater()])
    bounds: Tensor = field(default=torch.tensor([[150.0, 350.0], [150.0, 350.0]]))
    aug_bounds: Tensor = field(
        default=torch.tensor(
            [
                [150.0, 350.0],  # b
                [150.0, 350.0],  # h
                [0.341, 1.020],  # k
                [7575.014, 13134.040],  # E
                [2837.906, 3172.929],  # L
            ],
        ),
    )
    min_max: Tensor = field(default=torch.tensor([22_500, 122_500]))
    label_d: list[str] = field(default=[r"$b$", r"$h$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = {b, h, k, E, L}`` with a quasi-random sequence for ``z``.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs ``(b, h)``, shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 5)``. With ``N_SAMPLES == 1`` the designs
            completed with the mean of ``z`` are returned, shape ``(POP_SIZE, 5)``.
        """
        POP_SIZE, DIM_Z = design.shape[0], 3
        # Means and standard deviations of z = {k, E, L}
        loc_z = torch.tensor([0.6, 1e4, 3.0e3])
        scale_z = torch.tensor([0.1 * 0.6, 0.05 * 1e4, 0.01 * 3e3])

        if N_SAMPLES == 1:
            z_samples = loc_z.expand(POP_SIZE, -1)
            return torch.cat([design, z_samples], dim=-1)

        # The z samples are drawn once, from a seed that depends neither on the population nor
        # on the design, and shared by all the individuals: every design is then assessed
        # against the same realisations of the environmental variables.
        fixed_env_seed = self.seed

        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=DIM_Z,
            scramble=True,
            seed=fixed_env_seed,
        )

        u_samples = sobol_engine.draw(N_SAMPLES)
        # LogNormal inverse CDF: sqrt(2) * erfinv(2u - 1) is the standard normal quantile.
        loc_lognorm, scale_lognorm = self._get_lognormal_params(loc_z, scale_z)
        z_base = torch.exp(
            loc_lognorm + scale_lognorm * math.sqrt(2) * torch.erfinv(2 * u_samples - 1),
        )
        # Shape: [POP_SIZE, N_SAMPLES, DIM_Z]
        z_samples = z_base.unsqueeze(0).expand(POP_SIZE, -1, -1)
        # b and h are deterministic: each design is repeated along the sample dimension.
        # Shape: [POP_SIZE, N_SAMPLES, 2]
        x_samples = design.unsqueeze(1).expand(-1, N_SAMPLES, -1)
        return torch.cat([x_samples, z_samples], dim=-1)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = {b, h, k, E, L}`` for a crude Monte Carlo simulation."""
        torch.manual_seed(self.seed)
        bh = design.expand(N_SAMPLES, -1)
        loc_lognorm, scale_lognorm = self._get_lognormal_params(
            torch.tensor([0.6, 1e4, 3.0e3]),
            torch.tensor([0.1 * 0.6, 0.05 * 1e4, 0.01 * 3e3]),
        )
        kel = LogNormal(loc=loc_lognorm, scale=scale_lognorm).sample((N_SAMPLES,)).squeeze()  # pyright: ignore
        return torch.cat([bh, kel], dim=-1)

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{b, h, k, E, L}``.

        ``b`` and ``h`` are deterministic for a given design and are modelled by Dirac
        distributions; ``k``, ``E`` and ``L`` are independent LogNormal variables.

        Args:
            design: The design ``(b, h)``.

        Returns:
            The joint distribution, with marginals in the order ``[b, h, k, E, L]`` of ``get_W``.
        """
        design_params = design.squeeze().cpu().numpy().tolist()

        b_dist = ot.Dirac(design_params[0])
        h_dist = ot.Dirac(design_params[1])

        # Means and standard deviations of z = {k, E, L}
        loc_z = torch.tensor([0.6, 1e4, 3.0e3])
        scale_z = torch.tensor([0.1 * 0.6, 0.05 * 1e4, 0.01 * 3e3])

        # ot.LogNormal is parameterised by the mean and standard deviation of the underlying
        # Normal variable.
        loc_lognorm, scale_lognorm = self._get_lognormal_params(loc_z, scale_z)

        loc_lognorm_np = loc_lognorm.cpu().numpy().tolist()
        scale_lognorm_np = scale_lognorm.cpu().numpy().tolist()

        k_dist = ot.LogNormal(loc_lognorm_np[0], scale_lognorm_np[0])
        E_dist = ot.LogNormal(loc_lognorm_np[1], scale_lognorm_np[1])
        L_dist = ot.LogNormal(loc_lognorm_np[2], scale_lognorm_np[2])

        marginals = [b_dist, h_dist, k_dist, E_dist, L_dist]

        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class BracketStructure(BenchmarkBase):
    """Two-member bracket structure.

    Design variables are the means of the member widths ``wAB``, ``wCD`` and of the thickness
    ``t``, which are independent Normal with a 5 % coefficient of variation. The environmental
    variables are ``z = (P, E, sy, rho, L)``: ``P`` and ``E`` Gumbel, ``sy`` LogNormal, ``rho``
    Weibull and ``L`` Normal. The cost is the weight of the structure. There are two hard
    constraints (bending stress against the yield strength, and buckling of member AB) and no
    soft constraint.
    """

    id: str = "bracket_structure"
    dim: int = 3
    cost: CostFunction = field(default=Bracket())
    hard: Sequence[HardConstraint] = field(default=[BracketG1(), BracketG2()])
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[5.0, 30.0], [5.0, 30.0], [5.0, 30.0]]))
    aug_bounds: Tensor = field(
        default=torch.tensor(
            [
                [3.587, 38.162],  # wAB
                [3.587, 38.162],  # wCD
                [3.587, 38.162],  # t
                [47.749, 367.800],  # P
                [143.497, 469.427],  # E
                [144.533, 350.278],  # sy
                [1396.136, 10990.509],  # rho
                [3.640, 6.504],  # L
            ],
        ),
    )
    min_max: Tensor = field(default=torch.tensor([118.38, 8702.76]))
    cov: Tensor = field(default=torch.tensor([[0.05, 0.05, 0.05]]))
    label_d: list[str] = field(default=[r"$w_{\mathrm{AB}}$", r"$w_{\mathrm{CD}}$", r"$t$"])
    label_X: list[str] = field(default=[])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = {wAB, wCD, t, P, E, sy, rho, L}`` with quasi-random sequences.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs (means of ``wAB, wCD, t``), shape ``(POP_SIZE, 3)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 8)``. With ``N_SAMPLES == 1`` the designs
            completed with the nominal values ``loc_z`` of ``z`` are returned, shape
            ``(POP_SIZE, 8)``.
        """
        POP_SIZE, DIM_Z = design.shape[0], 5
        # Distribution parameters of z = {P, E, sy, rho, L}
        loc_z = torch.tensor([100.0, 200.0, 225.0, 7860.0, 5.0])
        scale_z = torch.tensor(
            [0.15 * 100.0, 0.08 * 200.0, 0.08 * 225.0, 0.10 * 7860.0, 0.05 * 5.0],
        )
        if N_SAMPLES == 1:
            z_samples = loc_z.expand(POP_SIZE, -1)
            return torch.cat([design, z_samples], dim=-1)

        # x ~ Normal(design, cov * design), from a Sobol sequence of its own (different seed
        # from the one of z) mapped through the Normal inverse CDF.
        sobol_engine_x = torch.quasirandom.SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=self.seed + 2,
        )
        u_x = sobol_engine_x.draw(N_SAMPLES)
        u_x = u_x.unsqueeze(0).expand(POP_SIZE, -1, -1)
        x_samples = design.unsqueeze(1) + (design * self.cov).unsqueeze(1) * math.sqrt(
            2,
        ) * torch.erfinv(2 * u_x - 1)
        # The z samples are drawn once, from a seed that depends neither on the population nor
        # on the design, and shared by all the individuals: every design is then assessed
        # against the same realisations of the environmental variables.
        fixed_env_seed = self.seed
        sobol_engine_z = torch.quasirandom.SobolEngine(
            dimension=DIM_Z,
            scramble=True,
            seed=fixed_env_seed,
        )
        u_z = sobol_engine_z.draw(N_SAMPLES)
        # Each column of u_z is mapped through the inverse CDF of its distribution.
        # Gumbel for {P, E}: scale_z is used as the Gumbel scale beta, and the location is set
        # so that the mean (location + gamma * beta) equals loc_z.
        gumbel_loc = loc_z[0:2] - scale_z[0:2] * EULER_MASCHERONI
        gumbel_samples = gumbel_loc - scale_z[0:2] * torch.log(-torch.log(u_z[:, 0:2]))
        # LogNormal for {sy}, with mean loc_z and standard deviation scale_z
        loc_sy, scale_sy = loc_z[2], scale_z[2]
        loc_lognorm, scale_lognorm = self._get_lognormal_params(loc_sy, scale_sy)
        lognorm_samples = torch.exp(
            loc_lognorm + scale_lognorm * math.sqrt(2) * torch.erfinv(2 * u_z[:, 2:3] - 1),
        )
        # Weibull for {rho}: the shape k is taken as the inverse of the coefficient of
        # variation, and the scale follows from the mean, mean = scale * Gamma(1 + 1/k).
        loc_rho, loc_cov = loc_z[3], scale_z[3] / loc_z[3]
        rho_k = 1 / loc_cov
        gamma_value = torch.exp(torch.lgamma((1 + 1 / rho_k).clone().detach()))
        scale_rho = loc_rho / gamma_value
        weibull_samples = scale_rho * (-torch.log(1 - u_z[:, 3:4])) ** (1 / rho_k)
        # Normal for {L}
        normal_samples = loc_z[4] + scale_z[4] * math.sqrt(2) * torch.erfinv(2 * u_z[:, 4:5] - 1)
        z_base = torch.cat(
            [gumbel_samples, lognorm_samples, weibull_samples, normal_samples],
            dim=-1,
        )
        # Shape: [POP_SIZE, N_SAMPLES, DIM_Z]
        z_samples = z_base.unsqueeze(0).expand(POP_SIZE, -1, -1)
        return torch.cat([x_samples, z_samples], dim=-1)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = {wAB, wCD, t, P, E, sy, rho, L}`` for a crude Monte Carlo simulation.

        The distribution parameters are derived as in ``get_W``.
        """
        torch.manual_seed(self.seed)
        wAB_wCD_t = (
            Normal(loc=design[:, : self.dim], scale=self.cov * design[:, : self.dim])
            .sample((N_SAMPLES,))
            .squeeze()
        )
        # Gumbel samples for P, E
        pe_loc, pe_scale = torch.tensor([100.0, 200.0]), torch.tensor([0.15 * 100.0, 0.08 * 200.0])
        gumbel_loc = pe_loc - pe_scale * EULER_MASCHERONI
        PE_distr = Gumbel(loc=gumbel_loc, scale=pe_scale)
        PE = PE_distr.sample((N_SAMPLES,)).squeeze()  # pyright: ignore
        # LogNormal samples for sy
        sy_loc = torch.tensor([225.0])
        sy_scale = torch.tensor([0.08 * 225.0])
        loc_lognorm, scale_lognorm = self._get_lognormal_params(sy_loc, sy_scale)
        sy = LogNormal(loc=loc_lognorm, scale=scale_lognorm).sample((N_SAMPLES,))
        # Weibull samples for rho
        loc_rho, loc_cov = torch.tensor([7860.0]), torch.tensor([0.10])
        rho_k = 1 / loc_cov
        gamma_value = torch.exp(torch.lgamma((1 + 1 / rho_k).clone().detach()))
        scale_rho = loc_rho / gamma_value
        rho = Weibull(scale=scale_rho, concentration=rho_k).sample((N_SAMPLES,))
        # Normal samples for L
        norm_loc, norm_scale = torch.tensor([5.0]), torch.tensor([0.05 * 5.0])
        normal_samples = Normal(loc=norm_loc, scale=norm_scale).sample((N_SAMPLES,))
        return torch.cat([wAB_wCD_t, PE, sy, rho, normal_samples], dim=-1)  # pyright: ignore

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{wAB, wCD, t, P, E, sy, rho, L}``.

        ``wAB``, ``wCD`` and ``t`` are Normal with mean the design and standard deviation
        ``cov * design``; ``P`` and ``E`` are Gumbel, ``sy`` is LogNormal, ``rho`` is Weibull
        and ``L`` is Normal. All the variables are independent and the distribution parameters
        are derived as in ``get_W``.

        Args:
            design: The design (means of ``wAB, wCD, t``).

        Returns:
            The joint distribution, with marginals in the column order of ``get_W``.
        """
        design_np = design.squeeze().cpu().numpy()

        # Design variables x = {wAB, wCD, t}
        scale_x_np = (self.cov.squeeze() * design.squeeze()).cpu().numpy()
        wAB_dist = ot.Normal(design_np[0], scale_x_np[0])
        wCD_dist = ot.Normal(design_np[1], scale_x_np[1])
        t_dist = ot.Normal(design_np[2], scale_x_np[2])

        # Environmental variables z = {P, E, sy, rho, L}
        loc_z = torch.tensor([100.0, 200.0, 225.0, 7860.0, 5.0])
        scale_z = torch.tensor(
            [0.15 * 100.0, 0.08 * 200.0, 0.08 * 225.0, 0.10 * 7860.0, 0.05 * 5.0],
        )
        loc_z_np = loc_z.cpu().numpy()
        scale_z_np = scale_z.cpu().numpy()

        # ot.Gumbel(beta, gamma): beta is the scale and gamma the location.
        gumbel_loc_np = loc_z_np[0:2] - scale_z_np[0:2] * EULER_MASCHERONI
        P_dist = ot.Gumbel(scale_z_np[0], gumbel_loc_np[0])
        E_dist = ot.Gumbel(scale_z_np[1], gumbel_loc_np[1])

        loc_sy, scale_sy = loc_z[2:3], scale_z[2:3]
        loc_lognorm, scale_lognorm = self._get_lognormal_params(loc_sy, scale_sy)
        sy_dist = ot.LogNormal(loc_lognorm.item(), scale_lognorm.item())

        # ot.WeibullMin(beta, alpha): beta is the scale and alpha the shape.
        loc_rho, loc_cov = loc_z[3], scale_z[3] / loc_z[3]
        rho_k = 1 / loc_cov
        gamma_value = torch.exp(torch.lgamma(1 + 1 / rho_k))
        scale_rho = loc_rho / gamma_value
        rho_dist = ot.WeibullMin(scale_rho.item(), rho_k.item())

        L_dist = ot.Normal(loc_z_np[4], scale_z_np[4])

        marginals = [wAB_dist, wCD_dist, t_dist, P_dist, E_dist, sy_dist, rho_dist, L_dist]

        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class TrussBridge(BenchmarkBase):
    """Truss bridge with two groups of bars.

    Design variables are the means of the cross-sections ``A1`` and ``A2``, which are
    independent LogNormal with standard deviations (2e-4, 1e-4). The environmental variables
    are the Young's moduli ``E1, E2`` (LogNormal) and the six loads ``P1..P6`` (Gumbel). The
    cost is a weighted sum of the two cross-sections. There is one hard constraint (on the
    maximum displacement) and no soft constraint.
    """

    id: str = "truss_bridge"
    dim: int = 2
    cost: CostFunction = field(default=Truss())
    hard: Sequence[HardConstraint] = field(default=[TrussBridgeG()])
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[6e-4, 6e-3], [3e-4, 3e-3]]))
    aug_bounds: Tensor = field(
        default=torch.tensor(
            [
                [9.4e-5, 7.2e-3],  # A1
                [4.5e-5, 3.6e-3],  # A2
                [1.209e11, 3.615e11],  # E1
                [1.209e11, 3.615e11],  # E2
                [2.430e04, 1.865e05],  # P1
                [2.430e04, 1.865e05],  # P2
                [2.430e04, 1.865e05],  # P3
                [2.430e04, 1.865e05],  # P4
                [2.430e04, 1.865e05],  # P5
                [2.430e04, 1.865e05],  # P6
            ],
        ),
    )
    min_max: Tensor = field(default=torch.tensor([0.036588, 0.36588]))
    scale: Tensor = field(default=torch.tensor([[2 * 1e-4, 1 * 1e-4]]))
    label_d: list[str] = field(default=[r"$\mu_{A_1}$", r"$\mu_{A_2}$"])
    label_X: list[str] = field(default=[r"$A_1$", r"$A_2$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = {A1, A2, E1, E2, P1..P6}`` with quasi-random sequences.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs (means of ``A1, A2``), shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 10)``. With ``N_SAMPLES == 1`` the
            designs completed with the nominal values ``loc_z`` of ``z`` are returned, shape
            ``(POP_SIZE, 10)``.
        """
        POP_SIZE, DIM_Z = design.shape[0], 8
        # Distribution parameters of z = {E1, E2, P1, P2, P3, P4, P5, P6}
        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
        scale_z = torch.tensor([2.10e10, 2.10e10, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3])
        if N_SAMPLES == 1:
            z_samples = loc_z.expand(POP_SIZE, -1)
            return torch.cat([design, z_samples], dim=-1)

        # x = {A1, A2} is LogNormal with mean the design and standard deviation self.scale,
        # from a Sobol sequence of its own (different seed from the one of z) mapped through
        # the LogNormal inverse CDF.
        loc_lognorm_x, scale_lognorm_x = self._get_lognormal_params(
            design.unsqueeze(1),
            self.scale.unsqueeze(1),
        )
        sobol_engine_x = torch.quasirandom.SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=self.seed + 1,
        )
        u_x = sobol_engine_x.draw(N_SAMPLES)
        u_x = u_x.unsqueeze(0).expand(POP_SIZE, -1, -1)
        x_samples = torch.exp(
            loc_lognorm_x + scale_lognorm_x * math.sqrt(2) * torch.erfinv(2 * u_x - 1),
        )
        # The z samples are drawn once, from a seed that depends neither on the population nor
        # on the design, and shared by all the individuals: every design is then assessed
        # against the same realisations of the environmental variables.
        fixed_env_seed = self.seed

        sobol_engine_z = torch.quasirandom.SobolEngine(
            dimension=DIM_Z,
            scramble=True,
            seed=fixed_env_seed,
        )
        u_z = sobol_engine_z.draw(N_SAMPLES)
        # Each column of u_z is mapped through the inverse CDF of its distribution.
        # LogNormal for {E1, E2}, with mean loc_z and standard deviation scale_z
        loc_Ei, scale_Ei = loc_z[0:2], scale_z[0:2]
        loc_lognorm, scale_lognorm = self._get_lognormal_params(loc_Ei, scale_Ei)
        lognorm_samples = torch.exp(
            loc_lognorm + scale_lognorm * math.sqrt(2) * torch.erfinv(2 * u_z[:, 0:2] - 1),
        )
        # Gumbel for {P1..P6}: scale_z is used as the Gumbel scale beta, and the location is
        # set so that the mean (location + gamma * beta) equals loc_z.
        gumbel_loc = loc_z[2:] - scale_z[2:] * EULER_MASCHERONI
        gumbel_samples = gumbel_loc - scale_z[2:] * torch.log(-torch.log(u_z[:, 2:]))
        z_base = torch.cat([lognorm_samples, gumbel_samples], dim=-1)
        # Shape: [POP_SIZE, N_SAMPLES, DIM_Z]
        z_samples = z_base.unsqueeze(0).expand(POP_SIZE, -1, -1)
        return torch.cat([x_samples, z_samples], dim=-1)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = {A1, A2, E1, E2, P1..P6}`` for a crude Monte Carlo simulation.

        The distribution parameters are derived as in ``get_W``.
        """
        torch.manual_seed(self.seed)
        # Distribution parameters of z = {E1, E2, P1, P2, P3, P4, P5, P6}
        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
        scale_z = torch.tensor([2.10e10, 2.10e10, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3])
        # LogNormal distributions for x(d) = {A1, A2}
        loc_lognorm, scale_lognorm = self._get_lognormal_params(design, self.scale)
        A1_A2 = LogNormal(loc=loc_lognorm, scale=scale_lognorm).sample((N_SAMPLES,)).squeeze()  # pyright: ignore
        # LogNormal distributions for z = {E1, E2}
        loc_Ei, scale_Ei = loc_z[0:2], scale_z[0:2]
        loc_lognorm, scale_lognorm = self._get_lognormal_params(loc_Ei, scale_Ei)
        E1_E2 = LogNormal(loc=loc_lognorm, scale=scale_lognorm).sample((N_SAMPLES,)).squeeze()  # pyright: ignore
        # Gumbel distributions for z = {P1, P2, P3, P4, P5, P6}
        gumbel_loc = loc_z[2:] - scale_z[2:] * EULER_MASCHERONI
        Pi = Gumbel(loc=gumbel_loc, scale=scale_z[2:]).sample((N_SAMPLES,)).squeeze()  # pyright: ignore
        return torch.cat([A1_A2, E1_E2, Pi], dim=-1)

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{A1, A2, E1, E2, P1..P6}``.

        ``A1`` and ``A2`` are LogNormal with mean the design and standard deviation
        ``self.scale``; ``E1`` and ``E2`` are LogNormal and ``P1`` to ``P6`` are Gumbel, with
        fixed parameters. All the variables are independent and the distribution parameters are
        derived as in ``get_W``.

        Args:
            design: The design (means of ``A1, A2``).

        Returns:
            The joint distribution, with marginals in the column order of ``get_W``.
        """
        # Design-dependent variables x = {A1, A2}
        loc_lognorm_x, scale_lognorm_x = self._get_lognormal_params(
            design.squeeze(),
            self.scale.squeeze(),
        )
        loc_lognorm_x_np = loc_lognorm_x.cpu().numpy().tolist()
        scale_lognorm_x_np = scale_lognorm_x.cpu().numpy().tolist()

        A1_dist = ot.LogNormal(loc_lognorm_x_np[0], scale_lognorm_x_np[0])
        A2_dist = ot.LogNormal(loc_lognorm_x_np[1], scale_lognorm_x_np[1])

        # Environmental variables z = {E1, E2, P1..P6}
        loc_z = torch.tensor([2.10e11, 2.10e11, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
        scale_z = torch.tensor([2.10e10, 2.10e10, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3])

        loc_Ei, scale_Ei = loc_z[0:2], scale_z[0:2]
        loc_lognorm_E, scale_lognorm_E = self._get_lognormal_params(loc_Ei, scale_Ei)
        loc_lognorm_E_np = loc_lognorm_E.cpu().numpy().tolist()
        scale_lognorm_E_np = scale_lognorm_E.cpu().numpy().tolist()

        E1_dist = ot.LogNormal(loc_lognorm_E_np[0], scale_lognorm_E_np[0])
        E2_dist = ot.LogNormal(loc_lognorm_E_np[1], scale_lognorm_E_np[1])

        # ot.Gumbel(beta, gamma): beta is the scale and gamma the location.
        gumbel_loc = loc_z[2:] - scale_z[2:] * EULER_MASCHERONI
        gumbel_scale = scale_z[2:]

        gumbel_loc_np = gumbel_loc.cpu().numpy().tolist()
        gumbel_scale_np = gumbel_scale.cpu().numpy().tolist()

        P_dists = [ot.Gumbel(gumbel_scale_np[i], gumbel_loc_np[i]) for i in range(6)]

        marginals = [A1_dist, A2_dist, E1_dist, E2_dist, *P_dists]

        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class LinearStackup(BenchmarkBase):
    """Tolerance design of a linear stack-up of two dimensions.

    Design variables are the tolerances ``t1, t2``. The random variables ``X1, X2`` are
    independent Normal centred on the fixed nominal values (6, 4) with standard deviation
    ``t / (6 Cp)``, ``Cp = 1``. The cost ``1 / t1 + 5 / t2`` decreases as the tolerances widen.
    There are two hard constraints (lower and upper limit on ``X1 + X2``) and no soft constraint.
    """

    id: str = "linear_stackup"
    dim: int = 2
    cost: CostFunction = field(default=Stackup())
    hard: Sequence[HardConstraint] = field(default=[StackupG1(), StackupG2()])
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[0.01, 1.0], [0.01, 1.0]]))
    aug_bounds: Tensor = field(default=torch.tensor([[5.1034, 7.0132], [3.0789, 4.8733]]))
    min_max: Tensor = field(default=torch.tensor([6.0, 600.0]))
    cp: Tensor = field(default=torch.tensor([[1.0, 1.0]]))
    cpk: Tensor = field(default=torch.tensor([[1.0, 1.0]]))
    nominal: Tensor = field(default=torch.tensor([[6.0, 4.0]]))
    label_d: list[str] = field(default=[r"$t_1$", r"$t_2$"])
    label_X: list[str] = field(default=[r"$X_1$", r"$X_2$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(nominal, design / (6 Cp))`` with a quasi-random sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs ``(t1, t2)``, shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``. With ``N_SAMPLES == 1`` the point
            ``nominal + standard deviation`` of each design is returned, shape ``(POP_SIZE, 2)``.
        """
        engine = NormalQMCEngine(d=self.dim, seed=self.seed)
        # A tolerance t spans 6 Cp standard deviations.
        scale = design / (6.0 * self.cp)
        if N_SAMPLES == 1:
            return self.nominal + scale
        return self._get_normal(N_SAMPLES=N_SAMPLES, loc=self.nominal, scale=scale, engine=engine)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x ~ Normal(nominal, design / (6 Cp))`` for a crude Monte Carlo simulation.

        With ``N_SAMPLES == 1`` the point ``nominal + standard deviation`` is returned.
        """
        torch.manual_seed(self.seed)
        scale = design / (6.0 * self.cp)
        if N_SAMPLES == 1:
            return self.nominal + scale
        return Normal(loc=self.nominal, scale=scale).sample((N_SAMPLES,)).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> None:
        """Not implemented for this benchmark: no OpenTURNS distribution is returned."""


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class LinearStackupMeanShift(BenchmarkBase):
    """Tolerance design of a linear stack-up of two dimensions, with a mean shift.

    Same design variables ``t1, t2``, nominal values, cost and hard constraints as
    ``LinearStackup``, with ``Cp = 1`` and ``Cpk = 0.8``. Each random variable is the sum of its
    nominal value, a uniform mean shift whose half-width derives from
    ``delta = t / 2 * (1 - Cpk / Cp)`` and a zero-mean Normal deviation whose standard deviation
    is the tolerance ``t`` itself.
    """

    id: str = "linear_stackup_mean_shift"
    dim: int = 2
    cost: CostFunction = field(default=Stackup())
    hard: Sequence[HardConstraint] = field(default=[StackupG1(), StackupG2()])
    soft: Sequence[SoftConstraint] | None = field(default=None)
    bounds: Tensor = field(default=torch.tensor([[0.01, 1.0], [0.01, 1.0]]))
    aug_bounds: Tensor = field(default=torch.tensor([[5.1034, 7.0132], [3.0789, 4.8733]]))
    min_max: Tensor = field(default=torch.tensor([6.0, 600.0]))
    cp: Tensor = field(default=torch.tensor([[1.0, 1.0]]))
    cpk: Tensor = field(default=torch.tensor([[0.8, 0.8]]))
    nominal: Tensor = field(default=torch.tensor([[6.0, 4.0]]))
    delta: Tensor = field(default=torch.empty(0))
    label_d: list[str] = field(default=[r"$t_1$", r"$t_2$"])
    label_X: list[str] = field(default=[r"$X_1$", r"$X_2$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = nominal + mean shift + Normal deviation`` with a Sobol sequence.

        The mean-shift samples are stored in ``self.delta``.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs ``(t1, t2)``, shape ``(POP_SIZE, 2)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``, squeezed when ``N_SAMPLES == 1``.
        """
        POP_SIZE = design.shape[0]
        nominal = self.nominal.expand(design.shape)
        tolerance = design
        # Half-width of the mean shift allowed by the capability indices Cp and Cpk.
        delta = tolerance / 2 * (1 - self.cpk / self.cp)

        if delta.size(0) == 0:
            # Empty population: use a zero shift so that the expressions below still broadcast.
            max_delta = torch.zeros(delta.size()[1:], dtype=delta.dtype, device=delta.device)
            warnings.warn("Empty delta tensor detected in get_W", stacklevel=2)
        else:
            # Largest half-width over the population, per variable: it is applied to all the
            # individuals.
            max_delta = delta.max(dim=0).values

        # One Sobol sequence, with a fixed seed, shared by all the individuals.
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=self.seed,
        )
        uniform_samples_base = sobol_engine.draw(N_SAMPLES)
        uniform_samples = uniform_samples_base.unsqueeze(0).expand(POP_SIZE, -1, -1)
        # The same uniform samples drive both terms: they are rescaled to
        # [-max_delta, max_delta] for the mean shift and mapped through the Normal inverse CDF
        # for the deviation around the shifted mean.
        u_samples = max_delta.unsqueeze(0).unsqueeze(0) * (2 * uniform_samples - 1)
        self.delta = u_samples
        normal_samples = 0.0 + tolerance.unsqueeze(1) * math.sqrt(2) * torch.erfinv(
            2 * uniform_samples - 1,
        )
        if N_SAMPLES == 1:
            return (nominal.unsqueeze(1) + u_samples + normal_samples).squeeze()
        return nominal.unsqueeze(1) + u_samples + normal_samples

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x`` for a crude Monte Carlo simulation.

        Unlike in ``get_W``, the uniform mean shift and the Normal deviation are drawn
        independently.
        """
        torch.manual_seed(self.seed)
        nominal = self.nominal.expand(design.shape)
        tolerance = design
        delta = tolerance / 2 * (1 - self.cpk / self.cp)
        max_delta = delta.max(dim=0).values
        uniform_samples = Uniform(-1, 1).sample((N_SAMPLES, self.dim))
        normal_samples = Normal(0, 1).sample((N_SAMPLES, self.dim))
        return (
            nominal
            + max_delta.unsqueeze(0) * uniform_samples
            + tolerance.unsqueeze(0) * normal_samples
        ).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> None:
        """Not implemented for this benchmark: no OpenTURNS distribution is returned."""


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class Cantilever2d(BenchmarkBase):
    """Tolerance design of a cantilever beam (nominal values and tolerances as design variables).

    The design vector is ``(T_h, T_L, t_h, t_L)``: nominal values then tolerances of the height
    ``h`` and of the length ``L``, which are independent Normal with mean ``T`` and standard
    deviation ``t / (6 Cp)``, ``Cp = 1.26``. The cost is minus the product of the tolerances.
    There are three hard constraints (minimum force, maximum force and maximum stress) and four
    soft constraints keeping each interval ``T +/- t / 2`` within the bounds of its variable.
    """

    id: str = "cantilever_2d"
    nominal_tolerance_design: bool = True
    dim: int = 4
    cost: CostFunction = field(default=ToleranceVolume())
    hard: Sequence[HardConstraint] = field(default=[CantileverG1(), CantileverG2(), CantileverG3()])
    soft: Sequence[SoftConstraint] | None = field(
        default=[
            Cantilever2dNominalToleranceGreater(var=0),
            Cantilever2dNominalToleranceLower(var=0),
            Cantilever2dNominalToleranceGreater(var=1),
            Cantilever2dNominalToleranceLower(var=1),
        ],
    )
    # Rows: nominal values T first, then tolerances t.
    bounds: Tensor = field(
        default=torch.tensor([[0.5, 2], [5.0, 20], [0.01, 1.5], [0.01, 15]]),
    )
    aug_bounds: Tensor = field(default=torch.tensor([[-0.4794, 2.7405], [-4.5722, 31.5917]]))
    min_max: Tensor = field(default=torch.tensor([-22.5, -0.0001]))
    cp: Tensor = field(default=torch.tensor([[1.26, 1.26]]))
    cpk: Tensor = field(default=torch.tensor([[1.10, 1.10]]))
    label_d: list[str] = field(default=[r"$T_h$", r"$T_L$", r"$t_h$", r"$t_L$"])
    label_X: list[str] = field(default=[r"$h$", r"$L$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = {h, L} ~ Normal(T, t / (6 Cp))`` with a quasi-random sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs ``(T_h, T_L, t_h, t_L)``, shape ``(POP_SIZE, 4)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``. With ``N_SAMPLES == 1`` the single
            quasi-random sample of each design is returned squeezed.
        """
        engine = NormalQMCEngine(d=int(self.dim / 2), seed=self.seed)
        loc = design[:, : int(self.dim / 2)]
        # A tolerance t spans 6 Cp standard deviations.
        scale = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        if N_SAMPLES == 1:
            return self._get_normal(
                N_SAMPLES=N_SAMPLES,
                loc=loc,
                scale=scale,
                engine=engine,
            ).squeeze()
        return self._get_normal(N_SAMPLES=N_SAMPLES, loc=loc, scale=scale, engine=engine)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = {h, L} ~ Normal(T, t / (6 Cp))`` for a crude Monte Carlo simulation."""
        torch.manual_seed(self.seed)
        loc = design[:, : int(self.dim / 2)]
        scale = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        return Normal(loc=loc, scale=scale).sample((N_SAMPLES,)).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{h, L}``: independent Normal variables.

        The means are the nominal values (first half of the design vector) and the standard
        deviations are the tolerances (second half) divided by ``6 Cp``.

        Args:
            design: The design vector ``(T_h, T_L, t_h, t_L)``.

        Returns:
            The joint distribution of ``h`` and ``L``.
        """
        design_squeezed = design.squeeze()

        loc_np = design_squeezed[: int(self.dim / 2)].cpu().numpy()

        tolerances = design_squeezed[int(self.dim / 2) :]
        scale_np = (tolerances / (6.0 * self.cp.squeeze())).cpu().numpy()

        h_dist = ot.Normal(loc_np[0], scale_np[0])
        L_dist = ot.Normal(loc_np[1], scale_np[1])

        marginals = [h_dist, L_dist]

        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class Cantilever2dMeanShift(BenchmarkBase):
    """Tolerance design of a cantilever beam with a mean shift.

    Same design vector ``(T_h, T_L, t_h, t_L)``, cost, hard and soft constraints as
    ``Cantilever2d``, with ``Cp = 1.26`` and ``Cpk = 1.10``. Each random variable (``h``, ``L``)
    is the sum of its nominal value ``T``, a uniform mean shift and a zero-mean Normal deviation
    of standard deviation ``sigma = t / (6 Cp)``. The half-width of the shift derives from
    ``delta = sigma / 2 * (1 - Cpk / Cp)``.
    """

    id: str = "cantilever_2d_mean_shift"
    nominal_tolerance_design: bool = True
    dim: int = 4
    cost: CostFunction = field(default=ToleranceVolume())
    hard: Sequence[HardConstraint] = field(default=[CantileverG1(), CantileverG2(), CantileverG3()])
    soft: Sequence[SoftConstraint] | None = field(
        default=[
            Cantilever2dNominalToleranceGreater(var=0),
            Cantilever2dNominalToleranceLower(var=0),
            Cantilever2dNominalToleranceGreater(var=1),
            Cantilever2dNominalToleranceLower(var=1),
        ],
    )
    # Rows: nominal values T first, then tolerances t.
    bounds: Tensor = field(
        default=torch.tensor([[0.5, 2], [5.0, 20], [0.01, 1.5], [0.01, 15]]),
    )
    aug_bounds: Tensor = field(
        default=torch.tensor([[-0.4794, 2.7405], [-4.5722, 31.5917]]),
    )
    min_max: Tensor = field(default=torch.tensor([-22.5, -0.0001]))
    cp: Tensor = field(default=torch.tensor([[1.26, 1.26]]))
    cpk: Tensor = field(default=torch.tensor([[1.10, 1.10]]))
    label_d: list[str] = field(default=[r"$T_h$", r"$T_L$", r"$t_h$", r"$t_L$"])
    label_X: list[str] = field(default=[r"$h$", r"$L$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = T + mean shift + Normal deviation`` with a Sobol sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs ``(T_h, T_L, t_h, t_L)``, shape ``(POP_SIZE, 4)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 2)``, squeezed when ``N_SAMPLES == 1``.
        """
        POP_SIZE = design.shape[0]
        nominal = design[:, : int(self.dim / 2)]
        # `tolerance` holds the process standard deviation t / (6 Cp), not the tolerance t.
        tolerance = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        # Half-width of the mean shift allowed by the capability indices Cp and Cpk.
        delta = tolerance / 2 * (1 - self.cpk / self.cp)

        if delta.size(0) == 0:
            # Empty population: use a zero shift so that the expressions below still broadcast.
            max_delta = torch.zeros(delta.size()[1:], dtype=delta.dtype, device=delta.device)
            warnings.warn("Empty delta tensor detected in get_W", stacklevel=2)
        else:
            # Largest half-width over the population, per variable: it is applied to all the
            # individuals.
            max_delta = delta.max(dim=0).values

        # One Sobol sequence, with a fixed seed, shared by all the individuals.
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=int(self.dim / 2),
            scramble=True,
            seed=self.seed,
        )
        uniform_samples_base = sobol_engine.draw(N_SAMPLES)

        uniform_samples = uniform_samples_base.unsqueeze(0).expand(POP_SIZE, -1, -1)

        # The same uniform samples drive both terms: they are rescaled to
        # [-max_delta, max_delta] for the mean shift and mapped through the Normal inverse CDF
        # for the deviation around the shifted mean.
        u_samples = max_delta.unsqueeze(0).unsqueeze(0) * (2 * uniform_samples - 1)
        normal_samples = 0.0 + tolerance.unsqueeze(1) * math.sqrt(2) * torch.erfinv(
            2 * uniform_samples - 1,
        )
        if N_SAMPLES == 1:
            return (nominal.unsqueeze(1) + u_samples + normal_samples).squeeze()
        return nominal.unsqueeze(1) + u_samples + normal_samples

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = {h, L}`` for a crude Monte Carlo simulation.

        Unlike in ``get_W``, the uniform mean shift and the Normal deviation are drawn
        independently.
        """
        torch.manual_seed(self.seed)
        nominal = design[:, : int(self.dim / 2)]
        tolerance = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        delta = tolerance / 2 * (1 - self.cpk / self.cp)
        max_delta = delta.max(dim=0).values
        uniform_samples = Uniform(-1, 1).sample((N_SAMPLES, int(self.dim / 2)))
        normal_samples = Normal(0, 1).sample((N_SAMPLES, int(self.dim / 2)))
        return (
            nominal
            + max_delta.unsqueeze(0) * uniform_samples
            + tolerance.unsqueeze(0) * normal_samples
        ).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{h, L}``.

        The two variables are independent. Each one is the sum of its nominal value, a Uniform
        mean shift and a Normal process spread::

            w = T + Uniform(-max_delta, +max_delta) + Normal(0, sigma)

        where ``max_delta`` is the largest ``delta`` over the two variables.

        Args:
            design: The design vector ``(T_h, T_L, t_h, t_L)``.

        Returns:
            The joint distribution of ``h`` and ``L``.
        """
        design_squeezed = design.squeeze()

        nominal = design_squeezed[: int(self.dim / 2)]
        tolerance = design_squeezed[int(self.dim / 2) :] / (6.0 * self.cp.squeeze())
        delta = tolerance / 2 * (1 - self.cpk.squeeze() / self.cp.squeeze())

        nominal_np = nominal.cpu().numpy()
        tolerance_np = tolerance.cpu().numpy()
        delta_np = delta.cpu().numpy()

        # A single half-width, the largest over the variables, is used for all the shifts.
        max_delta = np.max(delta_np)

        # Adding a Dirac at the nominal value translates the sum of the two random terms.
        h_uniform_dist = ot.Uniform(-max_delta, max_delta)
        h_normal_dist = ot.Normal(0.0, tolerance_np[0])
        h_dist = ot.Dirac(nominal_np[0]) + h_uniform_dist + h_normal_dist

        l_uniform_dist = ot.Uniform(-max_delta, max_delta)
        l_normal_dist = ot.Normal(0.0, tolerance_np[1])
        l_dist = ot.Dirac(nominal_np[1]) + l_uniform_dist + l_normal_dist

        marginals = [h_dist, l_dist]

        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class Cantilever4d(BenchmarkBase):
    """Tolerance design of a cantilever beam with four toleranced variables.

    The design vector is ``(T_h, T_L, T_b, T_f, t_h, t_L, t_b, t_f)``: nominal values then
    tolerances of the random variables ``h, L, b, f``, which are independent Normal with mean
    ``T`` and standard deviation ``t / (6 Cp)``, ``Cp = 1.26``. The cost is minus the product of
    the tolerances. There are three hard constraints (minimum force, maximum force and maximum
    stress) and eight soft constraints keeping each interval ``T +/- t / 2`` within the bounds
    of its variable.
    """

    id: str = "cantilever_4d"
    nominal_tolerance_design: bool = True
    dim: int = 8
    cost: CostFunction = field(default=ToleranceVolume())
    hard: Sequence[HardConstraint] = field(default=[CantileverG1(), CantileverG2(), CantileverG3()])
    soft: Sequence[SoftConstraint] | None = field(
        default=[
            Cantilever4dNominalToleranceGreater(var=0),
            Cantilever4dNominalToleranceLower(var=0),
            Cantilever4dNominalToleranceGreater(var=1),
            Cantilever4dNominalToleranceLower(var=1),
            Cantilever4dNominalToleranceGreater(var=2),
            Cantilever4dNominalToleranceLower(var=2),
            Cantilever4dNominalToleranceGreater(var=3),
            Cantilever4dNominalToleranceLower(var=3),
        ],
    )
    # Rows: nominal values T first, then tolerances t.
    bounds: Tensor = field(
        default=torch.tensor(
            [
                [0.5, 2],
                [5.0, 20],
                [1.0, 3],
                [0.5, 1.5],
                [0.01, 1.5],
                [0.01, 15.0],
                [0.01, 2.0],
                [0.01, 1.0],
            ],
        ),
    )
    aug_bounds: Tensor = field(
        default=torch.tensor(
            [
                [-0.3977, 2.8051],
                [-3.0927, 29.1747],
                [-0.3008, 4.0172],
                [-0.0295, 2.0520],
            ],
        ),
    )
    min_max: Tensor = field(default=torch.tensor([-45, -0.00000001]))
    cp: Tensor = field(default=torch.tensor([[1.26, 1.26, 1.26, 1.26]]))
    cpk: Tensor = field(default=torch.tensor([[1.10, 1.10, 1.10, 1.10]]))
    label_d: list[str] = field(
        default=[
            r"$T_h$",
            r"$T_L$",
            r"$T_b$",
            r"$T_f$",
            r"$t_h$",
            r"$t_L$",
            r"$t_b$",
            r"$t_f$",
        ],
    )
    label_X: list[str] = field(default=[r"$h$", r"$L$", r"$b$", r"$f$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = {h, L, b, f} ~ Normal(T, t / (6 Cp))`` with a quasi-random sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs (nominal values then tolerances), shape
                ``(POP_SIZE, 8)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 4)``. With ``N_SAMPLES == 1`` the single
            quasi-random sample of each design is returned squeezed.
        """
        engine = NormalQMCEngine(d=int(self.dim / 2), seed=self.seed)
        loc = design[:, : int(self.dim / 2)]
        # A tolerance t spans 6 Cp standard deviations.
        scale = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        if N_SAMPLES == 1:
            return self._get_normal(
                N_SAMPLES=N_SAMPLES,
                loc=loc,
                scale=scale,
                engine=engine,
            ).squeeze()
        return self._get_normal(N_SAMPLES=N_SAMPLES, loc=loc, scale=scale, engine=engine)

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = {h, L, b, f}`` for a crude Monte Carlo simulation."""
        torch.manual_seed(self.seed)
        loc = design[:, : int(self.dim / 2)]
        scale = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        return Normal(loc=loc, scale=scale).sample((N_SAMPLES,)).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{h, L, b, f}``: independent Normal variables.

        The means are the nominal values (first half of the design vector) and the standard
        deviations are the tolerances (second half) divided by ``6 Cp``.

        Args:
            design: The 8-component design vector.

        Returns:
            The joint distribution of the four random variables.
        """
        design_squeezed = design.squeeze()
        num_vars = int(self.dim / 2)

        loc_np = design_squeezed[:num_vars].cpu().numpy()
        tolerances = design_squeezed[num_vars:]
        scale_np = (tolerances / (6.0 * self.cp.squeeze())).cpu().numpy()

        marginals = [ot.Normal(loc_np[i], scale_np[i]) for i in range(num_vars)]

        return ot.ComposedDistribution(marginals)


@define(
    slots=True,
    weakref_slot=False,
    frozen=False,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class Cantilever4dMeanShift(BenchmarkBase):
    """Tolerance design of a cantilever beam with four toleranced variables and a mean shift.

    Same design vector, cost, hard and soft constraints as ``Cantilever4d``, with ``Cp = 1.26``
    and ``Cpk = 1.10``. Each random variable (``h``, ``L``, ``b``, ``f``) is the sum of its
    nominal value ``T``, a uniform mean shift and a zero-mean Normal deviation of standard
    deviation ``sigma = t / (6 Cp)``. The half-width of the shift derives from
    ``delta = sigma / 2 * (1 - Cpk / Cp)``.
    """

    id: str = "cantilever_4d_mean_shift"
    nominal_tolerance_design: bool = True
    dim: int = 8
    cost: CostFunction = field(default=ToleranceVolume())
    hard: Sequence[HardConstraint] = field(default=[CantileverG1(), CantileverG2(), CantileverG3()])
    soft: Sequence[SoftConstraint] | None = field(
        default=[
            Cantilever4dNominalToleranceGreater(var=0),
            Cantilever4dNominalToleranceLower(var=0),
            Cantilever4dNominalToleranceGreater(var=1),
            Cantilever4dNominalToleranceLower(var=1),
            Cantilever4dNominalToleranceGreater(var=2),
            Cantilever4dNominalToleranceLower(var=2),
            Cantilever4dNominalToleranceGreater(var=3),
            Cantilever4dNominalToleranceLower(var=3),
        ],
    )
    # Rows: nominal values T first, then tolerances t.
    bounds: Tensor = field(
        default=torch.tensor(
            [
                [0.5, 2],
                [5.0, 20],
                [1.0, 3],
                [0.5, 1.5],
                [0.01, 1.5],
                [0.01, 15.0],
                [0.01, 2.0],
                [0.01, 1.0],
            ],
        ),
    )
    aug_bounds: Tensor = field(
        default=torch.tensor(
            [
                [-0.3977, 2.8051],
                [-3.0927, 29.1747],
                [-0.3008, 4.0172],
                [-0.0295, 2.0520],
            ],
        ),
    )
    min_max: Tensor = field(default=torch.tensor([-45, -0.00000001]))
    cp: Tensor = field(default=torch.tensor([[1.26, 1.26, 1.26, 1.26]]))
    cpk: Tensor = field(default=torch.tensor([[1.10, 1.10, 1.10, 1.10]]))
    label_d: list[str] = field(
        default=[
            r"$T_h$",
            r"$T_L$",
            r"$T_b$",
            r"$T_f$",
            r"$t_h$",
            r"$t_L$",
            r"$t_b$",
            r"$t_f$",
        ],
    )
    label_X: list[str] = field(default=[r"$h$", r"$L$", r"$b$", r"$f$"])

    def get_W(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = T + mean shift + Normal deviation`` with a Sobol sequence.

        Args:
            N_SAMPLES: Number of samples per design.
            design: Population of designs (nominal values then tolerances), shape
                ``(POP_SIZE, 8)``.

        Returns:
            The samples, shape ``(POP_SIZE, N_SAMPLES, 4)``, squeezed when ``N_SAMPLES == 1``.
        """
        POP_SIZE = design.shape[0]
        nominal = design[:, : int(self.dim / 2)]
        # `tolerance` holds the process standard deviation t / (6 Cp), not the tolerance t.
        tolerance = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        # Half-width of the mean shift allowed by the capability indices Cp and Cpk.
        delta = tolerance / 2 * (1 - self.cpk / self.cp)

        if delta.size(0) == 0:
            # Empty population: use a zero shift so that the expressions below still broadcast.
            max_delta = torch.zeros(delta.size()[1:], dtype=delta.dtype, device=delta.device)
            warnings.warn("Empty delta tensor detected in get_W", stacklevel=2)
        else:
            # Largest half-width over the population, per variable: it is applied to all the
            # individuals.
            max_delta = delta.max(dim=0).values

        # One Sobol sequence, with a fixed seed, shared by all the individuals.
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=int(self.dim / 2),
            scramble=True,
            seed=self.seed,
        )
        uniform_samples_base = sobol_engine.draw(N_SAMPLES)

        uniform_samples = uniform_samples_base.unsqueeze(0).expand(POP_SIZE, -1, -1)

        # The same uniform samples drive both terms: they are rescaled to
        # [-max_delta, max_delta] for the mean shift and mapped through the Normal inverse CDF
        # for the deviation around the shifted mean.
        u_samples = max_delta.unsqueeze(0).unsqueeze(0) * (2 * uniform_samples - 1)
        normal_samples = 0.0 + tolerance.unsqueeze(1) * math.sqrt(2) * torch.erfinv(
            2 * uniform_samples - 1,
        )
        if N_SAMPLES == 1:
            return (nominal.unsqueeze(1) + u_samples + normal_samples).squeeze()
        return nominal.unsqueeze(1) + u_samples + normal_samples

    def get_W_MCS(self, N_SAMPLES: int, design: Tensor) -> Tensor:
        """Sample ``w = x = {h, L, b, f}`` for a crude Monte Carlo simulation.

        Unlike in ``get_W``, the uniform mean shift and the Normal deviation are drawn
        independently.
        """
        torch.manual_seed(self.seed)
        nominal = design[:, : int(self.dim / 2)]
        tolerance = design[:, int(self.dim / 2) :] / (6.0 * self.cp)
        delta = tolerance / 2 * (1 - self.cpk / self.cp)
        max_delta = delta.max(dim=0).values
        uniform_samples = Uniform(-1, 1).sample((N_SAMPLES, int(self.dim / 2)))
        normal_samples = Normal(0, 1).sample((N_SAMPLES, int(self.dim / 2)))
        return (
            nominal
            + max_delta.unsqueeze(0) * uniform_samples
            + tolerance.unsqueeze(0) * normal_samples
        ).squeeze()

    def get_probabilistic_model(self, design: Tensor) -> ot.ComposedDistribution:
        """Return the joint distribution of ``{h, L, b, f}``.

        The four variables are independent. Each one is the sum of its nominal value, a Uniform
        mean shift and a Normal process spread::

            w = T + Uniform(-max_delta, +max_delta) + Normal(0, sigma)

        where ``max_delta`` is the largest ``delta`` over the four variables.

        Args:
            design: The 8-component design vector.

        Returns:
            The joint distribution of the four random variables.
        """
        design_squeezed = design.squeeze()
        num_vars = int(self.dim / 2)

        nominal = design_squeezed[:num_vars]
        tolerance = design_squeezed[num_vars:] / (6.0 * self.cp.squeeze())
        delta = tolerance / 2 * (1 - self.cpk.squeeze() / self.cp.squeeze())

        nominal_np = nominal.cpu().numpy()
        tolerance_np = tolerance.cpu().numpy()
        delta_np = delta.cpu().numpy()

        # A single half-width, the largest over the variables, is used for all the shifts.
        max_delta = np.max(delta_np)

        marginals = []
        for i in range(num_vars):
            uniform_dist = ot.Uniform(-max_delta, max_delta)
            normal_dist = ot.Normal(0.0, tolerance_np[i])
            # Adding a Dirac at the nominal value translates the sum of the two random terms.
            final_dist = ot.Dirac(nominal_np[i]) + uniform_dist + normal_dist
            marginals.append(final_dist)

        return ot.ComposedDistribution(marginals)


register_benchmark: dict[str, type[BenchmarkBase]] = {
    "highly_nonlinear": HighlyNonlinear,
    "three_nonlinear": ThreeNonlinear,
    "four_branches": FourBranches,
    "column_under_compression": ColumnUnderCompression,
    "bracket_structure": BracketStructure,
    "truss_bridge": TrussBridge,
    "linear_stackup": LinearStackup,
    "linear_stackup_mean_shift": LinearStackupMeanShift,
    "cantilever_2d": Cantilever2d,
    "cantilever_2d_mean_shift": Cantilever2dMeanShift,
    "cantilever_4d": Cantilever4d,
    "cantilever_4d_mean_shift": Cantilever4dMeanShift,
}
