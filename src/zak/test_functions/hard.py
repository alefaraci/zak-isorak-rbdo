"""Limit-state functions g_k of the benchmark problems (the "hard" constraints).

Sign convention: a limit state fails when g <= 0 and is safe when g > 0. The reliability
constraints of the optimization bound the probability of the event g_k <= 0.

Every function takes a tensor whose last dimension stores the input variables and returns one
value per point. The limit states that depend on random parameters besides the design variables
accept either the full augmented vector (design variables followed by the random parameters) or
the design variables alone; in the second case the parameters are fixed, which is how the limit
state is evaluated in the design space.
"""

from abc import ABC, abstractmethod

from attrs import field, define
from torch import Tensor
import torch


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class HardConstraint(ABC):
    """Base class of the limit-state functions.

    Attributes:
        seed: Optional random seed; the analytical limit states of this module do not use it.
    """

    seed: int | None = field(default=None, repr=False)

    @abstractmethod
    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the limit state at the points stored along the last dimension of X.

        Returns:
            One value per point: failure when the value is <= 0.
        """


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class HighlyNonlinearG1(HardConstraint):
    """First limit state of the highly nonlinear problem (Haupt problem).

    g1(x) = -x1 * sin(4 * x1) - 1.1 * x2 * sin(2 * x2)
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g1 of the highly nonlinear problem."""
        x1, x2 = X[..., 0], X[..., 1]
        return -x1 * torch.sin(4 * x1) - 1.1 * x2 * torch.sin(2 * x2)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class HighlyNonlinearG2(HardConstraint):
    """Second limit state of the highly nonlinear problem (Haupt problem).

    g2(x) = x1 + x2 - 3
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g2 of the highly nonlinear problem."""
        x1, x2 = X[..., 0], X[..., 1]
        return x1 + x2 - 3


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ThreeNonlinearG1(HardConstraint):
    """First limit state of the three nonlinear problem (Choi problem).

    g1(x) = x1^2 * x2 / 20 - 1
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g1 of the three nonlinear problem."""
        x1, x2 = X[..., 0], X[..., 1]
        return (x1**2 * x2) / 20 - 1


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ThreeNonlinearG2(HardConstraint):
    """Second limit state of the three nonlinear problem (Choi problem).

    g2(x) = (x1 + x2 - 5)^2 / k + (x1 - x2 - 12)^2 / 120 - 1,    k = 30
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g2 of the three nonlinear problem."""
        # Two variants of this limit state exist: k = 30 (UQLab), used here, and k = 5 (Kim).
        x1, x2, k = X[..., 0], X[..., 1], 30
        return ((x1 + x2 - 5) ** 2) / k + ((x1 - x2 - 12) ** 2) / 120 - 1


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ThreeNonlinearG3(HardConstraint):
    """Third limit state of the three nonlinear problem (Choi problem).

    g3(x) = 80 / (x1^2 + 8 * x2 + 5) - 1
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g3 of the three nonlinear problem."""
        x1, x2 = X[..., 0], X[..., 1]
        return 80 / (x1**2 + 8 * x2 + 5) - 1


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class FourBranchesG(HardConstraint):
    """Limit state of the four-branch series system.

    g(x) = min(b1, b2, b3, b4), with k = 7 and

    b1 = 3 + 0.1 * (x1 - x2)^2 - (x1 + x2) / sqrt(2)
    b2 = 3 + 0.1 * (x1 - x2)^2 + (x1 + x2) / sqrt(2)
    b3 = (x1 - x2) + k / sqrt(2)
    b4 = (x2 - x1) + k / sqrt(2)
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the limit state of the four-branch series system."""
        x1, x2, k = X[..., 0], X[..., 1], 7
        g1 = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / torch.sqrt(torch.tensor(2))
        g2 = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / torch.sqrt(torch.tensor(2))
        g3 = (x1 - x2) + k / torch.sqrt(torch.tensor(2))
        g4 = (x2 - x1) + k / torch.sqrt(torch.tensor(2))
        # Series system: it fails as soon as one branch fails, hence the minimum.
        return torch.stack((g1, g2, g3, g4)).min(dim=0)[0]


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ColumnUnderCompressionDetermG(HardConstraint):
    """Limit state of the column under compression with fixed parameters.

    Euler buckling load of a rectangular column, of cross-section b x h, minus the service load:

    g(b, h) = k * pi^2 * E * b * h^3 / (12 * L^2) - F_ser

    with k = 0.6, E = 1e4, L = 3e3 and F_ser = 1.4622e6.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the limit state from the design variables (b, h) only."""
        x1, x2 = X[..., 0], X[..., 1]
        k, E, L, Fser = 0.6, 1e4, 3.0e3, 1.4622e6
        return k * torch.pi**2 * E * x1 * x2**3 / (12 * L**2) - Fser


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class ColumnUnderCompressionG(HardConstraint):
    """Limit state of the column under compression with random parameters.

    Euler buckling load of a rectangular column, of cross-section b x h, minus the service load:

    g(b, h, k, E, L) = k * pi^2 * E * b * h^3 / (12 * L^2) - F_ser,    F_ser = 1.4622e6
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the limit state.

        Args:
            X: Either the augmented points (b, h, k, E, L) or the design variables (b, h) alone;
                in the second case k, E and L take the values of `ColumnUnderCompressionDetermG`.
        """
        try:
            x1, x2, k, E, L = X[..., 0], X[..., 1], X[..., 2], X[..., 3], X[..., 4]
            Fser = 1.4622e6
            return k * torch.pi**2 * E * x1 * x2**3 / (12 * L**2) - Fser
        except:
            # Indexing the missing columns fails when X holds the design variables only: the
            # limit state is then evaluated with fixed parameters.
            post_process_class = ColumnUnderCompressionDetermG()
            return post_process_class.fn(X)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class BracketDetermG1(HardConstraint):
    """First limit state of the bracket structure with fixed parameters.

    Yield stress minus the maximum bending stress in the beam CD:

    g1 = sy - sigma_B,    sigma_B = 6 * M_B / (w_CD * t^2),
    M_B = P * L / 3 + rho * 9.81 * w_CD * t * L^2 / 18

    with P = 100 kN, sy = 225 MPa, rho = 7860 kg/m^3 and L = 5 m. The powers of ten in the code
    convert the self-weight from N to kN and the stress from kPa to MPa.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g1 from the design variables (w_AB, w_CD, t) only, given in centimetres."""
        # The design variables are converted from centimetres to metres.
        _, wCD, t = X[..., 0] * 1e-2, X[..., 1] * 1e-2, X[..., 2] * 1e-2
        P, E, sy, rho, L = 100, 200, 225, 7860, 5
        E = E * 1e3
        MB = P * L / 3 + 1e-3 * rho * 9.81 * wCD * t * L**2 / 18
        sigma_b = 6 * MB / (wCD * t**2)
        return sy - sigma_b * 1e-3


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class BracketDetermG2(HardConstraint):
    """Second limit state of the bracket structure with fixed parameters.

    Euler buckling load of the member AB minus the axial force it carries:

    g2 = F_b - F_AB,    F_b = pi^2 * E * t * w_AB^3 / (12 * L_AB^2),
    L_AB = 2 * L / (3 * sin(theta)),
    F_AB = (3 * P / 2 + 3 * rho * 9.81 * w_CD * t * L / 4) / cos(theta)

    with theta = pi / 3, P = 100 kN, E = 200 GPa, rho = 7860 kg/m^3 and L = 5 m. The powers of
    ten in the code express both forces in kN.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g2 from the design variables (w_AB, w_CD, t) only, given in centimetres."""
        wAB, wCD, t = X[..., 0] * 1e-2, X[..., 1] * 1e-2, X[..., 2] * 1e-2
        P, E, _, rho, L = 100, 200, 225, 7860, 5
        E = E * 1e3
        theta = torch.tensor(torch.pi / 3)
        FAB = (3 * P / 2 + 1e-3 * 3 * rho * 9.81 * wCD * t * L / 4) / torch.cos(theta)
        Fb = 1e3 * torch.pi**2 * E * t * wAB**3 / (12 * (2 * L / (3 * torch.sin(theta))) ** 2)
        return Fb - FAB


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class BracketG1(HardConstraint):
    """First limit state of the bracket structure with random parameters.

    Yield stress minus the maximum bending stress in the beam CD:

    g1 = sy - sigma_B,    sigma_B = 6 * M_B / (w_CD * t^2),
    M_B = P * L / 3 + rho * 9.81 * w_CD * t * L^2 / 18

    with P in kN, sy in MPa, rho in kg/m^3 and L in m. The powers of ten in the code convert the
    self-weight from N to kN and the stress from kPa to MPa.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g1.

        Args:
            X: Either the augmented points (w_AB, w_CD, t, P, E, sy, rho, L) or the design
                variables (w_AB, w_CD, t) alone; in the second case the parameters take the
                values of `BracketDetermG1`. The design variables are given in centimetres.
        """
        try:
            # The design variables are converted from centimetres to metres.
            _, wCD, t = X[..., 0] * 1e-2, X[..., 1] * 1e-2, X[..., 2] * 1e-2
            P, E, sy, rho, L = X[..., 3], X[..., 4], X[..., 5], X[..., 6], X[..., 7]
            E = E * 1e3
            MB = P * L / 3 + 1e-3 * rho * 9.81 * wCD * t * L**2 / 18
            return sy - 6 * MB / (wCD * t**2) * 1e-3
        except:
            # Indexing the missing columns fails when X holds the design variables only: the
            # limit state is then evaluated with fixed parameters.
            post_process_class = BracketDetermG1()
            return post_process_class.fn(X)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class BracketG2(HardConstraint):
    """Second limit state of the bracket structure with random parameters.

    Euler buckling load of the member AB minus the axial force it carries:

    g2 = F_b - F_AB,    F_b = pi^2 * E * I / L_AB^2,    I = t * w_AB^3 / 12,
    L_AB = 2 * L / (3 * sin(theta)),
    F_AB = (3 * P / 2 + 3 * rho * 9.81 * t * w_CD * L / 4) / cos(theta)

    with theta = pi / 3, P in kN, E in GPa, rho in kg/m^3 and L in m. The powers of ten in the
    code express both forces in kN.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g2.

        Args:
            X: Either the augmented points (w_AB, w_CD, t, P, E, sy, rho, L) or the design
                variables (w_AB, w_CD, t) alone; in the second case the parameters take the
                values of `BracketDetermG2`. The design variables are given in centimetres.
        """
        try:
            # The design variables are converted from centimetres to metres.
            wAB, wCD, t = X[..., 0] * 1e-2, X[..., 1] * 1e-2, X[..., 2] * 1e-2
            P, E, _, rho, L = X[..., 3], X[..., 4], X[..., 5], X[..., 6], X[..., 7]
            E = E * 1e3
            theta = torch.tensor(torch.pi / 3)
            inertia = t * wAB**3 / 12
            LAB = 2 * L / (3 * torch.sin(theta))
            return 1e3 * torch.pi**2 * E * inertia / LAB**2 - 1 / torch.cos(theta) * (
                3 * P / 2 + (3 * 1e-3 * rho * 9.81 * t * wCD * L) / 4
            )
        except:
            # Indexing the missing columns fails when X holds the design variables only: the
            # limit state is then evaluated with fixed parameters.
            post_process_class = BracketDetermG2()
            return post_process_class.fn(X)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class TrussBridgeBaseProblem(HardConstraint):
    """Base class of the truss-bridge limit states: closed-form displacement of the truss.

    The truss is made of horizontal bars (cross-section A1, Young's modulus E1) and of diagonal
    bars (cross-section A2, Young's modulus E2), and carries six vertical loads P1, ..., P6.
    """

    def get_displacement_wmax(self, x1, x2, E1, E2, P1, P2, P3, P4, P5, P6) -> Tensor:
        """Evaluate the maximum vertical displacement of the truss, in closed form.

        The four terms computed below sum up to

        w_max = -sqrt(2) * (2 * P1 + 6 * P2 + 10 * P3 + 10 * P4 + 6 * P5 + 2 * P6) / (x2 * E2)
                - (36 * P1 + 100 * P2 + 140 * P3 + 140 * P4 + 100 * P5 + 36 * P6) / (x1 * E1)

        Args:
            x1: Cross-section A1 of the horizontal bars.
            x2: Cross-section A2 of the diagonal bars.
            E1: Young's modulus of the horizontal bars.
            E2: Young's modulus of the diagonal bars.
            P1: First vertical load.
            P2: Second vertical load.
            P3: Third vertical load.
            P4: Fourth vertical load.
            P5: Fifth vertical load.
            P6: Sixth vertical load.

        Returns:
            The displacement, negative for positive loads.
        """
        numerator1 = -2 * torch.sqrt(torch.tensor(2.0)) * x1 * P1
        numerator2 = -6 * torch.sqrt(torch.tensor(2.0)) * x1 * P2
        numerator3 = -10 * torch.sqrt(torch.tensor(2.0)) * x1 * P3
        numerator4 = -10 * torch.sqrt(torch.tensor(2.0)) * x1 * P4
        denominator1 = x1 * x2 * E2
        term1 = (numerator1 + numerator2 + numerator3 + numerator4) / denominator1
        numerator5 = -6 * torch.sqrt(torch.tensor(2.0)) * x1 * P5
        numerator6 = -2 * torch.sqrt(torch.tensor(2.0)) * x1 * P6
        term2 = (numerator5 + numerator6) / denominator1
        numerator7 = -36 * x2 * E2 * P1 - 100 * x2 * E2 * P2
        denominator2 = x1 * x2 * E1 * E2
        term3 = numerator7 / denominator2
        numerator8 = (
            -140 * x2 * E2 * P3 - 140 * x2 * E2 * P4 - 100 * x2 * E2 * P5 - 36 * x2 * E2 * P6
        )
        term4 = numerator8 / denominator2
        return term1 + term2 + term3 + term4

    def fn(self, X: Tensor) -> Tensor:
        """Not implemented in the base class."""
        err_msg = "Method not implemented."
        raise NotImplementedError(err_msg)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class TrussBridgeDetermG(TrussBridgeBaseProblem):
    """Limit state of the truss bridge with fixed parameters.

    g(A1, A2) = 10 - 100 * |w_max|

    where w_max is the displacement of `get_displacement_wmax`, evaluated with the two Young's
    moduli equal to 2.1e11 and the six loads equal to 5e4.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the limit state from the design variables (A1, A2) only."""
        x1, x2 = X[..., 0], X[..., 1]
        Ei = torch.tensor(2.1 * 1e11).expand(x1.shape).to(x1.device)
        Pi = torch.tensor(5 * 1e4).expand(x1.shape).to(x1.device)
        wmax = self.get_displacement_wmax(x1, x2, Ei, Ei, Pi, Pi, Pi, Pi, Pi, Pi)
        return 10 - wmax.abs() * 100


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class TrussBridgeG(TrussBridgeBaseProblem):
    """Limit state of the truss bridge with random parameters.

    g(A1, A2, E1, E2, P1, ..., P6) = 10 - 100 * |w_max|

    where w_max is the displacement of `get_displacement_wmax`. The factor 100 expresses a
    displacement computed in metres in centimetres, to be compared with the limit of 10.
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate the limit state.

        Args:
            X: Either the augmented points (A1, A2, E1, E2, P1, ..., P6) or the cross-sections
                (A1, A2) alone; in the second case the parameters take the values of
                `TrussBridgeDetermG`.
        """
        try:
            x1, x2, E1, E2, P1, P2, P3, P4, P5, P6 = (
                X[..., 0],
                X[..., 1],
                X[..., 2],
                X[..., 3],
                X[..., 4],
                X[..., 5],
                X[..., 6],
                X[..., 7],
                X[..., 8],
                X[..., 9],
            )
            wmax = self.get_displacement_wmax(x1, x2, E1, E2, P1, P2, P3, P4, P5, P6)
            return 10 - wmax.abs() * 100
        except:
            # Indexing the missing columns fails when X holds the cross-sections only: the limit
            # state is then evaluated with fixed parameters.
            post_process_class = TrussBridgeDetermG()
            return post_process_class.fn(X)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class StackupG1(HardConstraint):
    """First limit state of the linear stack-up problems: lower bound on the stack-up.

    g1(x) = x1 + x2 - 9.5
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g1 of the linear stack-up problems."""
        x1, x2 = X[..., 0], X[..., 1]
        return x1 + x2 - 9.5


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class StackupG2(HardConstraint):
    """Second limit state of the linear stack-up problems: upper bound on the stack-up.

    g2(x) = 10.5 - (x1 + x2)
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g2 of the linear stack-up problems."""
        x1, x2 = X[..., 0], X[..., 1]
        return -(x1 + x2) + 10.5


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class CantileverBaseProblem(HardConstraint):
    """Base class of the limit states of the cantilever tolerance-design problems.

    Euler-Bernoulli cantilever beam of length L and rectangular cross-section b x h, whose tip is
    deflected by f. The force needed at the tip and the maximum bending stress are

    F = 3 * E * I * f / L^3,    I = b * h^3 / 12,    sigma = 3 * E * f * h / (2 * L^2)

    The limit states bound the tip force from below and from above, and the bending stress from
    above. They accept the points (h, L), with b and f fixed to the attributes below, or the
    points (h, L, b, f).

    Attributes:
        module: Young's modulus E.
        effort_min: Lower bound on the tip force.
        effort_max: Upper bound on the tip force.
        stress_max: Upper bound on the bending stress.
        f: Tip deflection used with two-dimensional points.
        b: Width of the cross-section used with two-dimensional points.
    """

    module: float = 1e4
    effort_min: float = 5.0
    effort_max: float = 25.0
    stress_max: float = 100.0
    f: float = 0.5
    b: float = 3.0

    def fn(self, X: Tensor) -> Tensor:
        """Not implemented in the base class."""
        err_msg = "Method not implemented."
        raise NotImplementedError(err_msg)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class CantileverG1(CantileverBaseProblem):
    """First limit state of the cantilever beam: lower bound on the tip force.

    g1 = F - effort_min,    F = 3 * E * b * h^3 * f / (12 * L^3)
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g1 at the points (h, L) or (h, L, b, f).

        Raises:
            ValueError: If the last dimension of X is neither 2 nor 4.
        """
        if X.shape[-1] == 2:
            x1, x2 = X[..., 0], X[..., 1]  # h, L
            inertia = (self.b * x1**3) / 12
            F = (3 * self.module * inertia * self.f) / (x2**3)
            return F - self.effort_min
        if X.shape[-1] == 4:
            x1, x2, x3, x4 = X[..., 0], X[..., 1], X[..., 2], X[..., 3]  # h, L, b, f
            return (3 * self.module * x3 * x1**3 * x4) / (12 * x2**3) - self.effort_min
        msg = "Invalid input shape for X to compute CantileverG1."
        raise ValueError(msg)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class CantileverG2(CantileverBaseProblem):
    """Second limit state of the cantilever beam: upper bound on the tip force.

    g2 = effort_max - F,    F = 3 * E * b * h^3 * f / (12 * L^3)
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g2 at the points (h, L) or (h, L, b, f).

        Raises:
            ValueError: If the last dimension of X is neither 2 nor 4.
        """
        if X.shape[-1] == 2:
            x1, x2 = X[..., 0], X[..., 1]  # h, L
            inertia = (self.b * x1**3) / 12
            F = (3 * self.module * inertia * self.f) / (x2**3)
            return self.effort_max - F
        if X.shape[-1] == 4:
            x1, x2, x3, x4 = X[..., 0], X[..., 1], X[..., 2], X[..., 3]  # h, L, b, f
            return self.effort_max - (3 * self.module * x3 * x1**3 * x4) / (12 * x2**3)
        msg = "Invalid input shape for X to compute CantileverG2."
        raise ValueError(msg)


@define(
    slots=True,
    weakref_slot=False,
    frozen=True,
    hash=False,
    cache_hash=False,
    kw_only=True,
    eq=False,
)
class CantileverG3(CantileverBaseProblem):
    """Third limit state of the cantilever beam: upper bound on the bending stress.

    g3 = stress_max - sigma,    sigma = 3 * E * f * h / (2 * L^2)
    """

    def fn(self, X: Tensor) -> Tensor:
        """Evaluate g3 at the points (h, L) or (h, L, b, f).

        Raises:
            ValueError: If the last dimension of X is neither 2 nor 4.
        """
        if X.shape[-1] == 2:
            x1, x2 = X[..., 0], X[..., 1]  # h, L
            stress = (3 * self.module * self.f * x1) / (2 * x2**2)
            return self.stress_max - stress
        if X.shape[-1] == 4:
            x1, x2, _, x4 = X[..., 0], X[..., 1], X[..., 2], X[..., 3]  # h, L, b, f
            stress = (3 * self.module * x4 * x1) / (2 * x2**2)
            return self.stress_max - stress
        msg = "Invalid input shape for X to compute CantileverG3."
        raise ValueError(msg)
