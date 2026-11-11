"""Sobol sampling of the design space for the initial design of experiments."""

from attrs import define
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize


@define(slots=True, kw_only=True)
class SobolTrainingPoints:
    """Scrambled Sobol points drawn inside the design bounds.

    The same seeded sequence provides first the initial training points of the surrogates and
    then, further along the sequence, any number of additional points.

    Attributes:
        dim: Dimension of the sampled space.
        batch_size: Number of initial points.
        bounds: Bounds of the sampled space, shape ``(dim, 2)`` with columns ``[lower, upper]``.
        seed: Seed of the scrambled Sobol sequence.
    """

    dim: int
    batch_size: int
    bounds: Tensor
    seed: int

    def get_initial_points(self) -> Tensor:
        """Return the first ``batch_size`` points of the sequence, scaled to the bounds."""
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        X_init = sobol.draw(n=self.batch_size)
        return unnormalize(X_init, bounds=self.bounds.t())

    def get_additional_points(self, n_enrich: int) -> Tensor:
        """Return the ``n_enrich`` points that follow the initial ones, scaled to the bounds."""
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        # Skip the initial points so that the new ones do not duplicate them.
        sobol.fast_forward(self.batch_size)
        X_new = sobol.draw(n=n_enrich)
        return unnormalize(X_new, bounds=self.bounds.t())
