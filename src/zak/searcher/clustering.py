"""DBSCAN filter rejecting candidate training points too close to the existing ones."""

from torch import Tensor
from sklearn.cluster import DBSCAN
import torch

from ..benchmarks import BenchmarkBase


def not_in_cluster(
    DoE_physical: Tensor,
    x_new_physical: Tensor,
    design: Tensor,
    benchmark: BenchmarkBase,
    min_samples: int,
    eps: float,
) -> bool:
    """Tell whether a candidate training point lies outside every cluster of existing points.

    The design of experiments and the candidate are mapped to the standard-normal U-space with
    the probabilistic model of the benchmark at ``design`` and clustered together with DBSCAN.
    The candidate is worth adding when DBSCAN labels it as noise, i.e. when it does not belong
    to a dense group of points.

    Args:
        DoE_physical: Training points of one surrogate in the augmented physical space, shape
            ``(n, n_dims)``.
        x_new_physical: Candidate point in the same space, shape ``(n_dims,)`` or
            ``(1, n_dims)``.
        design: Design defining the probabilistic model used for the transformation, shape
            ``(1, dim)``.
        benchmark: Benchmark providing the transformation to the U-space.
        min_samples: Number of neighbors (the point included) that makes a point a DBSCAN core
            point.
        eps: DBSCAN neighborhood radius, measured in the U-space.

    Returns:
        True if the candidate is not part of any cluster, False otherwise.
    """
    if DoE_physical.dim() == 1:
        DoE_physical = DoE_physical.unsqueeze(0)
    if x_new_physical.dim() == 1:
        x_new_physical = x_new_physical.unsqueeze(0)

    # The candidate is appended last, so that its label is the last one returned by DBSCAN.
    points_physical = torch.cat([DoE_physical, x_new_physical], dim=0)
    points_u_space = benchmark.transform_to_u_space(points_physical, design)
    points_u_space_np = points_u_space.cpu().numpy()
    # The U-space is already standardized: distances are comparable along every dimension and
    # no further scaling is needed before clustering.
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(points_u_space_np)
    new_point_label = labels[-1]
    # DBSCAN labels noise points with -1.
    return new_point_label == -1
