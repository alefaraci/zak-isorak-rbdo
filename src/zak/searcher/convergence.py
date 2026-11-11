"""Stagnation-based convergence check of the genetic algorithm."""

import torch

from ..loggers import PolarsLogger


def check_convergence(
    logger: PolarsLogger,
    n_iter: int,
    window: int,
    improvement_max: float,
) -> tuple[bool, str, list[str]]:
    """Check whether the genetic algorithm has stagnated over the last generations.

    The best and the median fitness of the population are read from the logger over the last
    ``window`` generations. For each of them, the normalized improvement is the absolute change
    between the first and the last value of the window, divided by the largest absolute value in
    the window. The search has stagnated when both improvements are below ``improvement_max``.

    Args:
        logger: Logger holding the history of the run (columns ``pop_best_eval`` and
            ``median_eval``).
        n_iter: Index of the current generation.
        window: Number of generations of the stagnation window.
        improvement_max: Threshold on the normalized improvement.

    Returns:
        A tuple ``(converged, msg, log)``: whether the search has stagnated, the message to
        display when it has (empty otherwise) and the lines detailing the check.
    """
    # Not enough generations to fill the window yet.
    if n_iter <= window:
        return False, "", [""]

    evolution = logger.to_dataframe()
    histories = {
        "best": torch.tensor(evolution["pop_best_eval"][-window:]),
        "median": torch.tensor(evolution["median_eval"][-window:]),
    }

    normalized_improvements, store_log = {}, []

    for key, hist in histories.items():
        # Normalizing by the largest absolute value of the window, rather than by the first or
        # the last value, keeps the ratio meaningful when the fitness changes sign or is close
        # to zero.
        hist_abs_max = hist.abs().max().clamp(min=1e-8)
        abs_change = (hist[-1] - hist[0]).abs()
        normalized_improvements[key] = abs_change / hist_abs_max
        store_log.append(
            f"{key}: first={hist[0]:.6f}, last={hist[-1]:.6f}, abs_change={abs_change:.6f}, abs_max={hist_abs_max:.6f}\n",
        )

    norm_improvement_tensor = torch.tensor(list(normalized_improvements.values()))
    max_norm_improvement = norm_improvement_tensor.max()

    store_log.append(
        f"Max_normalized_improvement: {max_norm_improvement * 100:.3f} % vs threshold: {improvement_max * 100:.3f} %\n",
    )

    store_log.append(
        f"Normalized improvements: best={normalized_improvements['best']:.6f}, "
        f"median={normalized_improvements['median']:.6f}",
    )

    store_log[:0] = ["\n"]

    if max_norm_improvement < improvement_max:
        msg = f"Converged for stagnation at generation {n_iter}! Normalized improvement: {max_norm_improvement:.6f}"
        return True, msg, store_log

    return False, "", store_log
