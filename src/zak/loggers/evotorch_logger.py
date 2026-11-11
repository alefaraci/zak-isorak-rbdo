"""EvoTorch logger that stores the status of the search in a Polars DataFrame.

One row is recorded per generation; the rows are saved as newline-delimited JSON.
"""

from copy import deepcopy
from typing import Any

from evotorch.logging import Logger
import torch
import polars as pl

from zak import utils


class PolarsLogger(Logger):
    """EvoTorch logger that keeps the status of every generation in memory.

    The status is the dictionary exposed by the searcher (best design, cost, failure
    probabilities, design of experiments, ...); its content depends on the analysis.
    """

    def __init__(self, Searcher) -> None:
        """Attach the logger to a searcher.

        Args:
            Searcher: The search algorithm to log.
        """
        super().__init__(Searcher)
        self._data = []  # pyright: ignore

    def _log(self, status: dict[str, Any]) -> None:
        """Store a copy of the status of the current generation."""
        self._data.append(deepcopy(status))

    def to_dataframe(self) -> pl.DataFrame:
        """Return the recorded statuses as a DataFrame, one row per generation.

        Tensors are converted to lists; the other values are kept as they are.
        """
        processed_data = []
        for entry in self._data:
            processed_entry = {}
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    processed_entry[key] = value.cpu().detach().tolist()
                else:
                    processed_entry[key] = value
            processed_data.append(processed_entry)
        return pl.DataFrame(processed_data)

    def postprocess_dataframe(self) -> pl.DataFrame:
        """Return the recorded statuses in the form that is written to disk.

        The columns that are only displayed during the search are dropped, the infinite values of
        "eps_max" are replaced and the remaining objects are converted to lists of floats.
        """
        results = self.to_dataframe()
        columns_to_drop = [
            "pop_evals_stdev",
            "cost",
            "rho_alpha",
            "admissibility",
        ]
        results = results.select(pl.exclude(columns_to_drop))

        # Only the surrogate-based analyses log a "doe" column, and "eps_max" with it. Infinite
        # values of "eps_max" are replaced by the largest finite one.
        if "doe" in results.columns:
            max_finite_value = results.select(
                pl.col("eps_max").filter(pl.col("eps_max").is_finite()).max(),
            ).item()
            results = results.with_columns(
                pl.col("eps_max").clip(upper_bound=max_finite_value).alias("eps_max"),
            )
        if "individuals" in results.columns:
            results = results.with_columns(
                pl.col("individuals").map_elements(
                    lambda x: x.cpu().numpy().tolist() if isinstance(x, torch.Tensor) else x,
                    return_dtype=pl.List(pl.List(pl.Float64)),
                ),
            )

        # The best individual is logged as an EvoTorch solution: only its values are kept.
        if "pop_best" in results.columns:
            results = results.with_columns(
                pl.col("pop_best").map_elements(
                    lambda x: x.values.cpu().numpy().tolist() if hasattr(x, "values") else x,
                    return_dtype=pl.List(pl.Float64),
                ),
            )

        if "g_calls" in results.columns:
            results = results.with_columns(
                pl.col("g_calls").map_elements(
                    lambda x: x.cpu().numpy().tolist() if isinstance(x, torch.Tensor) else x,
                    return_dtype=pl.List(pl.Float64),
                ),
            )

        if "pop_init_alpha" in results.columns:
            results = results.with_columns(
                pl.col("pop_init_alpha").map_elements(
                    lambda x: x.cpu().numpy().tolist() if isinstance(x, torch.Tensor) else x,
                    return_dtype=pl.List(pl.List(pl.Float64)),
                ),
            )

        return results

    def save(self, path: utils.Path) -> None:
        """Write the post-processed statuses to a newline-delimited JSON file.

        Args:
            path: Path of the output file.
        """
        results = self.postprocess_dataframe()
        results.write_ndjson(path)


def save_loggers_pl(loggers_pl: list[PolarsLogger], path: utils.Path) -> None:
    """Write the statuses of several loggers to a single newline-delimited JSON file.

    ISORAK-RBDO runs one search, hence one logger, per isocost region; their rows are concatenated
    in the order of the list.

    Args:
        loggers_pl: The loggers, in chronological order.
        path: Path of the output file.
    """
    processed_loggers_pl = [dfs.postprocess_dataframe() for dfs in loggers_pl]
    processed_loggers_pl = [df for df in processed_loggers_pl if not df.is_empty()]
    # The columns are sorted so that every DataFrame has the same column order.
    results = pl.concat([df.select(sorted(df.columns)) for df in processed_loggers_pl])
    results.write_ndjson(path)
