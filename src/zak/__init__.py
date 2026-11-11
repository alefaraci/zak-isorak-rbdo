"""zak: zeroth-order adaptive-Kriging algorithms for Reliability-Based Design Optimization.

The package implements two algorithms, ZAK-RBDO (Zeroth-order Adaptive Kriging RBDO) and
ISORAK-RBDO (Isocost-constrained Regions Adaptive Kriging RBDO). Both couple a genetic algorithm
with Kriging surrogates of the limit-state functions that are enriched adaptively during the
search. The code is organised in the following sub-packages:

* ``benchmarks``: definition of the benchmark problems;
* ``test_functions``: cost functions, hard (reliability) and soft (deterministic) constraints;
* ``random_vars``: sampling of the training points;
* ``gaussianprocess``: Kriging surrogates built with BoTorch;
* ``problem``: the optimization problem and the stopping criteria of the enrichment;
* ``searcher``: the genetic algorithm;
* ``loggers``: per-generation logging of the search;
* ``plotter``: post-processing figures and animations;
* ``utils``: console output, input/output and other helpers.

Copyright (c) 2024-2025 Alessio Faraci. MIT License.
"""

from . import (
    utils,
    loggers,
    plotter,
    problem,
    searcher,
    benchmarks,
    random_vars,
    test_functions,
    gaussianprocess,
)
from .utils import console, progressbar

__all__ = [
    "benchmarks",
    "console",
    "gaussianprocess",
    "loggers",
    "plotter",
    "problem",
    "progressbar",
    "random_vars",
    "searcher",
    "test_functions",
    "utils",
]
__version__ = "0.1.0"
