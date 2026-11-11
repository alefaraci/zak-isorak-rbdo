"""Compare ZAK-RBDO and ISORAK-RBDO on one benchmark.

Usage (from any directory)::

    python examples/postprocess_comparison.py --benchmark three_nonlinear

Both analyses must have been run on the benchmark, ideally with several seeds. The last generation
of every run is collected and the two algorithms are compared with box plots of the optimal
design, the cost, the failure probabilities and the number of limit-state function calls, saved
in ``examples/results/comparison/<benchmark>/``.
"""

import os
import argparse

from zak import utils
from zak.plotter import ModelComparison
from zak.plotter.boxplot_comparator import load_benchmark_data

HERE = utils.Path(__file__).parent


def main() -> None:
    """Generate the comparison box plots."""
    parser = argparse.ArgumentParser(description="ZAK-RBDO versus ISORAK-RBDO box plots")
    parser.add_argument("--benchmark", type=str, default="three_nonlinear", help="benchmark id")
    benchmark_id = parser.parse_args().benchmark

    os.chdir(HERE)
    comparison = ModelComparison(
        data1=load_benchmark_data(analysis="zakrbdo", benchmark_id=benchmark_id, base_path=HERE),
        data2=load_benchmark_data(analysis="isorakrbdo", benchmark_id=benchmark_id, base_path=HERE),
        benchmark_id=benchmark_id,
        output_path=HERE / "results" / "comparison",
    )
    config = utils.load_yaml(path=HERE, filename=f"{benchmark_id}.yaml")["ZAKRBDO"]
    comparison.plot_all(pf_thr=config["rbdo"]["Pf_thr"], plot_g_calls=True, show_plots=False)


if __name__ == "__main__":
    main()
