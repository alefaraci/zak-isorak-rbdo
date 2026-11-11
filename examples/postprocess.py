"""Post-process the results written by ``run.py``.

Usage (from any directory)::

    # one run: plots, evolution frames and videos
    python examples/postprocess.py --analysis zakrbdo --benchmark four_branches \
        --timestamp 2025_06_24_235730_seed27

    # every run of a benchmark: plots for all, animations for the first one,
    # plus box plots and the median-results table across seeds
    python examples/postprocess.py --analysis zakrbdo --benchmark four_branches

    # every benchmark of an analysis
    python examples/postprocess.py --analysis zakrbdo

Figures are saved as PDF inside each run directory (``pdf/`` and ``evolution/``). Rendering them
requires a LaTeX installation; turning the evolution frames into videos also requires ImageMagick
and ffmpeg (see ``run_animator.sh``).
"""

import os

import polars as pl

from zak import utils, console
from zak.utils import Path, Table
from zak.plotter import boxplot as bxp
from zak.plotter.postprocess import PostProcessing

HERE = utils.Path(__file__).parent


def get_path(analysis: str, benchmark_id: str, timestamp: str) -> Path:
    """Get the path to the results directory for a given analysis, benchmark ID, and timestamp."""
    return HERE / "results" / analysis / benchmark_id / timestamp


def post_process(
    analysis: str,
    benchmark_id: str,
    timestamp: str,
    *,
    animation: bool,
) -> tuple[str, tuple[Table, Table] | None, Path]:
    """Process a single benchmark result."""
    path = HERE / "results" / analysis / benchmark_id / timestamp
    log_str = (
        f"[cyan]Analysis[/]: [magenta]{analysis}[/] - "
        f"[cyan]Benchmark ID[/]: [green]{benchmark_id}[/] - "
        f"[cyan]Time stamp[/]: [yellow]{timestamp}\n\n"
    )
    results = pl.read_ndjson(path / "results.json")
    loader = utils.load(path=path, filename="opt_problem")

    post_production = PostProcessing(path=path, problem=loader["opt_problem"], results=results)
    log_str += post_production.optimal_design()
    tables = post_production.generate_plots(show=False)
    if animation:
        post_production.generate_animations(show=False)
        post_production.generate_animations_videos(
            analysis=analysis,
            benchmark=benchmark_id,
            timestamp=timestamp,
        )

    return log_str, tables, path


def post_process_median(analysis: str, benchmark_id: str) -> None:
    """Process median evaluation results for a single benchmark."""
    try:
        bid_dir = HERE / "results" / analysis / benchmark_id
        timestamps = [
            timestamp_dir.name for timestamp_dir in bid_dir.iterdir() if timestamp_dir.is_dir()
        ]
        results = pl.concat(
            [
                df.select(sorted(df.columns))
                for df in [pl.read_ndjson(bid_dir / ts / "results.json")[-1] for ts in timestamps]
            ],
        )
        bxp.box_plot(analysis=analysis, dfs=results, bench_id=benchmark_id, path=bid_dir)
    except Exception as e:
        console.log(f"[red bold]Error processing {analysis}: {e}[/]")


def post_process_median_all_benchmarks(analysis: str) -> None:
    """Process median evaluation results for all the benchmarks."""
    analysis_dir = HERE / "results" / analysis
    benchmarks_dir_names = [
        benchmark_dir.name for benchmark_dir in analysis_dir.iterdir() if benchmark_dir.is_dir()
    ]
    for benchmark_id in benchmarks_dir_names:
        try:
            post_process_median(analysis, benchmark_id)
        except Exception as e:
            console.log(f"[red bold]Error processing {analysis}: {e}[/]")


def post_process_single_benchmark(analysis: str, benchmark_id: str, timestamp: str) -> None:
    """Postprocess only a single benchmark with a defined timestamp."""
    msg = f"[cyan]Processing animations for {analysis} | {benchmark_id} | {timestamp} ...\n"
    utils.console.log(msg)
    log_str, tables, path = post_process(
        analysis=analysis,
        benchmark_id=benchmark_id,
        timestamp=timestamp,
        animation=True,
    )
    utils.console_direct.log(log_str)
    if tables:
        [utils.console_direct.log(tab) for tab in tables]
    utils.log_console(path=path, filename="postprocess.log")


def post_process_all_timestamps(analysis: str, benchmark_id: str) -> None:
    """Postprocess all timestamps for a single benchmark."""
    bid_dir = HERE / "results" / analysis / benchmark_id
    timestamps = [
        timestamp_dir.name for timestamp_dir in bid_dir.iterdir() if timestamp_dir.is_dir()
    ]
    animate_first = True
    with utils.live:
        task_id = utils.progressbar.add_task(
            "[bold white]📸 [bold light_coral]Post-Processing: ",
            total=len(timestamps),
        )
        for timestamp in timestamps:
            if animate_first:
                live_text = [
                    {
                        "content": f"   ↳  Generating animations for {analysis} | {benchmark_id} | {timestamp} ...",
                        "style": "cyan",
                    },
                ]
                utils.update_live(utils.progressbar, live_text)
            try:
                log_str, tables, _ = post_process(
                    analysis=analysis,
                    benchmark_id=benchmark_id,
                    timestamp=timestamp,
                    animation=animate_first,
                )
                animate_first = False
                live_text = [{"content": f"   ↳  {log_str}"}]
                live_text += [{"table": t} for t in tables[:2]] if tables else []
            except Exception as e:
                err_msg = f"[red]Error processing {analysis} | {benchmark_id} | {timestamp}: {e}"
                live_text = [
                    {
                        "content": f"   ↳  {analysis} | {benchmark_id} | {timestamp}\n",
                        "style": "yellow",
                    },
                    {"content": err_msg},
                ]
            utils.progressbar.advance(task_id)
            utils.update_live(utils.progressbar, live_text)
            utils.log_live_state_single(
                get_path(analysis, benchmark_id, timestamp),
                "postprocess.log",
            )


def post_process_all_benchmarks(analysis: str) -> None:
    """Postprocess all benchmarks for a single analysis."""
    analysis_dir = HERE / "results" / analysis
    benchmarks_dir_names = [
        benchmark_dir.name for benchmark_dir in analysis_dir.iterdir() if benchmark_dir.is_dir()
    ]

    animate_first = {}
    for benchmark_id in benchmarks_dir_names:
        animate_first[benchmark_id] = True

    jobs = [
        (benchmark_id, timestamp)
        for benchmark_id in benchmarks_dir_names
        for timestamp in [
            timestamp_dir.name
            for timestamp_dir in (HERE / "results" / analysis / benchmark_id).iterdir()
            if timestamp_dir.is_dir()
        ]
    ]

    with utils.live:
        task_id = utils.progressbar.add_task(
            "[bold white]📸 [bold light_coral]Post-Processing: ",
            total=len(jobs),
        )

        for benchmark_id, timestamp in jobs:
            if animate_first[benchmark_id]:
                live_text = [
                    {
                        "content": f"   ↳  Generating animations for {analysis} | {benchmark_id} | {timestamp} ...",
                        "style": "cyan",
                    },
                ]
                utils.update_live(utils.progressbar, live_text)
            try:
                log_str, tables, _ = post_process(
                    analysis=analysis,
                    benchmark_id=benchmark_id,
                    timestamp=timestamp,
                    animation=animate_first[benchmark_id],
                )
                live_text = [{"content": f"   ↳  {log_str}"}]
                live_text += [{"table": t} for t in tables[:2]] if tables else []
                animate_first[benchmark_id] = False
            except Exception as e:
                err_msg = f"[red]Error processing {analysis} | {benchmark_id} | {timestamp}: {e}"
                live_text = [
                    {
                        "content": f"   ↳  {analysis} | {benchmark_id} | {timestamp}\n",
                        "style": "yellow",
                    },
                    {"content": err_msg},
                ]
            utils.progressbar.advance(task_id)
            utils.update_live(utils.progressbar, live_text)
            utils.log_live_state_single(
                get_path(analysis, benchmark_id, timestamp),
                "postprocess.log",
            )


def main() -> None:
    """Parse the command line and post-process the selected runs."""
    # The video helper and the plotting code resolve paths from the working directory.
    os.chdir(HERE)
    analysis, benchmark_id, timestamp = utils.args_parser_postprocess()

    if timestamp and benchmark_id:
        post_process_single_benchmark(analysis, benchmark_id, timestamp)
        return
    if benchmark_id:
        post_process_median(analysis, benchmark_id)
        post_process_all_timestamps(analysis, benchmark_id)
        return
    post_process_median_all_benchmarks(analysis)
    post_process_all_benchmarks(analysis)


if __name__ == "__main__":
    main()
