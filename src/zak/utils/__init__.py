"""Console output, input/output and other helpers shared across the package.

Importing this module configures Rich (tracebacks, pretty printing), widens the NumPy and PyTorch
print options and creates the consoles, the progress bar and the live displays used during a run.
"""

from io import StringIO
from typing import Any
from pathlib import Path
from datetime import datetime
from collections.abc import Sequence
import gc
import _pickle as pickle
import argparse
import concurrent.futures

from rich import pretty
from rich.live import Live
from rich.text import Text
from rich.table import Table
from ruamel.yaml import YAML
from rich.console import Group, Capture, Console
from rich.progress import (
    TaskID,
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.traceback import install
import numpy
import torch

install()
pretty.install()
width_out = 150
numpy.set_printoptions(linewidth=width_out)
torch.set_printoptions(linewidth=width_out)
# The live displays write to stderr so that they do not mix with what `console` prints on stdout.
# `console_direct` records its output so that `log_console` can save it to a file.
console_live = Console(record=False, width=width_out, log_time=False, log_path=False, stderr=True)
console = Console(record=False, width=width_out, log_time=False, log_path=False, stderr=False)
console_direct = Console(record=True, width=width_out, log_time=False, log_path=False, stderr=False)

progressbar = Progress(
    TextColumn("[progress.description]{task.description}[progress.percentage]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("•"),
    TextColumn("[green]iter:"),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
)


live = Live(
    Group(progressbar, Text("")),
    console=console_live,
    auto_refresh=True,
)
live_simple = Live(Text(""), refresh_per_second=4, console=console_live)


def log_live_state(path: Path, filename: str) -> None:
    """Append the current content of the live display to a text file.

    The renderable is printed on a temporary in-memory console, so nothing reaches the terminal.

    Args:
        path: Directory of the log file.
        filename: Name of the log file.
    """
    string_io = StringIO()
    tmp_console = Console(file=string_io, width=200, highlight=False)
    tmp_console.print(live._renderable)
    content = string_io.getvalue()
    log_file = path / filename
    with log_file.open("a") as f:
        _ = f.write(content)


def log_live_state_single(path: Path, filename: str) -> None:
    """Write the current content of the live display to a text file, replacing its content.

    Args:
        path: Directory of the log file.
        filename: Name of the log file.
    """
    string_io = StringIO()
    tmp_console = Console(file=string_io, width=200, highlight=False)
    tmp_console.print(live._renderable)
    content = string_io.getvalue()
    log_file = path / filename
    with log_file.open("w") as f:
        _ = f.write(content)


def log_console(path: Path, filename: str, *, capture: Capture | None = None) -> None:
    """Save console output to a text file.

    Args:
        path: Directory of the log file.
        filename: Name of the log file.
        capture: Captured output to save. When None, everything recorded so far by
            `console_direct` is saved instead.
    """
    if capture is None:
        log_file = path / filename
        console_direct.save_text(str(log_file))
    else:
        log_str = capture.get()
        log_file = path / filename
        with log_file.open("w") as f:
            _ = f.write(log_str)


def update_live_simple(text_items: list[dict[str, str]]) -> None:
    """Replace the content of the text-only live display.

    Args:
        text_items: One dict per line with the keys "content" (the text) and, optionally,
            "style" (a Rich style). Items with an empty content are skipped.
    """
    group_items = []
    for item in text_items:
        content = item.get("content", "")
        style = item.get("style", "")
        if content:
            group_items.append(Text(content, style=style))
    live_simple.update(Group(*group_items))


def update_live(progress_bar, text_items: list[dict[str, Any]]) -> None:
    """Replace the content of the main live display: the progress bar followed by text and tables.

    Args:
        progress_bar: The Rich progress bar shown on top.
        text_items: One dict per item, with the keys

            - "content": the text;
            - "style": the Rich style applied to the text (optional);
            - "table": a Rich Table shown instead of the text (optional).

            A trailing newline item is appended to this list in place.
    """
    text_items.append({"content": "\n"})
    group_items = [progress_bar]

    for item in text_items:
        if "table" in item and item["table"] is not None:
            group_items.append(item["table"])
            continue

        content = item.get("content", "")
        style = item.get("style", "")

        if content:
            # Text that carries its own Rich markup tags is parsed as markup; otherwise the
            # given style is applied to the whole text.
            if "[" in content and "]" in content:
                text_obj = Text.from_markup(content)
            else:
                text_obj = Text(content, style=style)

            group_items.append(text_obj)

    live.update(Group(*group_items))


def table(headers: Sequence[str], title: str | None = None) -> Table:
    """Create an empty Rich table with the package styling.

    Args:
        headers: Column headers. The first column is left-aligned, the others right-aligned.
        title: Optional title printed above the table.

    Returns:
        The table, ready to receive rows.
    """
    style_1 = {"style": "blue", "no_wrap": True, "justify": "left"}
    style_2 = {"style": "cyan", "no_wrap": True, "justify": "right"}
    fields = {}
    for i, header in enumerate(headers):
        fields[header] = style_1 if i == 0 else style_2
    _title = f"\n--- {title} ---" if title else None
    table = Table(
        title=_title,
        title_style="dark_orange",
        title_justify="left",
        header_style="orchid",
        show_edge=True,
        pad_edge=False,
        expand=False,
    )
    for field, style in fields.items():
        table.add_column(field, **style)
    return table


def load_yaml(path: Path, filename: str) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Directory of the file.
        filename: Name of the file, extension included.

    Returns:
        The content of the file.
    """
    yaml = YAML(typ="safe", pure=True)
    with Path.open(path / filename, encoding="utf-8") as f:
        return yaml.load(f)


def save(variables: Any, path: Path, filename: str) -> None:
    """Pickle an object to ``<path>/<filename>.pickle``."""
    with (path / f"{filename}.pickle").open("wb") as file:
        pickle.dump(variables, file)


def load(path: Path, filename: str) -> dict[str, Any]:
    """Load the object pickled in ``<path>/<filename>.pickle``."""
    with (path / f"{filename}.pickle").open("rb") as f:
        return pickle.load(f)


def time_stamp() -> str:
    """Return the current local time formatted as ``YYYY_MM_DD_HHMMSS``."""
    # Naive local time on purpose: the stamp only names the output directory of a run.
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")  # noqa: DTZ005


def args_parser() -> tuple[str, str, int]:
    """Parse the command line of the analysis launcher.

    Returns:
        The analysis, the benchmark id and the random seed.
    """
    parser = argparse.ArgumentParser(description="Script with seed and benchmark parameters")
    parser.add_argument(
        "--analysis",
        type=str,
        default=None,
        help="analysis to run: ddo, rbdo, zakrbdo or isorakrbdo (default: None)",
    )
    parser.add_argument("--seed", type=int, default=27, help="random seed (default: 27)")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="benchmark id, e.g. four_branches (default: None)",
    )
    return (
        parser.parse_args().analysis,
        parser.parse_args().benchmark,
        parser.parse_args().seed,
    )


def args_parser_postprocess() -> tuple[str, str, str]:
    """Parse the command line of the post-processing launcher.

    Returns:
        The analysis, the benchmark id and the name of the run directory.
    """
    parser = argparse.ArgumentParser(description="Benchmark and time stamp parameters")
    parser.add_argument(
        "--analysis",
        type=str,
        default=None,
        help="analysis to post-process: ddo, rbdo, zakrbdo or isorakrbdo (default: None)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="benchmark id, e.g. four_branches; all benchmarks when omitted (default: None)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="name of the run directory; all runs of the benchmark when omitted (default: None)",
    )
    return (
        parser.parse_args().analysis,
        parser.parse_args().benchmark,
        parser.parse_args().timestamp,
    )


def parallelize(fun_to_eval: list[tuple[Any]], max_workers: int = 8) -> list[Any]:
    """Run function calls in a pool of processes.

    Args:
        fun_to_eval: One tuple per call: the function followed by its positional arguments.
        max_workers: Maximum number of worker processes.

    Returns:
        The results, in the order of the calls.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(*fun) for fun in fun_to_eval]
        return [fun.result() for fun in futures]


def clear_gpu_cache() -> None:
    """Run the garbage collector and release the cached memory of the MPS and CUDA devices."""
    gc.collect()
    if torch.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_difference(A, B) -> torch.Tensor:
    """Return the rows of A that are not rows of B.

    Both tensors are flattened to two dimensions, keeping the last one, and rows are compared
    with exact equality.

    Args:
        A: Tensor whose rows are filtered.
        B: Tensor of the rows to discard.

    Returns:
        The rows of A absent from B, in their original order, with shape (n, A.shape[-1]).
    """
    A_reshaped = A.reshape(-1, A.shape[-1])
    B_reshaped = B.reshape(-1, B.shape[-1])
    mask = torch.ones(A_reshaped.shape[0], dtype=torch.bool)
    for i in range(A_reshaped.shape[0]):
        matches = torch.all(A_reshaped[i : i + 1] == B_reshaped, dim=1)
        if torch.any(matches):
            mask[i] = False
    return A_reshaped[mask]


__all__ = [
    "Group",
    "Path",
    "Table",
    "TaskID",
    "Text",
    "args_parser",
    "args_parser_postprocess",
    "clear_gpu_cache",
    "console",
    "console_direct",
    "live",
    "load",
    "load_yaml",
    "log_console",
    "log_live_state",
    "log_live_state_single",
    "parallelize",
    "progressbar",
    "save",
    "table",
    "tensor_difference",
    "time_stamp",
    "update_live",
    "update_live_simple",
]
