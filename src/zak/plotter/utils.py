"""Helpers shared by the plotting modules: figure sizes, axis formatting and saving."""

from pathlib import Path

from matplotlib.ticker import ScalarFormatter


def save(
    plt,
    path: Path,
    filename: str = "autosave",
    extension: str = "pdf",
    *,
    pdf_dir: bool = True,
) -> None:
    """Save the current figure as a PDF or PNG file.

    Args:
        plt: The `matplotlib.pyplot` module.
        path: Output directory. PNG files go to its "png" sub-directory.
        filename: Name of the file, without extension.
        extension: "pdf" or "png".
        pdf_dir: Whether PDF files go to the "pdf" sub-directory of `path` rather than to `path`
            itself.

    Raises:
        ValueError: If the extension is neither "pdf" nor "png".
    """
    match extension:
        case "png":
            path_png = path / "png"
            path_png.mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{path_png}/{filename}.png", dpi=600, format="png")
        case "pdf":
            path_pdf = path / "pdf" if pdf_dir else path
            path_pdf.mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{path_pdf}/{filename}.pdf", dpi=1000, format="pdf")
        case _:
            msg_error = f"Invalid extension: {extension}"
            raise ValueError(msg_error)


def fig_size(
    figsize: str = "squareSmall",
    width: float = 6.93,
    factor: float = 2.54,
) -> tuple[float, float]:
    """Return the (width, height) of a named figure format.

    The sizes are multiples of the base unit `width / factor`; with the defaults, 6.93 cm
    converted to inches.

    Args:
        figsize: Name of the format: "squareColorbar", "squareColorbar2", "squareSmall",
            "squareMedium", "squareBig", "rectangleSmall", "rectangleMedium", "rectangleBig" or
            "rectangleQuasi".
        width: Base width, in centimetres with the default factor.
        factor: Divisor that converts the base width to inches.

    Returns:
        The figure size expected by Matplotlib.
    """
    unit_base = width / factor
    register_fig_size = {
        "squareColorbar": (1.3 * unit_base, unit_base),
        "squareColorbar2": (1.43 * unit_base, unit_base),
        "squareSmall": (unit_base, unit_base),
        "squareMedium": (2 * unit_base, 2 * unit_base),
        "squareBig": (width, width),
        "rectangleSmall": (2 * unit_base, unit_base),
        "rectangleMedium": (4 * unit_base, 2 * unit_base),
        "rectangleBig": (2 * width, width),
        "rectangleQuasi": (1.25 * unit_base, unit_base),
    }
    return register_fig_size[figsize]


def set_square_ratio(ax) -> None:
    """Make the plotting area of the axes square, whatever the data ranges."""
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")


def set_scientific_notation(ax, axis: str, power_limit: int = 1) -> None:
    """Format the tick labels of an axis in scientific notation.

    Args:
        ax: The Matplotlib axes.
        axis: "x" or "y"; any other value leaves the axes unchanged.
        power_limit: Scientific notation is used for magnitudes below 10^-power_limit or above
            10^power_limit.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-power_limit, power_limit))
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
