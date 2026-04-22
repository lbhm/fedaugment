import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from pypdf import PdfReader

PALETTE = sns.color_palette("colorblind")
COLOR_MAP = {
    # Projection methods
    "cl_optim": PALETTE[0],
    "cl_noval": PALETTE[0],
    "la2m_default": PALETTE[1],
    "la2m_nopca": PALETTE[2],
    "v2v": PALETTE[3],
    "union_minus": PALETTE[4],
    "union_plus": PALETTE[5],
    "max_metric": PALETTE[7],
    "centralized": PALETTE[7],
    "cl_default": PALETTE[8],
    # Data curation techniques
    "fft_cos": PALETTE[0],
    "fft_euc": PALETTE[1],
    "grid": PALETTE[2],
    "random": PALETTE[3],
    "full": PALETTE[4],
    # Finetuning experiments
    "deepjoin": PALETTE[0],
    "starmie": PALETTE[1],
}

# acmart defaults:
# \normalsize: 9 pt (baselineskip=11.0pt)
# \small: 8 pt (baselineskip=10.0pt)
# \footnotesize: 7 pt (baselineskip=8.0pt)
# \scriptsize: 6 pt (baselineskip=7.0pt)
FONT_SIZE = 7
FONT_SIZE_SMALL = 6
LINE_WIDTH = 0.5

# acmart column widths:
# - Single column: 241.14749pt
# - Double column: 506.295pt


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        color_codes=True,
        rc={
            # Generic settings #
            "backend": "pgf",
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": "\n".join(  # noqa: FLY002
                [
                    r"\usepackage[T1]{fontenc}",  # font encoding
                    r"\usepackage{microtype}",  # better typography
                    r"\usepackage[tt=false, type1=true]{libertine}",  # serif font
                    r"\usepackage[varqu]{zi4}",  # monospace font
                    r"\usepackage[libertine]{newtxmath}",  # math font
                    r"\newcommand{\system}[0]{\textsc{Fed\-Augment}}",
                ]
            ),
            "font.family": "serif",  # use serif font for text elements
            "text.usetex": True,  # use inline math for ticks
            "figure.autolayout": True,  # tight layout
            "savefig.format": "pdf",  # {png, ps, pdf, svg}
            "savefig.bbox": "tight",  # {tight, standard}
            "savefig.pad_inches": 0.0075,  # padding to be used, when bbox is set to 'tight'
            "savefig.transparent": False,  # transparent background
            # Layout settings #
            "axes.labelpad": 1.5,  # 4
            "axes.labelsize": FONT_SIZE,  # medium
            "axes.linewidth": LINE_WIDTH,  # 0.8
            "axes.titlesize": FONT_SIZE,  # large
            "font.size": FONT_SIZE,  # 10.0
            "grid.linewidth": LINE_WIDTH,  # 0.8
            "hatch.linewidth": LINE_WIDTH,  # 1.0
            "legend.borderaxespad": 0.25,  # 0.5
            "legend.borderpad": 0.4,  # 0.4
            "legend.columnspacing": 1.0,  # 2.0
            "legend.fontsize": FONT_SIZE_SMALL,  # medium
            "legend.frameon": False,  # True
            "legend.handleheight": 0.7,  # 0.7
            "legend.handlelength": 1.5,  # 2.0
            "legend.handletextpad": 0.7,  # 0.8
            "legend.labelspacing": 0.4,  # 0.5
            "lines.linewidth": LINE_WIDTH * 2,  # 1.5
            "lines.markersize": 3.0,  # 6.0
            "patch.edgecolor": "black",  # "black"
            "patch.force_edgecolor": True,  # False
            "patch.linewidth": LINE_WIDTH,  # 1.0
            "xtick.labelsize": FONT_SIZE_SMALL,
            "xtick.major.pad": 2.0,  # 3.5
            "xtick.major.size": 3.5,  # 2.5
            "xtick.major.width": LINE_WIDTH,  # 0.8
            "xtick.minor.pad": 2.0,  # 3.4
            "xtick.minor.size": 2.0,  # 2.0
            "xtick.minor.width": LINE_WIDTH * 0.75,  # 0.6
            "ytick.labelsize": FONT_SIZE_SMALL,
            "ytick.major.pad": 2.0,  # 3.5
            "ytick.major.size": 3.5,  # 3.5
            "ytick.major.width": LINE_WIDTH,  # 0.8
            "ytick.minor.pad": 2.0,  # 3.4
            "ytick.minor.size": 2.0,  # 2.0
            "ytick.minor.width": LINE_WIDTH * 0.75,  # 0.6
        },
    )


def delete_tex_cache() -> None:
    shutil.rmtree(Path.home() / ".cache" / "matplotlib" / "tex.cache", ignore_errors=True)


def save_and_measure(fig: Figure, path: Path | str, **fig_kwargs: Any) -> tuple[float, float]:
    """Save the figure and measure its dimensions.

    Args:
        fig: The matplotlib figure to save and measure.
        path: The file path to save the figure to.
        **fig_kwargs: Additional keyword arguments to pass to `fig.savefig()`.

    Returns:
        A tuple containing the width and height of the saved figure in inches.
    """
    fig.savefig(path, **fig_kwargs)
    plt.close(fig)

    reader = PdfReader(path)
    page = reader.pages[0]

    # Dimensions are in points
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    print(
        f"{path}: {width:.3f}x{height:.3f} pt "
        f"({width * 0.03527778:.2f}x{height * 0.03527778:.2f} cm)"
    )
    return width / 72, height / 72


def plot_legend(
    path: Path | str,
    handles: Sequence[Artist | BarContainer],
    labels: list[str] | None = None,
    ncol: int | None = None,
) -> None:
    """Plot a legend with the given handles and labels."""
    if ncol is None:
        ncol = len(handles)

    fig = plt.figure(figsize=(10, 3))
    fig.legend(handles=handles, labels=labels, loc="center", ncol=ncol)
    save_and_measure(fig, path)
