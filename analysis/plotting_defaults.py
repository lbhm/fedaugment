import shutil
from pathlib import Path

import seaborn as sns

COLOR_MAP = {
    "cl_optim": sns.color_palette()[0],
    "cl_noval": sns.color_palette()[0],  # Same as cl_optim but trained without early stopping
    "la2m_default": sns.color_palette()[1],
    "la2m_nopca": sns.color_palette()[2],
    "v2v": sns.color_palette()[3],
    "union_minus": sns.color_palette()[4],
    "union_plus": sns.color_palette()[5],
    "max_metric": sns.color_palette()[7],
    "centralized": sns.color_palette()[7],
    "cl_default": sns.color_palette()[8],
}


def set_style(font_scale: float = 6.0) -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        color_codes=True,
        rc={
            "axes.labelpad": 1.5,  # 4
            "axes.labelsize": font_scale,
            "axes.linewidth": 0.4,  # 0.8
            "axes.titlesize": font_scale,
            "font.family": "Computer Modern",
            "font.size": font_scale,
            "grid.linewidth": 0.4,  # 0.8
            "hatch.linewidth": 0.5,  # 1.0
            "legend.borderpad": 0.4,  # 0.4
            "legend.borderaxespad": 0.25,  # 0.5
            "legend.columnspacing": 1.0,  # 2.0
            "legend.fontsize": font_scale,
            "legend.frameon": False,  # True
            "legend.handleheight": 0.7,  # 0.7
            "legend.handlelength": 1.5,  # 2.0
            "legend.handletextpad": 0.7,  # 0.8
            "legend.labelspacing": 0.4,  # 0.5
            "legend.title_fontsize": font_scale,
            "lines.linewidth": 1.0,  # 1.5
            "lines.markersize": 3.0,  # 6.0
            "patch.linewidth": 0.5,  # 1.0
            "xtick.labelsize": font_scale,
            "xtick.major.pad": 2.0,  # 3.5
            "xtick.major.size": 3.5,  # 2.5
            "xtick.major.width": 0.4,  # 0.8
            "xtick.minor.pad": 2.0,  # 3.4
            "xtick.minor.size": 2.0,  # 2.0
            "xtick.minor.width": 0.3,  # 0.6
            "ytick.labelsize": font_scale,
            "ytick.major.pad": 2.0,  # 3.5
            "ytick.major.size": 3.5,  # 3.5
            "ytick.major.width": 0.4,  # 0.8
            "ytick.minor.pad": 2.0,  # 3.4
            "ytick.minor.size": 2.0,  # 2.0
            "ytick.minor.width": 0.3,  # 0.6
            "text.usetex": True,
            "text.latex.preamble": (r"\newcommand{\system}[0]{\textsc{FedAugment}}"),
        },
    )


def delete_tex_cache() -> None:
    shutil.rmtree(Path.home() / ".cache" / "matplotlib" / "tex.cache", ignore_errors=True)
