"""Publication-Grade Matplotlib Style Configuration (`plot_style.py`)."""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "static": "#6C7A89",
    "memoryless": "#3A8FB7",
    "no_unc": "#D9822B",
    "avid_v2": "#16694A",
    "oracle": "#2C3E75",
    "unconstrained": "#B03A2E",
    "orbit": "#7D3C98",
    "accent": "#C0392B",
    "gold": "#B7950B",
    "teal": "#148F77",
    "slate": "#2E4053",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.fontsize": 8.2,
            "legend.framealpha": 0.92,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def save_pub_figure(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
