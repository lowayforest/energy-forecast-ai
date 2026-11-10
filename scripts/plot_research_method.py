"""Draw the paper's method as a compact, title-free vector diagram."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from audit_research_dataset import ROOT


OUT = ROOT / "reports/figures"
INK = "#181818"
TRAIN = "#666666"
ACCENT = "#93AA2D"
ACCENT_FILL = "#F2F6D9"


def block(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: tuple[str, ...],
    *,
    accent: bool = False,
    sizes: tuple[float, ...] | None = None,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.018,rounding_size=0.075",
            facecolor=ACCENT_FILL if accent else "white",
            edgecolor=ACCENT if accent else INK,
            linewidth=1.35,
        )
    )
    center_x = x + width / 2
    center_y = y + height / 2
    spacing = 0.37 if height > 1.45 else 0.34
    for index, line in enumerate(lines):
        ax.text(
            center_x,
            center_y + (len(lines) - 1 - 2 * index) * spacing / 2,
            line,
            ha="center",
            va="center",
            fontsize=sizes[index] if sizes else 10.5,
            color=INK,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    training: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.35 if training else 1.55,
            color=TRAIN if training else INK,
            linestyle=(0, (3.5, 3)) if training else "solid",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3",
        )
    )


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "mathtext.fontset": "stix",
        "text.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "pv-paper-method-v2",
    })
    fig, ax = plt.subplots(figsize=(13.8, 5.55), facecolor="white")
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 5.55)
    ax.axis("off")

    top_y, top_h = 3.10, 1.55
    bottom_y, bottom_h = 0.65, 1.35
    top_blocks = (
        (0.40, 2.10, ("Target-day NWP", r"History $d-9$ to $d-3$")),
        (2.90, 2.15, ("Ross + PVWatts", "Site / GHI calibration", r"$H^{\mathrm{cond}}$")),
        (5.45, 2.25, ("Qwen3.5-9B",
                      r"NWP, history, $H^{\mathrm{cond}}$",
                      r"Base / LoRA; logits $z_k$")),
        (8.10, 2.15, (r"Class probabilities $p_k$", r"$\hat r=\sum_k p_k\tilde r_k$")),
        (10.65, 2.75, ("Day-ahead energy",
                        r"$\hat H=\mathrm{clip}(H^{\mathrm{cond}}+\hat r,0,24)$",
                        r"$\hat E=C\hat H$ (MWh)")),
    )
    for index, (x, width, lines) in enumerate(top_blocks):
        block(
            ax, x, top_y, width, top_h, lines,
            accent=index == 2,
            sizes=(10.5, 10.1, 10.5) if index == 4 else
                  (10.5, 10.0, 9.8) if index == 2 else None,
        )

    mid_y = top_y + top_h / 2
    for x0, x1 in ((2.52, 2.87), (5.07, 5.42), (7.72, 8.07), (10.27, 10.62)):
        arrow(ax, (x0, mid_y), (x1, mid_y))

    block(ax, 0.40, bottom_y, 2.10, bottom_h,
          ("Observed AC power", r"$E^{\mathrm{actual}}$ (train / score)"),
          sizes=(10.3, 9.9))
    block(ax, 2.90, bottom_y, 2.15, bottom_h,
          (r"$r=H^{\mathrm{actual}}-H^{\mathrm{cond}}$",
           "Train terciles: A / B / C", r"Class medians $\tilde r_k$"),
          sizes=(10.4, 10.0, 10.0))
    block(ax, 5.45, bottom_y, 2.25, bottom_h,
          ("LoRA adaptation", "Rank 8; final 2 layers", "890 training steps"),
          sizes=(10.5, 9.8, 9.8))
    block(ax, 10.65, bottom_y, 2.75, bottom_h,
          ("Held-out scoring", "MAE / RMSE by date",
           "Actual used after forecast"),
          sizes=(10.5, 9.9, 9.9))

    low_mid_y = bottom_y + bottom_h / 2
    arrow(ax, (2.52, low_mid_y), (2.87, low_mid_y), training=True)
    arrow(ax, (5.07, low_mid_y), (5.42, low_mid_y), training=True)
    arrow(ax, (3.98, top_y - 0.03), (3.98, bottom_y + bottom_h + 0.03),
          training=True)
    arrow(ax, (6.58, bottom_y + bottom_h + 0.03), (6.58, top_y - 0.03),
          training=True)
    arrow(ax, (12.03, top_y - 0.03), (12.03, bottom_y + bottom_h + 0.03),
          training=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "research_method"
    fig.savefig(
        stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08,
        metadata={"Software": f"matplotlib {matplotlib.__version__}"},
    )
    svg_path = stem.with_suffix(".svg")
    fig.savefig(
        svg_path, bbox_inches="tight", pad_inches=0.08,
        metadata={"Date": None, "Creator": "plot_research_method.py"},
    )
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    metadata = {
        "figure": "research_method",
        "scope": "v3 LoRA inference with frozen base control; train-only fit and held-out score boundaries",
        "method_sources": [
            "reports/research_daily_dataset.md",
            "reports/research_physics_baseline.md",
            "reports/research_conditional_v2_results.md",
            "reports/research_class_v3_protocol.md",
        ],
        "matplotlib_version": matplotlib.__version__,
    }
    (OUT / "research_method_data.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
