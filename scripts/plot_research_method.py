"""Draw the physics-guided 9B PV forecasting method as a vector figure."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from audit_research_dataset import ROOT


OUT = ROOT / "reports/figures"
INK = "#253746"
MUTED = "#5F6E7B"
LINE = "#CCD6DD"
BLUE = "#39799B"
TEAL = "#188C86"
ORANGE = "#C46B43"
SLATE = "#71818E"


def box(ax: plt.Axes, x: float, y: float, w: float, h: float,
        title: str, lines: list[str], color: str, fill: str,
        title_size: float = 10.2, body_size: float = 8.8) -> None:
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.13",
                           linewidth=1.2, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    ax.plot([x + 0.20, x + w - 0.20], [y + h - 0.55, y + h - 0.55],
            color=color, lw=0.85, alpha=0.45)
    ax.text(x + 0.20, y + h - 0.27, title, ha="left", va="center",
            color=color, fontsize=title_size, fontweight="bold")
    ax.text(x + 0.20, y + h - 0.77, "\n".join(lines),
            ha="left", va="top", color=INK, fontsize=body_size,
            linespacing=1.43)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = SLATE, dashed: bool = False, width: float = 1.7) -> None:
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            linewidth=width, color=color,
                            linestyle=(0, (4, 3)) if dashed else "solid",
                            shrinkA=0, shrinkB=0, connectionstyle="arc3")
    ax.add_patch(patch)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "pv-paper-method-v1",
    })
    fig, ax = plt.subplots(figsize=(14, 7.25), facecolor="white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.25)
    ax.axis("off")

    ax.text(0.42, 7.02, "Physics-guided local 9B model for day-ahead PV energy",
            fontsize=16, fontweight="bold", va="top")
    ax.text(0.45, 6.47, "INFERENCE  ·  AVAILABLE BEFORE TARGET DAY d",
            fontsize=9.4, fontweight="bold", color=BLUE, va="center")
    ax.text(13.50, 6.47, "No target-day measurements in the prompt",
            fontsize=8.7, color=MUTED, ha="right", va="center")

    top_y, top_h = 3.85, 2.25
    box(ax, 0.45, top_y, 2.45, top_h, "01  Inputs", [
        "Target-day NWP: irradiance,",
        "temperature, wind, humidity",
        "History: d−9 to d−3",
    ], BLUE, "#F4F9FC")
    box(ax, 3.15, top_y, 2.53, top_h, "02  Physics anchor", [
        "Ross cell temperature",
        "PVWatts DC approximation",
        "Train-fitted site / GHI bias",
        r"Output: $H^{\mathrm{cond}}$ (h)",
    ], BLUE, "#F4F9FC")
    box(ax, 5.93, top_y, 2.52, top_h, "03  Local 9B model", [
        "Qwen3.5-9B, 4-bit",
        "LoRA; base as control",
        "Prompt: site, NWP, history",
        r"Physics anchor $H^{\mathrm{cond}}$",
    ], ORANGE, "#FFF8F4")
    box(ax, 8.70, top_y, 2.13, top_h, "04  Decode", [
        "A / B / C token logits",
        "pₖ: restricted softmax",
        "mₖ: train-class medians",
    ], ORANGE, "#FFF8F4")
    ax.text(8.90, 4.28, r"$\hat r=\sum_k p_km_k$",
            ha="left", va="center", fontsize=10.0, color=INK)
    box(ax, 11.08, top_y, 2.47, top_h, "05  Daily forecast", [
        r"$\hat H=\mathrm{clip}(H^{\mathrm{cond}}+\hat r,0,24)$",
        r"$\hat E=C\hat H$  (MWh)",
        "Next-day site energy",
    ], TEAL, "#F2FAF8", body_size=8.5)

    for start_x, end_x, color in ((2.91, 3.13, BLUE), (5.69, 5.91, BLUE),
                                  (8.46, 8.68, ORANGE), (10.84, 11.06, TEAL)):
        arrow(ax, (start_x, top_y + 1.10), (end_x, top_y + 1.10), color)

    ax.plot([0.45, 13.55], [3.45, 3.45], color=LINE, lw=1)
    ax.text(0.45, 3.18, "TRAINING & EVALUATION",
            fontsize=9.4, fontweight="bold", color=SLATE, va="center")

    lower_y, lower_h = 0.90, 1.95
    box(ax, 0.45, lower_y, 2.45, lower_h, "Observed AC power", [
        "Training-period actuals",
        r"$H^{\mathrm{actual}}=E^{\mathrm{actual}}/C$",
        "Never in inference prompt",
    ], SLATE, "#F6F8FA", title_size=9.8, body_size=8.4)
    box(ax, 3.15, lower_y, 2.95, lower_h, "Train-only residual labels", [
        r"$r=H^{\mathrm{actual}}-H^{\mathrm{cond}}$",
        "Train terciles → A / B / C",
        "Train class medians → mₖ",
    ], SLATE, "#F6F8FA", title_size=9.8, body_size=8.4)
    box(ax, 6.35, lower_y, 2.48, lower_h, "Supervised adaptation", [
        "890 training station-days",
        "Rank 8; final 2 layers",
        "890 updates; batch size 1",
    ], ORANGE, "#FFF8F4", title_size=9.8, body_size=8.4)
    box(ax, 10.15, lower_y, 3.40, lower_h, "Held-out scoring", [
        "Actual energy used after prediction",
        "MAE / RMSE; date-block intervals",
        "Base model uses no LoRA adapter",
    ], TEAL, "#F2FAF8", title_size=9.8, body_size=8.4)

    arrow(ax, (2.91, lower_y + 0.99), (3.13, lower_y + 0.99), SLATE, dashed=True)
    arrow(ax, (6.11, lower_y + 0.99), (6.33, lower_y + 0.99), ORANGE, dashed=True)
    arrow(ax, (7.60, lower_y + lower_h + 0.02), (7.60, top_y - 0.03),
          ORANGE, dashed=True)
    arrow(ax, (12.31, top_y - 0.03), (12.31, lower_y + lower_h + 0.02),
          TEAL, dashed=True)

    ax.text(0.47, 0.42,
            "All site calibration, GHI-band offsets, class thresholds and class medians are fitted on the training split only.",
            fontsize=8.6, color=MUTED, va="center")
    ax.text(13.54, 0.42, "Solid: inference    Dashed: training / scoring",
            fontsize=8.6, color=MUTED, va="center", ha="right")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "research_method"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.18,
                metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.18,
                metadata={"Date": None, "Creator": "plot_research_method.py"})
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18,
                metadata={"Title": "Physics-guided 9B PV forecasting method"})
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
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
