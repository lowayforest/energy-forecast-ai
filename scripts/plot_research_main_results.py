"""Create the paper's reproducible main-results figure from frozen v3 scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_research_dataset import ROOT
from compare_research_mlx import key, paired_improvement


DATA = ROOT / "data/processed/research"
FIGURES = ROOT / "reports/figures"
EXPECTED = {
    "mlx_v3_comparison_valid.json": "23aca494d19fbf5f0e14f7a387f3d2d5c1eb444751e6750803c86224bed434a9",
    "v3_final_test_comparison.json": "fd5b64665b4943641646368f2de5b5c26d0a3b1127aea89c8313a99d502a79f6",
    "daily_test.jsonl": "7fd4970581a695f2ef497017aab5065e7e0d5db3e9e737b5f971c060b97a14fe",
    "conditional_test.jsonl": "8e28da04ae859ddc511f1e2ec4336e1e1ed4c7a36106f0a30a0012b8f16feb8d",
    "mlx_v3_base_test.jsonl": "146fece212c4c24978c9d963a2d11c421f7cc6882e494c6a4b7b98b0c93441e5",
    "mlx_v3_adapter_test.jsonl": "9b1e9d6832c0422922892517f1b771707080cc75e420291b26af1c55ee161702",
}
METHODS = ("physical", "conditional", "base", "adapter")
LABELS = {
    "physical": "Calibrated physics",
    "conditional": "Conditioned physics",
    "base": "9B base + physics",
    "adapter": "9B LoRA + physics",
}
COLORS = {
    "physical": "#758391",
    "conditional": "#39799B",
    "base": "#188C86",
    "adapter": "#C46B43",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.065, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")


def draw_overall(ax: plt.Axes, scores: dict, title: str, letter: str) -> None:
    y_positions = np.arange(len(METHODS))[::-1]
    ax.set_title(title, loc="left", pad=15, fontsize=10.5, fontweight="normal")
    panel_label(ax, letter)
    for method, y in zip(METHODS, y_positions):
        value = scores[method]["mae_hours"]
        ax.scatter(value, y, s=66, color=COLORS[method], edgecolor="white", lw=0.8, zorder=3)
        ax.text(value + 0.014, y, f"{value:.3f}", va="center", ha="left",
                fontsize=8.5, color="#24303B")
    ax.set_yticks(y_positions, [LABELS[m] for m in METHODS])
    ax.set_xlim(0.55, 1.16)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_xlabel("MAE (equivalent full-load hours)")
    ax.grid(axis="x", color="#E6EBEF", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def draw_station(ax: plt.Axes, per_station: dict) -> None:
    panel_label(ax, "C")
    ax.set_title("Test by station", loc="left", pad=15,
                 fontsize=10.5, fontweight="normal")
    stations = ("S-1", "S-2", "S-3", "S-4")
    x = np.arange(len(stations))
    for method, marker, offset in (("conditional", "o", -0.12),
                                   ("base", "s", 0.0), ("adapter", "D", 0.12)):
        values = [per_station[s][method]["mae_hours"] for s in stations]
        ax.scatter(x + offset, values, color=COLORS[method], marker=marker,
                   s=67, edgecolor="white", linewidth=0.7,
                   label=LABELS[method], zorder=3)
    ax.set_xticks(x, stations)
    ax.set_ylim(0, 1.7)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("MAE (h)")
    ax.grid(axis="y", color="#E6EBEF", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=1,
              handlelength=0.8, borderpad=0.2, labelspacing=0.35)


def draw_paired(ax: plt.Axes, comparisons: list[tuple[str, dict, str]]) -> None:
    panel_label(ax, "D")
    ax.set_title("Paired MAE reduction", loc="left", pad=15,
                 fontsize=10.5, fontweight="normal")
    y_positions = np.arange(len(comparisons))[::-1]
    ax.axvline(0, color="#5A6570", lw=1.0)
    for (label, result, color), y in zip(comparisons, y_positions):
        center = result["mean_mae_improvement_hours"]
        low, high = result["bootstrap_95pct_interval_hours"]
        ax.plot([low, high], [y, y], color=color, lw=2.3, solid_capstyle="round")
        ax.plot([low, low], [y - 0.085, y + 0.085], color=color, lw=1.6)
        ax.plot([high, high], [y - 0.085, y + 0.085], color=color, lw=1.6)
        ax.scatter(center, y, s=70, color=color, edgecolor="white", lw=0.7, zorder=3)
        ax.text(high + 0.005, y, f"{center:+.3f}", ha="left", va="center",
                fontsize=8.3, color="#24303B")
    ax.set_yticks(y_positions, [r[0] for r in comparisons])
    ax.set_xlim(-0.095, 0.145)
    ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12])
    ax.set_xlabel("MAE reduction (h); positive favors second model")
    ax.grid(axis="x", color="#E6EBEF", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    for name, expected in EXPECTED.items():
        if sha256(DATA / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    valid = json.loads((DATA / "mlx_v3_comparison_valid.json").read_text())
    test = json.loads((DATA / "v3_final_test_comparison.json").read_text())[
        "sections"]["primary_clean_58_dates"]
    if valid["rows"] != 184 or test["rows"] != 232 or test["date_blocks"] != 58:
        raise ValueError("Unexpected validation or frozen-test size")
    valid_scores = {"physical": valid["metrics"]["physical"],
                    "conditional": valid["metrics"]["conditional"],
                    "base": valid["metrics"]["base_v3"],
                    "adapter": valid["metrics"]["adapter_v3"]}
    truth = {key(r): r["target"]["actual_full_load_hours"]
             for r in rows("daily_test.jsonl") if r["target_date"] != "2019-04-16"}
    conditional = {key(r): {"forecast_full_load_hours": r["conditioned_physics_full_load_hours"]}
                   for r in rows("conditional_test.jsonl") if key(r) in truth}
    base = {key(r): r for r in rows("mlx_v3_base_test.jsonl") if key(r) in truth}
    if len(truth) != 232 or set(truth) != set(conditional) or set(truth) != set(base):
        raise ValueError("Incomplete frozen-test prediction alignment")
    base_vs_conditional = paired_improvement(conditional, base, truth)
    comparisons = [
        ("Cond. physics → 9B base", base_vs_conditional, COLORS["base"]),
        ("Cond. physics → 9B LoRA", test["paired"]["adapter_vs_conditional"], COLORS["adapter"]),
        ("9B base → 9B LoRA", test["paired"]["adapter_vs_base"], COLORS["adapter"]),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#9BA8B3",
        "axes.linewidth": 0.75,
        "axes.labelcolor": "#24303B",
        "text.color": "#24303B",
        "xtick.color": "#485562",
        "ytick.color": "#485562",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "pv-paper-main-results-v1",
    })
    fig = plt.figure(figsize=(11.8, 6.6), facecolor="white", layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], wspace=0.10, hspace=0.18)
    ax_valid = fig.add_subplot(grid[0, 0])
    ax_test = fig.add_subplot(grid[0, 1])
    ax_station = fig.add_subplot(grid[1, 0])
    ax_paired = fig.add_subplot(grid[1, 1])
    draw_overall(ax_valid, valid_scores, "Validation (184 station-days)", "A")
    draw_overall(ax_test, test["metrics"], "Test (232 station-days)", "B")
    draw_station(ax_station, test["per_station"])
    draw_paired(ax_paired, comparisons)
    for ax in (ax_valid, ax_test, ax_station, ax_paired):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "research_main_results"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.15,
                metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.15,
                metadata={"Date": None, "Creator": "plot_research_main_results.py"})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    manifest = {
        "figure": "research_main_results",
        "source_sha256": EXPECTED,
        "validation_rows": valid["rows"],
        "frozen_test_rows": test["rows"],
        "frozen_test_date_blocks": test["date_blocks"],
        "base_vs_conditional_paired": base_vs_conditional,
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    }
    (FIGURES / "research_main_results_data.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
