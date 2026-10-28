from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        "--base-csv",
        type=Path,
        default=ROOT
        / "data/evidence/experiments/matched_300k/base_300k_seed_outcomes.csv",
    )
    parser.add_argument(
        "--extended",
        "--extended-csv",
        type=Path,
        default=ROOT
        / "data/evidence/experiments/g10_five_seed/five_seed_generation_summary.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figure1.png")
    args = parser.parse_args()
    rows = read_rows(args.base)
    pairs = {}
    for row in rows:
        pairs.setdefault(row["seed_index"], {})[row["arm"]] = 100 * float(
            row["G10_val_relabel_accuracy"]
        )
    if len(pairs) != 5 or any(
        set(pair) != {"clean_only", "recursive_replacement"} for pair in pairs.values()
    ):
        raise ValueError("Expected five paired Base G10 outcomes")
    extended = [r for r in read_rows(args.extended) if r["task"] == "extended"]
    extended.sort(key=lambda r: int(r["generation"]))
    if [int(r["generation"]) for r in extended] != list(range(1, 11)) or any(
        int(r["n_seeds"]) != 5 for r in extended
    ):
        raise ValueError(
            "Expected five-seed Extended summaries for generations 1 through 10"
        )
    plt.rcParams.update(
        {"font.size": 8, "axes.titlesize": 10, "axes.titleweight": "bold"}
    )
    fig, (base_ax, ext_ax) = plt.subplots(
        1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1, 1.17]}
    )
    blue, red = "#1473ae", "#ce4051"
    for pair in pairs.values():
        base_ax.plot(
            [0, 1],
            [pair["clean_only"], pair["recursive_replacement"]],
            color="#aab5c0",
            linewidth=0.65,
            zorder=1,
        )
        base_ax.scatter(
            [0],
            [pair["clean_only"]],
            s=13,
            color=blue,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
        base_ax.scatter(
            [1],
            [pair["recursive_replacement"]],
            s=15,
            marker="D",
            color=red,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
    for x, arm in [(0, "clean_only"), (1, "recursive_replacement")]:
        base_ax.scatter(
            x,
            mean(p[arm] for p in pairs.values()),
            marker="_",
            s=150,
            linewidths=1.8,
            color="#27333e",
            zorder=3,
        )
    base_ax.set_xlim(-0.3, 1.3)
    base_ax.set_ylim(40, 77)
    base_ax.set_yticks([40, 50, 60, 70])
    base_ax.set_xticks([0, 1], ["Real targets", "Recursive"])
    base_ax.set_ylabel("Validation accuracy (%)")
    base_ax.set_title("Base: G10 label accuracy", pad=14)
    base_ax.text(
        0.5,
        1.0,
        "Five matched seeds",
        transform=base_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#53616d",
    )
    generations = [int(r["generation"]) for r in extended]
    for key, label, color, marker in [
        ("control", "Real-target control", blue, "o"),
        ("recursive", "Recursive", red, "D"),
    ]:
        values = [100 * float(r[key + "_mean"]) for r in extended]
        errors = [100 * float(r[key + "_sd"]) for r in extended]
        ext_ax.plot(
            generations,
            values,
            marker=marker,
            markersize=2.5,
            linewidth=1.5,
            color=color,
            label=label,
        )
        ext_ax.fill_between(
            generations,
            [v - e for v, e in zip(values, errors)],
            [v + e for v, e in zip(values, errors)],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ext_ax.set_xlim(0.8, 10.2)
    ext_ax.set_ylim(60, 101)
    ext_ax.set_yticks([60, 70, 80, 90, 100])
    ext_ax.set_xticks([1, 3, 5, 7, 10])
    ext_ax.set_xlabel("Generation")
    ext_ax.set_title("Extended: valid next symbol", pad=6)
    for ax in (base_ax, ext_ax):
        ax.grid(axis="y", color="#d8dee5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#aab5c0")
        ax.tick_params(length=0, pad=3)
    handles, labels = ext_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
