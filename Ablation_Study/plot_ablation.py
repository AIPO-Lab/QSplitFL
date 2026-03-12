"""
Ablation Study Visualisation

Generates five figures from ablation study results:

  1. accuracy_bar_chart.png  — Final accuracy comparison (bar chart)
  2. convergence_curves.png  — Accuracy vs. round for all 8 configs
  3. split_layer_curves.png  — Split-layer selection vs. round
  4. results_heatmap.png     — Summary results as a colour-coded heatmap
  5. reward_curves.png       — Reward signal over training rounds

Usage
-----
  # After run_ablation.py has written CSVs to ./results:
  python plot_ablation.py

  # Or call from run_ablation.py (automatic):
  generate_all_plots(all_results, outdir, num_rounds)
"""

import os
import csv
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ablation_configs import ABLATION_CONFIGS, CONFIG_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (one colour per configuration, consistent across all plots)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    "#2196F3",   # 0 Full QSplitFL       — blue
    "#F44336",   # 1 Single-head (M=1)   — red
    "#4CAF50",   # 2 Committee M=5       — green
    "#FF9800",   # 3 No Decay            — orange
    "#9C27B0",   # 4 High Decay          — purple
    "#00BCD4",   # 5 Equal Weights       — cyan
    "#795548",   # 6 CPU-only State      — brown
    "#607D8B",   # 7 Random Split        — grey-blue
]

LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bar chart — final accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_chart(all_results: dict, outdir: str, num_rounds: int):
    """Bar chart comparing final test accuracy across all 8 configurations."""
    names  = list(all_results.keys())
    accs   = [all_results[n]["final_accuracy"] * 100 for n in names]
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), accs, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.2)

    # Annotate each bar
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([CONFIG_LABELS[i] if i < len(CONFIG_LABELS) else n
                        for i, n in enumerate(names)],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title(
        f"Ablation Study — Final Test Accuracy\n"
        f"(CIFAR-10, ResNet50, 10 clients, {num_rounds} rounds)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 6))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(outdir, "accuracy_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Convergence curves — accuracy vs. round
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(all_results: dict, outdir: str, num_rounds: int):
    """Line plot of test accuracy over training rounds for all configs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, result) in enumerate(all_results.items()):
        accs   = [a * 100 for a in result["accuracies"]]
        rounds = list(range(1, len(accs) + 1))
        label  = CONFIG_LABELS[i] if i < len(CONFIG_LABELS) else name
        ax.plot(rounds, accs,
                color=COLORS[i % len(COLORS)],
                linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                linewidth=1.8, label=label, alpha=0.9)

    ax.axhline(80, color="black", linewidth=1.0, linestyle=":", alpha=0.5,
               label="80% threshold")

    ax.set_xlabel("Training Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title(
        "Ablation Study — Accuracy Convergence over Training Rounds\n"
        "(CIFAR-10, ResNet50, 10 clients)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_xlim(1, num_rounds)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(outdir, "convergence_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Split-layer selection curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_split_layers(all_results: dict, outdir: str, num_rounds: int):
    """Line plot of selected split layer over training rounds."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (name, result) in enumerate(all_results.items()):
        splits = result["split_layers"]
        rounds = list(range(1, len(splits) + 1))
        label  = CONFIG_LABELS[i] if i < len(CONFIG_LABELS) else name
        ax.plot(rounds, splits,
                color=COLORS[i % len(COLORS)],
                linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                linewidth=1.6, label=label, alpha=0.85)

    ax.axhline(25, color="black", linewidth=1.0, linestyle=":",
               alpha=0.5, label="Min split (25) — Random baseline")

    ax.set_xlabel("Training Round", fontsize=11)
    ax.set_ylabel("Selected Split Layer", fontsize=11)
    ax.set_title(
        "Ablation Study — Split Layer Selection over Training Rounds\n"
        "(CIFAR-10, ResNet50, action space: layers 25–49)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(22, 52)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_xlim(1, num_rounds)

    plt.tight_layout()
    path = os.path.join(outdir, "split_layer_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Results heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_heatmap(all_results: dict, outdir: str, num_rounds: int):
    """
    Render ablation results as a colour-coded heatmap where each cell's
    colour intensity reflects the metric value (higher = better).
    """
    names  = list(all_results.keys())
    accs   = [all_results[n]["final_accuracy"] * 100 for n in names]
    splits = [all_results[n]["avg_split"]             for n in names]
    r80    = [all_results[n]["rounds_to_80"]           for n in names]

    # Normalise each column to [0, 1] for colour mapping
    def norm(vals, higher_is_better=True):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        normed = [(v - mn) / (mx - mn) for v in vals]
        return normed if higher_is_better else [1 - v for v in normed]

    acc_n   = norm(accs,   higher_is_better=True)
    split_n = norm(splits, higher_is_better=True)
    r80_n   = norm(r80,    higher_is_better=False)  # fewer rounds = better

    col_headers = [
        f"Final Acc (%)\n(CIFAR-10, {num_rounds}R)",
        "Avg Split Layer",
        "Rounds to 80%",
    ]
    data_vals  = [[f"{a:.2f}", f"{s:.1f}", str(r)]
                  for a, s, r in zip(accs, splits, r80)]
    data_norms = [[an, sn, rn]
                  for an, sn, rn in zip(acc_n, split_n, r80_n)]

    n_rows = len(names)
    n_cols = 3
    fig, ax = plt.subplots(figsize=(9, 0.55 * n_rows + 1.8))
    ax.axis("off")

    # Draw cells
    cell_h, cell_w = 0.9, 2.8
    x_starts = [0.5, 0.5 + cell_w, 0.5 + 2 * cell_w]
    y_starts  = [n_rows - i - 0.5 for i in range(n_rows)]

    cmap = plt.cm.RdYlGn

    for row_i, (name, vals, norms_row) in enumerate(
            zip(names, data_vals, data_norms)):
        y = y_starts[row_i]
        # Row label
        ax.text(-0.1, y, CONFIG_LABELS[row_i] if row_i < len(CONFIG_LABELS) else name,
                va="center", ha="right", fontsize=8.5, fontweight="bold")
        for col_j, (val, norm_val) in enumerate(zip(vals, norms_row)):
            x = x_starts[col_j]
            colour = cmap(norm_val)
            rect = mpatches.FancyBboxPatch(
                (x - cell_w / 2 + 0.05, y - cell_h / 2 + 0.05),
                cell_w - 0.1, cell_h - 0.1,
                boxstyle="round,pad=0.05",
                facecolor=colour, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            text_color = "black" if norm_val > 0.3 else "white"
            ax.text(x, y, val, va="center", ha="center",
                    fontsize=9.5, fontweight="bold", color=text_color)

    # Column headers
    for j, (header, x) in enumerate(zip(col_headers, x_starts)):
        ax.text(x, n_rows + 0.1, header, va="bottom", ha="center",
                fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlim(-2.5, 0.5 + 3 * cell_w)
    ax.set_ylim(-0.7, n_rows + 0.8)
    ax.set_title(
        "Ablation Study — Results Summary\n"
        "Green = better performance",
        fontsize=11, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    path = os.path.join(outdir, "results_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward curves (bonus)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rewards(all_results: dict, outdir: str, num_rounds: int):
    """Reward signal over training rounds — shows effect of λ on reward scale."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (name, result) in enumerate(all_results.items()):
        rws    = result["rewards"]
        rounds = list(range(1, len(rws) + 1))
        # Smooth with a window-5 moving average
        smoothed = np.convolve(rws, np.ones(5) / 5, mode="same")
        label  = CONFIG_LABELS[i] if i < len(CONFIG_LABELS) else name
        ax.plot(rounds, smoothed,
                color=COLORS[i % len(COLORS)],
                linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                linewidth=1.6, label=label, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)
    ax.set_xlabel("Training Round", fontsize=11)
    ax.set_ylabel("Reward (smoothed, window=5)", fontsize=11)
    ax.set_title(
        "Ablation Study — Reward Signal over Training Rounds\n"
        "(illustrates effect of λ on reward magnitude)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_xlim(1, num_rounds)

    plt.tight_layout()
    path = os.path.join(outdir, "reward_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master function called by run_ablation.py
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(all_results: dict, outdir: str, num_rounds: int):
    """Generate all five plots. Called automatically from run_ablation.py."""
    os.makedirs(outdir, exist_ok=True)
    print("\nGenerating plots…")
    plot_bar_chart(all_results, outdir, num_rounds)
    plot_convergence(all_results, outdir, num_rounds)
    plot_split_layers(all_results, outdir, num_rounds)
    plot_results_heatmap(all_results, outdir, num_rounds)
    plot_rewards(all_results, outdir, num_rounds)
    print(f"All plots saved to {os.path.abspath(outdir)}/")


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone usage: load CSVs from results/ and re-generate plots
# ─────────────────────────────────────────────────────────────────────────────

def load_results_from_csvs(results_dir: str) -> dict:
    """
    Reconstruct all_results dict by reading per-round CSVs written by
    run_ablation.py.  Useful if you want to regenerate plots without
    re-running training.
    """
    all_results = {}
    csv_files   = sorted(glob.glob(os.path.join(results_dir, "*_per_round.csv")))

    for csv_path in csv_files:
        config_name = (
            os.path.basename(csv_path)
            .replace("_per_round.csv", "")
            .replace("_", " ")
            .replace("lambda", "λ")
        )
        accuracies, losses, split_layers, rewards = [], [], [], []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                accuracies.append(float(row["accuracy"]))
                losses.append(float(row["cluster_loss"]))
                split_layers.append(int(row["split_layer"]))
                rewards.append(float(row["reward"]))

        if not accuracies:
            continue

        rounds_to_80 = len(accuracies)
        for i, acc in enumerate(accuracies):
            if acc >= 0.80:
                rounds_to_80 = i + 1
                break

        all_results[config_name] = {
            "accuracies":     accuracies,
            "losses":         losses,
            "split_layers":   split_layers,
            "rewards":        rewards,
            "final_accuracy": accuracies[-1],
            "avg_split":      float(np.mean(split_layers)),
            "rounds_to_80":   rounds_to_80,
        }

    return all_results


def main():
    p = argparse.ArgumentParser(
        description="Re-generate ablation plots from saved CSVs"
    )
    p.add_argument("--indir",  type=str, default="results",
                   help="Directory containing *_per_round.csv files")
    p.add_argument("--outdir", type=str, default="results",
                   help="Output directory for plots")
    p.add_argument("--rounds", type=int, default=100)
    args = p.parse_args()

    all_results = load_results_from_csvs(args.indir)
    if not all_results:
        print(f"No per-round CSV files found in '{args.indir}'.")
        print("Run  python run_ablation.py  first.")
        return

    print(f"Loaded results for {len(all_results)} configuration(s): "
          f"{list(all_results.keys())}")
    generate_all_plots(all_results, args.outdir, args.rounds)


if __name__ == "__main__":
    main()
