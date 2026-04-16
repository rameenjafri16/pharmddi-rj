"""
visualize_experiment.py

Generates publication-quality figures for the pathway vs Tanimoto retrieval
comparison experiment. Can be run at two stages:

  Stage 1 (after dataset preparation, before experiment):
    --stage dataset
    Produces exploratory plots about the structure of the two datasets.
    Helps us understand what we're working with before running the comparison.

  Stage 2 (after experiment):
    --stage results
    Produces the main comparison plots showing pathway vs Tanimoto performance.

All figures saved to: outputs/figures/experiment/

USAGE:
------
    # After dataset prep (explore the data):
    python scripts/visualize_experiment.py --stage dataset

    # After experiment (show results):
    python scripts/visualize_experiment.py --stage results

    # Both at once:
    python scripts/visualize_experiment.py --stage all

No GPU needed. Runs on login node in ~1 minute.
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

# ── Output directory ────────────────────────────────────────────────────────
FIGURES_DIR = Path("outputs/figures/experiment")

# ── Color palette ────────────────────────────────────────────────────────────
# Consistent colors used across all plots
COLORS = {
    "tanimoto":  "#2E86AB",   # blue  — structural similarity (Tanimoto)
    "pathway":   "#E84855",   # red   — biological pathway (our method)
    "dataset_a": "#3A3A3A",   # dark grey — strict dataset (Mohammadreza's)
    "dataset_b": "#7B9E87",   # sage green — relaxed dataset (ours)
    "head":      "#F4A261",   # orange — head classes (most common)
    "mid":       "#457B9D",   # steel blue — mid classes
    "tail":      "#A8DADC",   # light teal — tail classes (rarest)
}

# ── Style settings ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})


# ── Data loading helpers ─────────────────────────────────────────────────────

def load_dataset_summary(dataset_dir: str) -> dict:
    """Load the summary JSON produced by prepare_experiment_datasets.py"""
    path = Path(dataset_dir) / "dataset_summary.json"
    with open(path) as f:
        return json.load(f)


def load_train_df(dataset_dir: str) -> pd.DataFrame:
    """Load the training split for a dataset."""
    return pd.read_json(Path(dataset_dir) / "train.jsonl", lines=True)


def load_tier_map(dataset_dir: str) -> dict:
    """Load the {label_id -> 'head'|'mid'|'tail'} tier mapping."""
    with open(Path(dataset_dir) / "tier_map.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_label_map(dataset_dir: str) -> dict:
    """Load the {label_id -> interaction template text} mapping."""
    with open(Path(dataset_dir) / "label_map.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_experiment_results(results_path: str) -> dict:
    """Load the comparison results JSON produced by run_retrieval_comparison.py"""
    with open(results_path) as f:
        return json.load(f)


# ── Stage 1: Dataset exploration plots ──────────────────────────────────────

def plot_class_frequency_distribution(out_dir: Path):
    """
    Figure 1: Class frequency distribution for Dataset A and B.

    Shows the long-tail problem: a few head classes have hundreds of thousands
    of pairs, while tail classes have very few. Dataset B extends the tail.
    This motivates why Tanimoto retrieval struggles for rare classes --
    there simply aren't many structurally similar pairs to retrieve from.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Class Frequency Distribution: Long-Tail Problem in DDI Data",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, (dataset_dir, label, color) in zip(axes, [
        ("data/processed/dataset_A", "Dataset A (≥130 pairs, 129 classes)", COLORS["dataset_a"]),
        ("data/processed/dataset_B", "Dataset B (≥20 pairs, 194 classes)",  COLORS["dataset_b"]),
    ]):
        train_df = load_train_df(dataset_dir)
        tier_map = load_tier_map(dataset_dir)

        # Count pairs per class, sort by frequency
        class_counts = train_df["label"].value_counts().sort_values(ascending=False)
        ranks = np.arange(1, len(class_counts) + 1)
        counts = class_counts.values

        # Color bars by tier
        bar_colors = [
            COLORS["head"]  if tier_map.get(lbl) == "head" else
            COLORS["mid"]   if tier_map.get(lbl) == "mid"  else
            COLORS["tail"]
            for lbl in class_counts.index
        ]

        ax.bar(ranks, counts, color=bar_colors, width=1.0, alpha=0.85)

        # Add tier boundary lines
        n_head = sum(1 for v in tier_map.values() if v == "head")
        n_mid  = sum(1 for v in tier_map.values() if v == "mid")
        ax.axvline(x=n_head + 0.5, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5)
        ax.axvline(x=n_head + n_mid + 0.5, color="black", linestyle="--",
                   linewidth=0.8, alpha=0.5)

        # Tier labels at top of plot
        ax.text(n_head / 2, ax.get_ylim()[1] * 0.92,
                f"Head\n({n_head} classes)", ha="center", fontsize=9,
                color=COLORS["head"], fontweight="bold")
        ax.text(n_head + n_mid / 2, ax.get_ylim()[1] * 0.85,
                f"Mid\n({n_mid} classes)", ha="center", fontsize=9,
                color=COLORS["mid"], fontweight="bold")
        n_tail = len(class_counts) - n_head - n_mid
        ax.text(n_head + n_mid + n_tail / 2, ax.get_ylim()[1] * 0.78,
                f"Tail\n({n_tail} classes)", ha="center", fontsize=9,
                color=COLORS["tail"], fontweight="bold")

        ax.set_title(label, pad=10)
        ax.set_xlabel("Class rank (sorted by frequency, 1 = most common)")
        ax.set_ylabel("Training pairs per class")
        ax.set_xlim(0, len(class_counts) + 1)

    plt.tight_layout()
    path = out_dir / "fig1_class_frequency_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dataset_comparison_overview(out_dir: Path):
    """
    Figure 2: Side-by-side comparison of Dataset A vs B key statistics.

    Shows what we gain by relaxing the filtering threshold:
    65 additional rare classes that Mohammadreza excluded from PharmCoT.
    """
    summary_a = load_dataset_summary("data/processed/dataset_A")
    summary_b = load_dataset_summary("data/processed/dataset_B")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Dataset A vs Dataset B: What Relaxed Filtering Adds",
        fontsize=14, fontweight="bold",
    )

    bar_width = 0.35
    x = np.array([0])

    # ── Plot 1: Number of classes ──────────────────────────────────────────
    ax = axes[0]
    bars_a = ax.bar(x - bar_width/2, [summary_a["n_classes"]],
                    bar_width, color=COLORS["dataset_a"],
                    label="Dataset A (≥130)", alpha=0.85)
    bars_b = ax.bar(x + bar_width/2, [summary_b["n_classes"]],
                    bar_width, color=COLORS["dataset_b"],
                    label="Dataset B (≥20)", alpha=0.85)
    ax.bar_label(bars_a, fmt="%d", padding=3, fontsize=11, fontweight="bold")
    ax.bar_label(bars_b, fmt="%d", padding=3, fontsize=11, fontweight="bold")
    ax.set_title("Interaction Classes")
    ax.set_ylabel("Number of classes")
    ax.set_xticks([])
    ax.set_ylim(0, summary_b["n_classes"] * 1.2)
    ax.legend()

    # Add annotation showing the gain
    gain = summary_b["n_classes"] - summary_a["n_classes"]
    ax.annotate(
        f"+{gain} rare classes\nfrom Dataset B",
        xy=(bar_width/2, summary_b["n_classes"]),
        xytext=(bar_width/2 + 0.3, summary_b["n_classes"] * 0.7),
        arrowprops=dict(arrowstyle="->", color=COLORS["dataset_b"]),
        color=COLORS["dataset_b"], fontsize=9, fontweight="bold",
    )

    # ── Plot 2: Tier breakdown ─────────────────────────────────────────────
    ax = axes[1]
    tiers = ["Head", "Mid", "Tail"]
    tier_keys = ["n_head_classes", "n_mid_classes", "n_tail_classes"]
    tier_colors = [COLORS["head"], COLORS["mid"], COLORS["tail"]]

    x_tiers = np.arange(len(tiers))
    vals_a = [summary_a[k] for k in tier_keys]
    vals_b = [summary_b[k] for k in tier_keys]

    bars_a = ax.bar(x_tiers - bar_width/2, vals_a, bar_width,
                    color=tier_colors, alpha=0.5,
                    label="Dataset A", edgecolor=COLORS["dataset_a"],
                    linewidth=1.5)
    bars_b = ax.bar(x_tiers + bar_width/2, vals_b, bar_width,
                    color=tier_colors, alpha=0.85,
                    label="Dataset B", edgecolor=COLORS["dataset_b"],
                    linewidth=1.5)
    ax.bar_label(bars_a, fmt="%d", padding=3, fontsize=10)
    ax.bar_label(bars_b, fmt="%d", padding=3, fontsize=10, fontweight="bold")
    ax.set_title("Classes by Frequency Tier")
    ax.set_ylabel("Number of classes")
    ax.set_xticks(x_tiers)
    ax.set_xticklabels(tiers)
    ax.legend()

    # ── Plot 3: Training pairs per tier ───────────────────────────────────
    ax = axes[2]
    train_a = load_train_df("data/processed/dataset_A")
    train_b = load_train_df("data/processed/dataset_B")
    tier_map_a = load_tier_map("data/processed/dataset_A")
    tier_map_b = load_tier_map("data/processed/dataset_B")

    def count_pairs_by_tier(df, tier_map):
        counts = {"head": 0, "mid": 0, "tail": 0}
        for label, count in df["label"].value_counts().items():
            tier = tier_map.get(int(label), "tail")
            counts[tier] += count
        return counts

    pairs_a = count_pairs_by_tier(train_a, tier_map_a)
    pairs_b = count_pairs_by_tier(train_b, tier_map_b)

    vals_a = [pairs_a[t.lower()] for t in tiers]
    vals_b = [pairs_b[t.lower()] for t in tiers]

    bars_a = ax.bar(x_tiers - bar_width/2, vals_a, bar_width,
                    color=tier_colors, alpha=0.5,
                    label="Dataset A", edgecolor=COLORS["dataset_a"],
                    linewidth=1.5)
    bars_b = ax.bar(x_tiers + bar_width/2, vals_b, bar_width,
                    color=tier_colors, alpha=0.85,
                    label="Dataset B", edgecolor=COLORS["dataset_b"],
                    linewidth=1.5)

    def fmt_k(x, pos=None):
        return f"{int(x/1000)}K" if x >= 1000 else str(int(x))

    ax.bar_label(bars_a, labels=[fmt_k(v) for v in vals_a], padding=3, fontsize=9)
    ax.bar_label(bars_b, labels=[fmt_k(v) for v in vals_b], padding=3,
                 fontsize=9, fontweight="bold")
    ax.set_title("Training Pairs by Frequency Tier")
    ax.set_ylabel("Training pairs")
    ax.set_xticks(x_tiers)
    ax.set_xticklabels(tiers)
    ax.legend()

    plt.tight_layout()
    path = out_dir / "fig2_dataset_comparison_overview.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tanimoto_similarity_by_tier(out_dir: Path):
    """
    Figure 3: Distribution of Tanimoto similarity scores by class frequency tier.

    KEY MOTIVATION PLOT: If tail classes have lower mean Tanimoto similarity
    to their nearest neighbours, that proves Tanimoto retrieval is structurally
    blind for rare classes -- directly motivating pathway retrieval.

    Uses the precomputed similarity matrix from build_fingerprints.py.
    """
    import pickle

    sim_path = Path("data/processed/drug_similarity_matrix.npz")
    fp_path  = Path("data/processed/drug_fingerprints.pkl")
    id_path  = Path("data/processed/drug_id_order.json")

    if not all(p.exists() for p in [sim_path, fp_path, id_path]):
        print("  SKIP fig3: fingerprint files not found")
        return

    print("  Loading similarity matrix (this may take ~30 seconds)...")
    sim_data = np.load(sim_path)
    sim_matrix = sim_data["matrix"]
    with open(id_path) as f:
        drug_id_order = json.load(f)
    id_to_idx = {did: i for i, did in enumerate(drug_id_order)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Tanimoto Similarity Between Drug Pairs by Class Tier\n"
        "(Low similarity = Tanimoto retrieval finds poor structural matches)",
        fontsize=13, fontweight="bold",
    )

    for ax, (dataset_dir, ds_label) in zip(axes, [
        ("data/processed/dataset_A", "Dataset A (≥130 pairs/class)"),
        ("data/processed/dataset_B", "Dataset B (≥20 pairs/class)"),
    ]):
        train_df = load_train_df(dataset_dir)
        tier_map = load_tier_map(dataset_dir)

        # For each pair, compute the Tanimoto similarity between drug1 and drug2
        # This tells us how structurally similar the two interacting drugs are
        tier_similarities = {"head": [], "mid": [], "tail": []}

        # Sample up to 500 pairs per tier for speed
        rng = np.random.RandomState(42)
        for tier in ("head", "mid", "tail"):
            tier_rows = train_df[
                train_df["label"].map(lambda l: tier_map.get(int(l), "tail")) == tier
            ]
            if len(tier_rows) > 500:
                tier_rows = tier_rows.sample(n=500, random_state=rng)

            for _, row in tier_rows.iterrows():
                i1 = id_to_idx.get(row["drug1_id"])
                i2 = id_to_idx.get(row["drug2_id"])
                if i1 is not None and i2 is not None:
                    sim = float(sim_matrix[i1, i2])
                    tier_similarities[tier].append(sim)

        # Plot distributions
        tier_labels = {"head": "Head", "mid": "Mid", "tail": "Tail"}
        tier_color_list = [COLORS["head"], COLORS["mid"], COLORS["tail"]]
        positions = [1, 2, 3]

        parts = ax.violinplot(
            [tier_similarities[t] for t in ("head", "mid", "tail")],
            positions=positions,
            showmedians=True,
            showextrema=True,
        )
        for i, (pc, color) in enumerate(zip(parts["bodies"], tier_color_list)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # Add mean annotations
        for pos, tier, color in zip(positions, ("head", "mid", "tail"),
                                    tier_color_list):
            vals = tier_similarities[tier]
            if vals:
                mean_val = np.mean(vals)
                ax.text(pos, mean_val + 0.02, f"μ={mean_val:.3f}",
                        ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_title(ds_label)
        ax.set_ylabel("Tanimoto similarity between the two interacting drugs")
        ax.set_xticks(positions)
        ax.set_xticklabels(["Head\nclasses", "Mid\nclasses", "Tail\nclasses"])
        ax.set_ylim(-0.05, 1.0)
        ax.axhline(y=0.1, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(3.4, 0.1, "mean\nbaseline", fontsize=8, color="grey", va="center")

    plt.tight_layout()
    path = out_dir / "fig3_tanimoto_similarity_by_tier.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pathway_coverage_by_tier(out_dir: Path):
    """
    Figure 4: Pathway annotation coverage by class tier.

    Shows what fraction of drug pairs have enzyme/transporter/target data
    available for pathway retrieval, broken down by tier.

    We expect pathway coverage to be reasonably high even for tail classes,
    since enzyme/transporter annotations are independent of interaction
    frequency. This is the key advantage over Tanimoto (SMILES-dependent).
    """
    import json

    with open("data/processed/drug_profiles.json") as f:
        profiles = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Pathway Annotation Coverage vs Tanimoto (SMILES) Coverage by Tier\n"
        "(Higher coverage = more pairs can use this retrieval strategy)",
        fontsize=13, fontweight="bold",
    )

    for ax, (dataset_dir, ds_label) in zip(axes, [
        ("data/processed/dataset_A", "Dataset A"),
        ("data/processed/dataset_B", "Dataset B"),
    ]):
        train_df = load_train_df(dataset_dir)
        tier_map = load_tier_map(dataset_dir)

        tiers = ["Head", "Mid", "Tail"]
        tanimoto_coverage = []
        pathway_coverage  = []

        for tier in ("head", "mid", "tail"):
            tier_rows = train_df[
                train_df["label"].map(lambda l: tier_map.get(int(l), "tail")) == tier
            ]
            if len(tier_rows) == 0:
                tanimoto_coverage.append(0)
                pathway_coverage.append(0)
                continue

            # Tanimoto coverage: both drugs must have SMILES
            n_tan = sum(
                1 for _, row in tier_rows.iterrows()
                if profiles.get(row["drug1_id"], {}).get("smiles")
                and profiles.get(row["drug2_id"], {}).get("smiles")
            )

            # Pathway coverage: both drugs must have at least one of
            # enzymes, transporters, or targets
            n_path = sum(
                1 for _, row in tier_rows.iterrows()
                if any([
                    profiles.get(row["drug1_id"], {}).get("enzymes"),
                    profiles.get(row["drug1_id"], {}).get("transporters"),
                    profiles.get(row["drug1_id"], {}).get("targets"),
                ])
                and any([
                    profiles.get(row["drug2_id"], {}).get("enzymes"),
                    profiles.get(row["drug2_id"], {}).get("transporters"),
                    profiles.get(row["drug2_id"], {}).get("targets"),
                ])
            )

            tanimoto_coverage.append(100 * n_tan / len(tier_rows))
            pathway_coverage.append(100 * n_path / len(tier_rows))

        x = np.arange(len(tiers))
        bar_width = 0.35

        bars_tan  = ax.bar(x - bar_width/2, tanimoto_coverage, bar_width,
                           color=COLORS["tanimoto"], alpha=0.85,
                           label="Tanimoto (needs SMILES)")
        bars_path = ax.bar(x + bar_width/2, pathway_coverage, bar_width,
                           color=COLORS["pathway"], alpha=0.85,
                           label="Pathway (needs enzyme/transporter/target)")

        ax.bar_label(bars_tan,  fmt="%.1f%%", padding=3, fontsize=9)
        ax.bar_label(bars_path, fmt="%.1f%%", padding=3, fontsize=9,
                     fontweight="bold")

        ax.set_title(ds_label)
        ax.set_ylabel("% of pairs with coverage for this retrieval strategy")
        ax.set_xticks(x)
        ax.set_xticklabels(tiers)
        ax.set_ylim(0, 115)
        ax.legend()
        ax.axhline(y=100, color="grey", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    path = out_dir / "fig4_coverage_by_tier.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Stage 2: Experiment results plots ────────────────────────────────────────

def plot_mor_comparison(results: dict, out_dir: Path):
    """
    Figure 5: Main result — Mechanistic Overlap Rate by tier and dataset.

    Pathway retrieval achieves ~99% MOR vs Tanimoto's ~80-84%.
    Y-axis starts at 0.6 to make the difference clearly visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Mechanistic Overlap Rate: Pathway vs Tanimoto Retrieval\n"
        "Fraction of retrieved examples sharing a biological pathway node with the query pair",
        fontsize=13, fontweight="bold",
    )

    tiers = ["Head", "Mid", "Tail"]
    bar_width = 0.35

    for ax, dataset_key in zip(axes, ["dataset_A", "dataset_B"]):
        ds_results = results.get(dataset_key, {})
        ds_label = ("Dataset A (≥130 pairs/class)\nMohammadreza's setup"
                    if dataset_key == "dataset_A"
                    else "Dataset B (≥20 pairs/class)\nIncludes 65 rare classes")

        tan_vals  = [ds_results.get(f"{t.lower()}_tanimoto_mor", 0) for t in tiers]
        path_vals = [ds_results.get(f"{t.lower()}_pathway_mor",  0) for t in tiers]

        x = np.arange(len(tiers))

        # Use a zoomed-in y-axis starting at 0.6 so the gap is clearly visible
        # (all values are between 0.79 and 0.999 — starting at 0 would make
        # the difference look trivially small)
        ax.set_ylim(0.6, 1.08)

        bars_tan  = ax.bar(x - bar_width/2, tan_vals,  bar_width,
                           color=COLORS["tanimoto"], alpha=0.85,
                           label="Tanimoto (structural similarity)",
                           bottom=0)
        bars_path = ax.bar(x + bar_width/2, path_vals, bar_width,
                           color=COLORS["pathway"], alpha=0.85,
                           label="Pathway (biological annotation)",
                           bottom=0)

        # Value labels on top of each bar
        for xi, val in enumerate(tan_vals):
            ax.text(xi - bar_width/2, val + 0.004, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    color=COLORS["tanimoto"], fontweight="bold")
        for xi, val in enumerate(path_vals):
            ax.text(xi + bar_width/2, val + 0.004, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    color=COLORS["pathway"], fontweight="bold")

        # Delta annotations between bars
        for xi, (t, p) in enumerate(zip(tan_vals, path_vals)):
            delta = p - t
            ax.annotate(
                f"Δ={delta:+.3f}",
                xy=(xi, (t + p) / 2),
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", alpha=0.8),
            )

        ax.set_title(ds_label, pad=10)
        ax.set_ylabel("Mechanistic Overlap Rate (MOR)")
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"Head\n({ds_results.get('head_n_classes', '?')} classes)",
            f"Mid\n({ds_results.get('mid_n_classes', '?')} classes)",
            f"Tail\n({ds_results.get('tail_n_classes', '?')} classes)",
        ])

        # Reference line at 1.0
        ax.axhline(y=1.0, color="grey", linewidth=0.8,
                   linestyle=":", alpha=0.6)
        ax.text(2.6, 1.002, "Perfect", fontsize=8, color="grey", va="bottom")

        ax.legend(loc="lower right", fontsize=9)

        # Broken axis indicator to show y doesn't start at 0
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
        ax.text(-0.5, 0.605, "↑ axis\nbreaks\nat 0.6",
                fontsize=7, color="grey", va="bottom", ha="left")

    plt.tight_layout()
    path = out_dir / "fig5_mor_comparison_by_tier.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_delta_vs_frequency(results: dict, out_dir: Path):
    """
    Figure 6: Per-class MOR delta (pathway - Tanimoto) vs class frequency.

    If our hypothesis is correct, we should see a negative correlation:
    as classes get rarer (lower frequency), pathway retrieval's advantage
    over Tanimoto grows. This is the most compelling single plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Per-Class MOR Delta (Pathway − Tanimoto) vs Class Frequency\n"
        "(Negative correlation = pathway retrieval helps most for rare classes)",
        fontsize=13, fontweight="bold",
    )

    for ax, dataset_key in zip(axes, ["dataset_A", "dataset_B"]):
        per_class = results.get(dataset_key, {}).get("per_class_results", [])
        if not per_class:
            ax.text(0.5, 0.5, "Results not yet available",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        freqs  = [r["class_frequency"] for r in per_class]
        deltas = [r["pathway_mor"] - r["tanimoto_mor"] for r in per_class]
        tiers  = [r["tier"] for r in per_class]

        tier_color_map = {
            "head": COLORS["head"],
            "mid":  COLORS["mid"],
            "tail": COLORS["tail"],
        }
        point_colors = [tier_color_map.get(t, "grey") for t in tiers]

        ax.scatter(freqs, deltas, c=point_colors, alpha=0.7, s=40,
                   edgecolors="white", linewidths=0.5)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Trend line
        if len(freqs) > 5:
            z = np.polyfit(np.log1p(freqs), deltas, 1)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(freqs), max(freqs), 100)
            ax.plot(x_smooth, p(np.log1p(x_smooth)),
                    color="black", linewidth=1.5, linestyle="-", alpha=0.6,
                    label="Trend")

        # Legend
        patches = [
            mpatches.Patch(color=COLORS["head"], label="Head classes"),
            mpatches.Patch(color=COLORS["mid"],  label="Mid classes"),
            mpatches.Patch(color=COLORS["tail"], label="Tail classes"),
        ]
        ax.legend(handles=patches, fontsize=9)

        ds_label = "A (strict)" if dataset_key == "dataset_A" else "B (relaxed)"
        ax.set_title(f"Dataset {ds_label}")
        ax.set_xlabel("Class frequency (training pairs)")
        ax.set_ylabel("MOR delta (pathway − Tanimoto)")
        ax.set_xscale("log")

    plt.tight_layout()
    path = out_dir / "fig6_delta_vs_frequency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_coverage_vs_mor(results: dict, out_dir: Path):
    """
    Figure 7: Coverage AND MOR side by side — the double win.

    The surprising finding: pathway retrieval wins on BOTH coverage AND
    mechanistic relevance. Expected Tanimoto to have higher coverage
    (more drugs have SMILES than pathway annotations) but combined
    enzyme+transporter+target coverage is 98-99.9% vs Tanimoto's 81-84%.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Coverage and MOR: Pathway Retrieval Wins on Both Dimensions\n"
        "Solid bars = coverage (% pairs with examples) | "
        "Hatched bars = MOR x100 (% retrieved examples that are mechanistically relevant)",
        fontsize=11, fontweight="bold",
    )

    tiers = ["Head", "Mid", "Tail"]
    bar_width = 0.2
    x = np.arange(len(tiers))

    for ax, dataset_key in zip(axes, ["dataset_A", "dataset_B"]):
        ds_results = results.get(dataset_key, {})
        ds_label = ("Dataset A (≥130 pairs/class)"
                    if dataset_key == "dataset_A"
                    else "Dataset B (≥20 pairs/class)")

        tan_cov  = [ds_results.get(f"{t.lower()}_tanimoto_coverage", 0) for t in tiers]
        path_cov = [ds_results.get(f"{t.lower()}_pathway_coverage",  0) for t in tiers]
        tan_mor  = [ds_results.get(f"{t.lower()}_tanimoto_mor", 0) * 100 for t in tiers]
        path_mor = [ds_results.get(f"{t.lower()}_pathway_mor",  0) * 100 for t in tiers]

        # Four groups: Tan coverage, Path coverage, Tan MOR%, Path MOR%
        configs = [
            (x - 1.5*bar_width, tan_cov,  COLORS["tanimoto"], 0.5, "",    "Tanimoto coverage"),
            (x - 0.5*bar_width, path_cov, COLORS["pathway"],  0.5, "",    "Pathway coverage"),
            (x + 0.5*bar_width, tan_mor,  COLORS["tanimoto"], 0.9, "///", "Tanimoto MOR×100"),
            (x + 1.5*bar_width, path_mor, COLORS["pathway"],  0.9, "///", "Pathway MOR×100"),
        ]

        for pos, vals, color, alpha, hatch, label in configs:
            bars = ax.bar(pos, vals, bar_width, color=color,
                          alpha=alpha, hatch=hatch, label=label,
                          edgecolor="white" if not hatch else color)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if "Pathway" in label else "normal",
                    color=color,
                )

        ax.set_title(ds_label, pad=10)
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"Head\n({ds_results.get('head_n_classes', '?')} classes)",
            f"Mid\n({ds_results.get('mid_n_classes', '?')} classes)",
            f"Tail\n({ds_results.get('tail_n_classes', '?')} classes)",
        ])
        ax.set_ylim(0, 112)
        ax.axhline(y=100, color="grey", linewidth=0.6, linestyle=":")
        ax.legend(fontsize=8, loc="lower right", ncol=2)

    plt.tight_layout()
    path = out_dir / "fig7_coverage_and_mor.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_summary_table(results: dict, out_dir: Path):
    """
    Figure 8: Clean summary table of all key numbers.

    A single figure you can drop directly into a paper or presentation
    showing all the key metrics side by side.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    fig.suptitle(
        "Summary: Pathway RAG vs Tanimoto RAG — Key Metrics",
        fontsize=14, fontweight="bold",
    )

    rows = []
    col_labels = [
        "Dataset", "Tier", "Classes",
        "Tanimoto Coverage", "Pathway Coverage",
        "Tanimoto MOR", "Pathway MOR", "Delta (P−T)", "Winner",
    ]

    for dataset_key, ds_label in [
        ("dataset_A", "A (≥130)"),
        ("dataset_B", "B (≥20)"),
    ]:
        ds_results = results.get(dataset_key, {})
        for tier_key, tier_label in [
            ("head", "Head"), ("mid", "Mid"), ("tail", "Tail")
        ]:
            t_cov  = ds_results.get(f"{tier_key}_tanimoto_coverage", 0)
            p_cov  = ds_results.get(f"{tier_key}_pathway_coverage",  0)
            t_mor  = ds_results.get(f"{tier_key}_tanimoto_mor", 0)
            p_mor  = ds_results.get(f"{tier_key}_pathway_mor",  0)
            delta  = p_mor - t_mor
            # Categorize by magnitude since pathway wins everywhere
            if delta > 0.20:    winner = "Pathway ✓✓✓"
            elif delta > 0.15:  winner = "Pathway ✓✓"
            elif delta > 0.05:  winner = "Pathway ✓"
            elif delta < -0.01: winner = "Tanimoto ✓"
            else:               winner = "Tied"
            n_cls  = ds_results.get(f"{tier_key}_n_classes", "?")

            rows.append([
                ds_label, tier_label, str(n_cls),
                f"{t_cov:.1f}%", f"{p_cov:.1f}%",
                f"{t_mor:.3f}", f"{p_mor:.3f}",
                f"{delta:+.3f}", winner,
            ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2E2E2E")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color the Delta and Winner columns
    for i, row in enumerate(rows, start=1):
        delta_val = float(row[7])
        if delta_val > 0.01:
            table[i, 7].set_facecolor("#d4edda")   # light green
            table[i, 8].set_facecolor("#d4edda")
        elif delta_val < -0.01:
            table[i, 7].set_facecolor("#f8d7da")   # light red
            table[i, 8].set_facecolor("#f8d7da")
        else:
            table[i, 7].set_facecolor("#fff3cd")   # light yellow
            table[i, 8].set_facecolor("#fff3cd")

        # Alternate row shading
        if i % 2 == 0:
            for j in range(7):
                if table[i, j].get_facecolor() == (1, 1, 1, 1):
                    table[i, j].set_facecolor("#f8f9fa")

    plt.tight_layout()
    path = out_dir / "fig8_summary_table.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment visualizations"
    )
    parser.add_argument(
        "--stage",
        choices=["dataset", "results", "all"],
        default="dataset",
        help=(
            "dataset: exploratory plots after data prep (no GPU needed). "
            "results: comparison plots after running the experiment. "
            "all: both."
        ),
    )
    parser.add_argument(
        "--results-file",
        default="outputs/experiments/retrieval_comparison/comparison_results.json",
        help="Path to comparison results JSON (only needed for --stage results)"
    )
    args = parser.parse_args()

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {FIGURES_DIR}/\n")

    # ── Stage 1: Dataset exploration ─────────────────────────────────────────
    if args.stage in ("dataset", "all"):
        print("=== Stage 1: Dataset Exploration Plots ===\n")

        print("Figure 1: Class frequency distribution...")
        plot_class_frequency_distribution(FIGURES_DIR)

        print("Figure 2: Dataset A vs B overview...")
        plot_dataset_comparison_overview(FIGURES_DIR)

        print("Figure 3: Tanimoto similarity by tier...")
        plot_tanimoto_similarity_by_tier(FIGURES_DIR)

        print("Figure 4: Pathway vs Tanimoto coverage by tier...")
        plot_pathway_coverage_by_tier(FIGURES_DIR)

        print("\nDataset exploration complete.")
        print(f"Figures in: {FIGURES_DIR}/")
        print("\nThese plots show:")
        print("  Fig 1: The long-tail structure of DDI data")
        print("  Fig 2: What Dataset B adds over Dataset A (65 rare classes)")
        print("  Fig 3: Why Tanimoto struggles for rare classes")
        print("  Fig 4: Coverage comparison for each retrieval strategy")

    # ── Stage 2: Experiment results ───────────────────────────────────────────
    if args.stage in ("results", "all"):
        print("\n=== Stage 2: Experiment Results Plots ===\n")

        results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            print("Run the comparison experiment first:")
            print("  python scripts/run_retrieval_comparison.py")
            return

        results = load_experiment_results(str(results_path))

        print("Figure 5: MOR comparison by tier (main result)...")
        plot_mor_comparison(results, FIGURES_DIR)

        print("Figure 6: Per-class delta vs frequency (trend plot)...")
        plot_delta_vs_frequency(results, FIGURES_DIR)

        print("Figure 7: Coverage vs MOR tradeoff...")
        plot_coverage_vs_mor(results, FIGURES_DIR)

        print("Figure 8: Summary table...")
        plot_summary_table(results, FIGURES_DIR)

        print("\nResults visualization complete.")


if __name__ == "__main__":
    main()
