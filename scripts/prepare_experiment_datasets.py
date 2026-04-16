"""
prepare_experiment_datasets.py

Builds two filtered datasets from the raw DrugBank extraction for our
pathway vs Tanimoto retrieval comparison experiment.

WHY TWO DATASETS?
-----------------
Dataset A (strict, >= 130 pairs/class):
    This is exactly what Mohammadreza uses in PharmCoT. ~129 classes.
    We use this to confirm our experiment is comparable to his pipeline.

Dataset B (relaxed, >= 20 pairs/class):
    This includes rare interaction types that Mohammadreza excluded.
    ~200+ classes. We use this to test whether pathway retrieval helps
    specifically for rare classes -- classes Tanimoto struggles with
    because there are few structurally similar training pairs available.

The core hypothesis we are testing:
    Pathway retrieval's advantage over Tanimoto is LARGEST for rare
    (tail) classes in Dataset B. If true, this justifies expanding the
    PharmCoT pipeline to include rare classes using pathway retrieval.

WHAT THIS SCRIPT DOES:
----------------------
1. Loads the raw extracted interactions and drug profiles
2. For each threshold (130 and 20), filters to classes with enough pairs
3. Also filters out pairs where drugs have no useful pharmacological data
   (description, mechanism, enzymes, targets, or transporters) -- these
   would produce bad teacher traces anyway
4. Remaps labels to contiguous IDs 1..N
5. Assigns each class to a coarse interaction category
6. Does an 80/20 stratified train/test split
7. Applies a per-class training cap (5000) to prevent head-class dominance
8. Saves everything to clearly labelled output directories

OUTPUTS:
--------
data/processed/dataset_A/   (strict >= 130 pairs/class)
    train.jsonl
    test.jsonl
    label_map.json
    coarse_category_map.json
    dataset_summary.json    <- key stats for quick reference

data/processed/dataset_B/   (relaxed >= 20 pairs/class)
    train.jsonl
    test.jsonl
    label_map.json
    coarse_category_map.json
    dataset_summary.json

USAGE:
------
    python scripts/prepare_experiment_datasets.py

Runtime: ~5 minutes on the login node (no GPU needed)
"""

import json
import os
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────

# Where the raw extracted data lives (output of extract_dataset_from_xml.py)
RAW_DATA_DIR = "data/processed"

# Where to save each dataset
DATASET_A_DIR = "data/processed/dataset_A"   # strict filtering
DATASET_B_DIR = "data/processed/dataset_B"   # relaxed filtering

# Filtering thresholds
DATASET_A_MIN_PAIRS = 130   # Mohammadreza's exact threshold -> ~129 classes
DATASET_B_MIN_PAIRS = 20    # Our relaxed threshold -> ~200+ classes

# Training cap: max pairs per class in the training set
# Prevents the model from being dominated by a few very common interactions.
# The largest class has 180K pairs -- without a cap it would swamp everything.
MAX_TRAIN_PER_CLASS = 5000

# Train/test split ratio
TRAIN_RATIO = 0.80

# Random seed for reproducibility
SEED = 42

# Pharmacological fields that make a drug profile "useful".
# Both drugs in a pair must have at least one of these, otherwise the teacher
# model has no pharmacological information to reason from and will hallucinate.
USEFUL_PROFILE_FIELDS = [
    "description",
    "mechanism_of_action",
    "enzymes",
    "targets",
    "transporters",
]

# Coarse interaction category keywords.
# Maps fine-grained interaction templates to one of 18 broad categories.
# Used to group classes for the head/mid/tail tier analysis.
CATEGORY_KEYWORDS = {
    "metabolism_decrease":  ["metabolism of #drug2 can be decreased",
                             "metabolism of #drug1 can be decreased"],
    "metabolism_increase":  ["metabolism of #drug2 can be increased",
                             "metabolism of #drug1 can be increased"],
    "serum_increase":       ["serum concentration of #drug2 can be increased",
                             "serum concentration of #drug1 can be increased",
                             "serum concentration of the active metabolite"],
    "serum_decrease":       ["serum concentration of #drug2 can be decreased",
                             "serum concentration of #drug1 can be decreased"],
    "adverse_effects":      ["risk or severity of adverse effects"],
    "efficacy_decrease":    ["therapeutic efficacy of #drug2 can be decreased",
                             "therapeutic efficacy of #drug1 can be decreased"],
    "efficacy_increase":    ["therapeutic efficacy of #drug2 can be increased",
                             "therapeutic efficacy of #drug1 can be increased"],
    "excretion_decrease":   ["may decrease the excretion rate",
                             "excretion of #drug2 can be decreased",
                             "excretion of #drug1 can be decreased"],
    "excretion_increase":   ["may increase the excretion rate",
                             "excretion of #drug2 can be increased",
                             "excretion of #drug1 can be increased"],
    "absorption_decrease":  ["absorption of #drug2 can be decreased",
                             "absorption of #drug1 can be decreased"],
    "absorption_increase":  ["absorption of #drug2 can be increased",
                             "absorption of #drug1 can be increased"],
    "qtc_cardiac":          ["qtc-prolonging", "qtc interval", "cardiac",
                             "arrhythmia", "torsade"],
    "cns_effects":          ["cns depressant", "serotonergic", "sedation",
                             "respiratory depression"],
    "bleeding":             ["hemorrhagic", "bleeding", "anticoagulant"],
    "nephrotoxicity":       ["nephrotoxic", "renal"],
    "hepatotoxicity":       ["hepatotoxic", "liver"],
    "hypotension":          ["hypotensive"],
    "hyperkalemia":         ["hyperkalemic", "hyperkalemia"],
}


# ── Helper functions ───────────────────────────────────────────────────────────

def categorize_interaction(template: str) -> str:
    """Map a fine-grained interaction template to a coarse category.

    For example:
        "The serum concentration of #Drug1 can be increased..." -> "serum_increase"
        "The risk or severity of adverse effects..."            -> "adverse_effects"

    Returns "other" if no keyword matches.
    """
    # Normalise to lowercase and remove drug name placeholders so keywords match
    t = template.lower().replace("#drug1", "").replace("#drug2", "")

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            # Also normalise the keyword the same way
            kw = keyword.lower().replace("#drug1", "").replace("#drug2", "")
            if kw in t:
                return category

    return "other"


def drug_has_useful_profile(drug_id: str, profiles: dict) -> bool:
    """Return True if a drug has at least one pharmacological field populated.

    A drug with no description, no mechanism, no enzymes, no targets, and no
    transporters is essentially a black box -- the teacher model would have
    nothing to reason from and would hallucinate a mechanism. We exclude these.
    """
    profile = profiles.get(drug_id, {})
    if not profile:
        return False
    return any(profile.get(field) for field in USEFUL_PROFILE_FIELDS)


def assign_frequency_tiers(label_counts: Counter) -> dict:
    """Assign each class to a frequency tier: head, mid, or tail.

    Mirrors Mohammadreza's analysis (slide 13):
        Head: top 15 classes by pair count  (very common interactions)
        Mid:  classes 16-80 by pair count   (moderately common)
        Tail: classes 81+ by pair count     (rare interactions)

    Returns dict {label_id -> "head" | "mid" | "tail"}
    """
    # Sort labels by frequency, most common first
    sorted_labels = [label for label, count in label_counts.most_common()]

    tiers = {}
    for rank, label in enumerate(sorted_labels, start=1):
        if rank <= 15:
            tiers[label] = "head"
        elif rank <= 80:
            tiers[label] = "mid"
        else:
            tiers[label] = "tail"

    return tiers


def build_dataset(
    interactions: list,
    profiles: dict,
    raw_label_map: dict,
    min_pairs_per_class: int,
    output_dir: str,
    dataset_name: str,
) -> dict:
    """Filter, split, and save a dataset for one threshold level.

    Arguments:
        interactions      : all raw interaction pairs (from interactions_full.jsonl)
        profiles          : drug profiles dict (from drug_profiles.json)
        raw_label_map     : original label_id -> template mapping (354 classes)
        min_pairs_per_class: keep only classes with at least this many pairs
        output_dir        : where to save the output files
        dataset_name      : human-readable name for logging ("A (>=130)" etc.)

    Returns a summary dict with key statistics.
    """
    print(f"\n{'='*60}")
    print(f"Building Dataset {dataset_name}")
    print(f"Threshold: >= {min_pairs_per_class} pairs per class")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Filter by minimum class size ──────────────────────────────────
    # Count how many pairs each class has in the raw data
    raw_class_counts = Counter(ix["label"] for ix in interactions)

    # Keep only classes that meet the minimum threshold
    classes_to_keep = {
        label for label, count in raw_class_counts.items()
        if count >= min_pairs_per_class
    }

    # Filter the interactions list to only kept classes
    filtered = [ix for ix in interactions if ix["label"] in classes_to_keep]

    print(f"\nStep 1: Class size filter (>= {min_pairs_per_class} pairs)")
    print(f"  Classes before: {len(raw_class_counts):,}")
    print(f"  Classes kept:   {len(classes_to_keep):,}")
    print(f"  Classes dropped:{len(raw_class_counts) - len(classes_to_keep):,}")
    print(f"  Pairs kept:     {len(filtered):,}")

    # ── Step 2: Filter out pairs with no useful drug profile ──────────────────
    # Both drugs in a pair must have at least one useful pharmacological field.
    # This prevents bad teacher traces for drugs the model knows nothing about.
    pairs_before_quality = len(filtered)
    filtered = [
        ix for ix in filtered
        if drug_has_useful_profile(ix["drug1_id"], profiles)
        and drug_has_useful_profile(ix["drug2_id"], profiles)
    ]
    pairs_removed = pairs_before_quality - len(filtered)

    print(f"\nStep 2: Profile quality filter")
    print(f"  Pairs removed (drug has no profile data): {pairs_removed:,}")
    print(f"  Pairs remaining: {len(filtered):,}")

    # Re-check class sizes -- some classes may have dropped below threshold
    # after the quality filter removed some of their pairs
    post_quality_counts = Counter(ix["label"] for ix in filtered)
    classes_below_threshold = {
        label for label in classes_to_keep
        if post_quality_counts.get(label, 0) < min_pairs_per_class
    }
    if classes_below_threshold:
        classes_to_keep -= classes_below_threshold
        filtered = [ix for ix in filtered if ix["label"] in classes_to_keep]
        print(f"  Classes dropped after quality filter: {len(classes_below_threshold):,}")
        print(f"  Final classes: {len(classes_to_keep):,}, pairs: {len(filtered):,}")

    # ── Step 3: Remap labels to contiguous IDs 1..N ───────────────────────────
    # The raw labels are arbitrary integers from extract_dataset_from_xml.py.
    # We remap to 1..N ordered by frequency (most common class = label 1).
    # This makes the label space clean and consistent.
    final_class_counts = Counter(ix["label"] for ix in filtered)

    old_label_to_new = {}   # old arbitrary ID -> new clean ID
    new_label_map = {}      # new clean ID -> interaction template text

    for new_id, (old_id, template) in enumerate(
        # Sort by frequency descending so label 1 = most common class
        sorted(
            [(label, raw_label_map[label]) for label in classes_to_keep],
            key=lambda x: -final_class_counts[x[0]]
        ),
        start=1,
    ):
        old_label_to_new[old_id] = new_id
        new_label_map[new_id] = template

    # Apply the remapping to all pairs
    for ix in filtered:
        ix["label"] = old_label_to_new[ix["label"]]

    print(f"\nStep 3: Label remapping")
    print(f"  Labels remapped to 1..{len(new_label_map)}")
    print(f"  (Label 1 = most common class, label {len(new_label_map)} = rarest)")

    # ── Step 4: Assign coarse categories and frequency tiers ──────────────────
    # Coarse categories group fine-grained classes by mechanism type
    # (e.g. all "serum concentration increase" classes -> "serum_increase")
    coarse_map = {
        label_id: categorize_interaction(template)
        for label_id, template in new_label_map.items()
    }

    # Frequency tiers for the head/mid/tail analysis
    new_class_counts = Counter(ix["label"] for ix in filtered)
    tier_map = assign_frequency_tiers(new_class_counts)

    # Add label_text (template with real drug names), coarse_category,
    # and frequency_tier to every pair
    for ix in filtered:
        template = new_label_map[ix["label"]]
        ix["label_text"] = (
            template
            .replace("#Drug1", ix["drug1_name"])
            .replace("#Drug2", ix["drug2_name"])
        )
        ix["coarse_category"] = coarse_map[ix["label"]]
        ix["frequency_tier"] = tier_map[ix["label"]]
        # Severity is unknown since we have no DDInter files
        ix.setdefault("severity", "Unknown")

    # Log coarse category distribution
    coarse_counts = Counter(coarse_map.values())
    print(f"\nStep 4: Coarse categories")
    for cat, count in coarse_counts.most_common():
        print(f"  {cat}: {count} fine-grained classes")

    # Log tier distribution
    tier_counts = Counter(tier_map.values())
    print(f"\n  Frequency tiers:")
    for tier in ("head", "mid", "tail"):
        n_classes = tier_counts.get(tier, 0)
        n_pairs = sum(
            new_class_counts[label]
            for label, t in tier_map.items() if t == tier
        )
        print(f"  {tier:4s}: {n_classes:3d} classes, {n_pairs:,} pairs")

    # ── Step 5: Stratified 80/20 train/test split ─────────────────────────────
    # "Stratified" means each class appears in both train and test at the same
    # proportion as in the full dataset. This prevents any class from being
    # entirely absent from either split.
    df = pd.DataFrame(filtered)
    train_df, test_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        stratify=df["label"],   # ensure each class is proportionally split
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\nStep 5: Train/test split ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})")
    print(f"  Train: {len(train_df):,} pairs")
    print(f"  Test:  {len(test_df):,} pairs")

    # ── Step 6: Per-class training cap ────────────────────────────────────────
    # The largest class has up to 180K pairs -- without a cap, it would
    # dominate training and macro F1 would be misleading.
    # Cap each class at MAX_TRAIN_PER_CLASS examples in the training set.
    rng = np.random.RandomState(SEED)
    capped_parts = []
    n_capped_classes = 0

    for label in sorted(train_df["label"].unique()):
        class_rows = train_df[train_df["label"] == label]
        if len(class_rows) > MAX_TRAIN_PER_CLASS:
            class_rows = class_rows.sample(n=MAX_TRAIN_PER_CLASS, random_state=rng)
            n_capped_classes += 1
        capped_parts.append(class_rows)

    train_df = pd.concat(capped_parts, ignore_index=True)
    # Shuffle so classes aren't in order
    train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    print(f"\nStep 6: Per-class training cap ({MAX_TRAIN_PER_CLASS:,} max per class)")
    print(f"  Classes that were capped: {n_capped_classes}")
    print(f"  Training pairs after cap: {len(train_df):,}")

    # Log final class size distribution
    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()
    print(f"\n  Training class sizes:")
    print(f"    Min: {train_counts.min()}, Max: {train_counts.max()}, "
          f"Median: {train_counts.median():.0f}")
    print(f"  Test class sizes:")
    print(f"    Min: {test_counts.min()}, Max: {test_counts.max()}, "
          f"Median: {test_counts.median():.0f}")

    # ── Step 7: Save all outputs ───────────────────────────────────────────────
    out = Path(output_dir)

    # Main data files
    train_df.to_json(out / "train.jsonl", orient="records", lines=True)
    test_df.to_json(out / "test.jsonl", orient="records", lines=True)

    # Label maps
    with open(out / "label_map.json", "w") as f:
        json.dump(new_label_map, f, indent=2)

    with open(out / "coarse_category_map.json", "w") as f:
        json.dump(coarse_map, f, indent=2)

    # Tier map (for experiment analysis)
    with open(out / "tier_map.json", "w") as f:
        json.dump(tier_map, f, indent=2)

    # Summary statistics for quick reference
    summary = {
        "dataset_name": dataset_name,
        "min_pairs_per_class": min_pairs_per_class,
        "n_classes": len(new_label_map),
        "n_train_pairs": len(train_df),
        "n_test_pairs": len(test_df),
        "n_pairs_total": len(train_df) + len(test_df),
        "n_drugs": len(
            set(train_df["drug1_id"]) | set(train_df["drug2_id"]) |
            set(test_df["drug1_id"]) | set(test_df["drug2_id"])
        ),
        "class_size_min_train": int(train_counts.min()),
        "class_size_max_train": int(train_counts.max()),
        "class_size_median_train": float(train_counts.median()),
        "n_head_classes": tier_counts.get("head", 0),
        "n_mid_classes": tier_counts.get("mid", 0),
        "n_tail_classes": tier_counts.get("tail", 0),
        "coarse_categories": dict(coarse_counts),
    }
    with open(out / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStep 7: Saved outputs to {output_dir}/")
    print(f"  train.jsonl           ({len(train_df):,} pairs)")
    print(f"  test.jsonl            ({len(test_df):,} pairs)")
    print(f"  label_map.json        ({len(new_label_map)} classes)")
    print(f"  coarse_category_map.json")
    print(f"  tier_map.json")
    print(f"  dataset_summary.json")

    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment Dataset Preparation")
    print("Building Dataset A (strict) and Dataset B (relaxed)")
    print("=" * 60)

    raw_dir = Path(RAW_DATA_DIR)

    # ── Load raw extracted data ────────────────────────────────────────────────
    print("\nLoading raw extracted data...")

    interactions_path = raw_dir / "interactions_full.jsonl"
    if not interactions_path.exists():
        print(f"ERROR: {interactions_path} not found.")
        print("Run: python scripts/extract_dataset_from_xml.py first.")
        return

    interactions = []
    with open(interactions_path) as f:
        for line in f:
            interactions.append(json.loads(line))
    print(f"  Loaded {len(interactions):,} raw interaction pairs")

    profiles_path = raw_dir / "drug_profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)
    print(f"  Loaded {len(profiles):,} drug profiles")

    # Load the raw label map (354 classes before filtering)
    # Try raw_label_map.json first (backup), fall back to label_map.json
    raw_lm_path = raw_dir / "raw_label_map.json"
    label_map_path = raw_dir / "label_map.json"

    if raw_lm_path.exists():
        with open(raw_lm_path) as f:
            raw_label_map = {int(k): v for k, v in json.load(f).items()}
    else:
        with open(label_map_path) as f:
            raw_label_map = {int(k): v for k, v in json.load(f).items()}
        # Save backup so data_preparation.py doesn't overwrite it
        with open(raw_lm_path, "w") as f:
            json.dump({str(k): v for k, v in raw_label_map.items()}, f, indent=2)
    print(f"  Loaded raw label map ({len(raw_label_map)} classes)")

    # ── Build Dataset A (strict, >= 130 pairs/class) ───────────────────────────
    summary_A = build_dataset(
        interactions=interactions,
        profiles=profiles,
        raw_label_map=raw_label_map,
        min_pairs_per_class=DATASET_A_MIN_PAIRS,
        output_dir=DATASET_A_DIR,
        dataset_name=f"A (>= {DATASET_A_MIN_PAIRS} pairs/class, Mohammadreza's threshold)",
    )

    # ── Build Dataset B (relaxed, >= 20 pairs/class) ───────────────────────────
    summary_B = build_dataset(
        interactions=interactions,
        profiles=profiles,
        raw_label_map=raw_label_map,
        min_pairs_per_class=DATASET_B_MIN_PAIRS,
        output_dir=DATASET_B_DIR,
        dataset_name=f"B (>= {DATASET_B_MIN_PAIRS} pairs/class, relaxed threshold)",
    )

    # ── Final comparison summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DATASET COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Dataset A':>12} {'Dataset B':>12}")
    print(f"{'-'*54}")
    metrics = [
        ("Min pairs/class threshold", "min_pairs_per_class"),
        ("Number of classes",         "n_classes"),
        ("Training pairs",            "n_train_pairs"),
        ("Test pairs",                "n_test_pairs"),
        ("Unique drugs",              "n_drugs"),
        ("Head classes",              "n_head_classes"),
        ("Mid classes",               "n_mid_classes"),
        ("Tail classes",              "n_tail_classes"),
    ]
    for label, key in metrics:
        val_a = summary_A[key]
        val_b = summary_B[key]
        if isinstance(val_a, int) and val_a > 999:
            print(f"  {label:<28} {val_a:>12,} {val_b:>12,}")
        else:
            print(f"  {label:<28} {val_a:>12} {val_b:>12}")

    print(f"\nDataset A classes not in B: 0 (A is a strict subset of B)")
    extra_in_b = summary_B["n_classes"] - summary_A["n_classes"]
    print(f"Extra classes in B only:    {extra_in_b} (the rare tail classes)")
    print(f"\nThese {extra_in_b} rare classes are what our experiment tests.")
    print(f"If pathway retrieval helps them, we can justify including them")
    print(f"in teacher generation -- expanding PharmCoT's taxonomy.")

    print(f"\nDone. Both datasets ready for the retrieval comparison experiment.")
    print(f"Next step: python scripts/build_fingerprints.py")


if __name__ == "__main__":
    main()
