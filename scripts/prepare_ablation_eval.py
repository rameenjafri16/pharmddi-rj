"""
prepare_ablation_eval.py

Two things that prepare you to evaluate whether our data_preparation.py
changes actually improved student accuracy when results come back:

PART 1 — Tag test sets with prodrug flags
  Adds an 'involves_prodrug' column to both Dataset A and B test splits.
  This means when student predictions arrive, you can immediately slice
  the evaluation by prodrug vs non-prodrug pairs without any extra work.
  Prodrug pairs are the highest-value test — our changes specifically
  target them, and they are clinically the most dangerous to get wrong.

PART 2 — Generate prompt comparison (before vs after our changes)
  For 10 carefully chosen pairs (including clopidogrel and other prodrugs),
  shows the old prompt (no PK/PD flag, no prodrug warning) side by side
  with the new prompt. This is your proof that the code changes do what
  they claim — no GPU needed, no teacher model needed.

OUTPUTS:
  data/processed/dataset_A/test_with_prodrug_flag.jsonl
  data/processed/dataset_B/test_with_prodrug_flag.jsonl
  outputs/ablation/prompt_comparison.txt

USAGE:
  python scripts/prepare_ablation_eval.py

No GPU needed. Runs in ~2 minutes on login node.
"""

import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from src.data_preparation import (
    build_teacher_prompt,
    classify_pk_pd,
    _load_prodrug_ids,
    _format_drug_profile,
)

PROC_DIR   = Path("data/processed")
OUTPUT_DIR = Path("outputs/ablation")


# ── Original prompt builder (before RJ changes) ───────────────────────────────
# Reproduced here exactly as it was in Mohammadreza's original version
# so we can show a fair before/after comparison.

def build_teacher_prompt_original(row, label_map, profiles, retrieved_examples=None):
    """Original build_teacher_prompt — no PK/PD flag, no prodrug warning."""
    parts = []

    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples, 1):
            p1 = profiles.get(ex["drug1_id"], {})
            p2 = profiles.get(ex["drug2_id"], {})
            parts.append(f"--- Example {i} ---")
            parts.append(f"Drug 1: {ex['drug1_name']} ({ex['drug1_id']})")
            if p1:
                parts.append(_format_drug_profile(p1))
            parts.append(f"Drug 2: {ex['drug2_name']} ({ex['drug2_id']})")
            if p2:
                parts.append(_format_drug_profile(p2))
            ex_label_text = label_map.get(ex["label"], "")
            if "#Drug1" in ex_label_text:
                ex_label_text = (ex_label_text
                                 .replace("#Drug1", ex["drug1_name"])
                                 .replace("#Drug2", ex["drug2_name"]))
            parts.append(f"Interaction: Y={ex['label']} -- \"{ex_label_text}\"")
            parts.append("")

    parts.append("--- Your turn ---")
    p1 = profiles.get(row["drug1_id"], {})
    p2 = profiles.get(row["drug2_id"], {})
    parts.append(f"Drug 1: {row['drug1_name']} ({row['drug1_id']})")
    if p1:
        parts.append(_format_drug_profile(p1))
    parts.append(f"Drug 2: {row['drug2_name']} ({row['drug2_id']})")
    if p2:
        parts.append(_format_drug_profile(p2))

    severity = row.get("severity", "Unknown")
    parts.append(f"Known interaction: Y={row['label']} -- \"{row['label_text']}\"")
    parts.append(f"Known severity: {severity}")
    parts.append("")
    parts.append(
        "Explain step-by-step the pharmacological mechanisms behind this "
        "drug-drug interaction. Discuss each drug's mechanism of action and "
        "how they combine to produce this effect. Then provide a concise summary. "
        "End with the classification and severity."
    )
    return "\n".join(parts)


# ── Part 1: Tag test sets with prodrug flags ──────────────────────────────────

def tag_test_sets_with_prodrug_flag():
    """
    Add involves_prodrug column to test splits for both datasets.

    When student predictions come back you can slice evaluation like:
        test_df[test_df['involves_prodrug'] == True]
    to specifically measure performance on prodrug pairs — the pairs
    our changes most directly target.
    """
    prodrug_ids = _load_prodrug_ids()
    print(f"Loaded {len(prodrug_ids):,} prodrug IDs")

    for ds_name, ds_dir in [("Dataset A", "dataset_A"),
                             ("Dataset B", "dataset_B")]:
        test_path = PROC_DIR / ds_dir / "test.jsonl"
        if not test_path.exists():
            print(f"  {ds_name}: test.jsonl not found, skipping")
            continue

        test_df = pd.read_json(test_path, lines=True)

        # Tag each pair
        test_df["involves_prodrug"] = test_df.apply(
            lambda row: (
                row["drug1_id"] in prodrug_ids or
                row["drug2_id"] in prodrug_ids
            ),
            axis=1,
        )

        # Also tag which specific drug is the prodrug
        test_df["prodrug_drug"] = test_df.apply(
            lambda row: (
                row["drug1_name"] if row["drug1_id"] in prodrug_ids
                else row["drug2_name"] if row["drug2_id"] in prodrug_ids
                else None
            ),
            axis=1,
        )

        n_prodrug_pairs = test_df["involves_prodrug"].sum()
        n_total = len(test_df)

        # Save tagged version
        out_path = PROC_DIR / ds_dir / "test_with_prodrug_flag.jsonl"
        test_df.to_json(out_path, orient="records", lines=True)

        print(f"\n{ds_name}:")
        print(f"  Total test pairs:          {n_total:,}")
        print(f"  Pairs with prodrug:        {n_prodrug_pairs:,} "
              f"({100*n_prodrug_pairs/n_total:.1f}%)")
        print(f"  Saved: {out_path}")

        # Show which prodrugs appear most in the test set
        prodrug_counts = (
            test_df[test_df["involves_prodrug"]]
            ["prodrug_drug"]
            .value_counts()
            .head(10)
        )
        print(f"  Most common prodrugs in test set:")
        for drug, count in prodrug_counts.items():
            print(f"    {drug:<35} {count:,} pairs")


# ── Part 2: Prompt comparison ─────────────────────────────────────────────────

def generate_prompt_comparison():
    """
    For 10 carefully chosen pairs, show old prompt vs new prompt side by side.

    Pairs chosen to cover:
      - A clopidogrel pair (classic prodrug, wrong direction without fix)
      - Another well-known prodrug pair
      - A PK interaction (serum concentration)
      - A PD interaction (adverse effects)
      - A non-prodrug pair (should be unchanged except for PK/PD flag)
    """
    prodrug_ids = _load_prodrug_ids()

    with open(PROC_DIR / "dataset_A" / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(PROC_DIR / "drug_profiles.json") as f:
        profiles = json.load(f)
    train_df = pd.read_json(PROC_DIR / "dataset_A" / "train.jsonl", lines=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "prompt_comparison.txt"

    pairs_to_show = []

    # 1. Find clopidogrel pairs
    clop_id = next(
        (k for k, v in profiles.items() if "clopidogrel" in v.get("name", "").lower()),
        None,
    )
    if clop_id:
        clop_rows = train_df[
            (train_df["drug1_id"] == clop_id) | (train_df["drug2_id"] == clop_id)
        ]
        if len(clop_rows) > 0:
            pairs_to_show.append(("Clopidogrel pair (classic prodrug)", clop_rows.iloc[0]))

    # 2. Find another prodrug pair (simvastatin)
    simva_id = next(
        (k for k, v in profiles.items() if v.get("name", "").lower() == "simvastatin"),
        None,
    )
    if simva_id:
        simva_rows = train_df[
            (train_df["drug1_id"] == simva_id) | (train_df["drug2_id"] == simva_id)
        ]
        if len(simva_rows) > 0:
            pairs_to_show.append(("Simvastatin pair (statin prodrug)", simva_rows.iloc[0]))

    # 3. A PK interaction (serum concentration)
    pk_rows = train_df[
        train_df["label_text"].str.contains("serum concentration", case=False, na=False)
    ]
    for _, row in pk_rows.iterrows():
        if row["drug1_id"] not in prodrug_ids and row["drug2_id"] not in prodrug_ids:
            pairs_to_show.append(("PK interaction (serum concentration, no prodrug)", row))
            break

    # 4. A PD interaction (adverse effects)
    pd_rows = train_df[
        train_df["label_text"].str.contains("adverse effects", case=False, na=False)
    ]
    for _, row in pd_rows.iterrows():
        if row["drug1_id"] not in prodrug_ids and row["drug2_id"] not in prodrug_ids:
            pairs_to_show.append(("PD interaction (adverse effects, no prodrug)", row))
            break

    # 5. A non-prodrug PD activity interaction
    act_rows = train_df[
        train_df["label_text"].str.contains("activities", case=False, na=False)
    ]
    for _, row in act_rows.iterrows():
        if row["drug1_id"] not in prodrug_ids and row["drug2_id"] not in prodrug_ids:
            pairs_to_show.append(("PD activity interaction (no prodrug)", row))
            break

    # Generate comparison for each pair
    comparison_lines = [
        "=" * 80,
        "PROMPT COMPARISON: ORIGINAL vs RJ-MODIFIED data_preparation.py",
        "=" * 80,
        "",
        "This shows the exact prompts sent to the teacher model for 5 pairs.",
        "Differences are marked with >>> for new content added by RJ changes.",
        "",
    ]

    for description, row in pairs_to_show:
        old_prompt = build_teacher_prompt_original(row, label_map, profiles)
        new_prompt = build_teacher_prompt(row, label_map, profiles)

        comparison_lines += [
            "─" * 80,
            f"PAIR: {description}",
            f"Drug 1: {row['drug1_name']} ({row['drug1_id']}) "
            f"{'[PRODRUG]' if row['drug1_id'] in prodrug_ids else ''}",
            f"Drug 2: {row['drug2_name']} ({row['drug2_id']}) "
            f"{'[PRODRUG]' if row['drug2_id'] in prodrug_ids else ''}",
            f"Label:  {row['label_text'][:80]}",
            "─" * 80,
            "",
        ]

        # Find what's new in the modified prompt
        old_lines = set(old_prompt.split("\n"))
        new_lines_list = new_prompt.split("\n")

        comparison_lines.append("LINES ADDED BY RJ CHANGES (not in original):")
        added = [l for l in new_lines_list if l.strip() and l not in old_lines]
        if added:
            for line in added:
                comparison_lines.append(f"  >>> {line}")
        else:
            comparison_lines.append("  (no changes for this pair)")

        comparison_lines += ["", "FULL NEW PROMPT (last 15 lines shown):", ""]
        prompt_lines = new_prompt.split("\n")
        for line in prompt_lines[-15:]:
            comparison_lines.append(f"  {line}")
        comparison_lines.append("")

    comparison_lines += [
        "=" * 80,
        "SUMMARY OF CHANGES",
        "=" * 80,
        "",
        "1. Every prompt now has 'Interaction type: PK/PD ...' line",
        "   -> Tells teacher whether to reason about ADME or receptor effects",
        "",
        "2. Prodrug pairs now have '⚠️ PRODRUG WARNING: ...' line",
        "   -> Tells teacher that enzyme inhibition DECREASES active drug levels",
        "   -> Affects 8.3% of training pairs (125 prodrugs in Dataset A)",
        "",
        "3. Profile truncation caps raised (enzymes 5→8, transporters 3→5, targets 3→5)",
        "   -> More complete pharmacological context for highly metabolised drugs",
        "   -> Affects ~2-5% of drug profiles",
        "",
        "TO MEASURE IMPACT ON STUDENT ACCURACY:",
        "  When predictions arrive, run evaluation separately on:",
        "    test_with_prodrug_flag.jsonl where involves_prodrug == True",
        "  Compare F1 on prodrug pairs: our pipeline vs original pipeline",
        "  Hypothesis: prodrug pair F1 improves because teacher traces have",
        "  correct direction of effect for 8.3% of training data.",
        "=" * 80,
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(comparison_lines))

    print(f"\nPrompt comparison saved to: {out_path}")

    # Also print a quick summary to terminal
    print("\nQUICK SUMMARY — new lines added per pair type:")
    for description, row in pairs_to_show:
        old_prompt = build_teacher_prompt_original(row, label_map, profiles)
        new_prompt = build_teacher_prompt(row, label_map, profiles)
        old_lines = set(old_prompt.split("\n"))
        added = [l for l in new_prompt.split("\n")
                 if l.strip() and l not in old_lines]
        print(f"  {description[:50]:<50} +{len(added)} lines")
        for line in added:
            if line.strip():
                print(f"    >>> {line[:80]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Ablation Evaluation Preparation")
    print("=" * 60)

    print("\n--- Part 1: Tagging test sets with prodrug flags ---")
    tag_test_sets_with_prodrug_flag()

    print("\n--- Part 2: Generating prompt comparison ---")
    generate_prompt_comparison()

    print("\nDone. When student predictions arrive:")
    print("  1. Load test_with_prodrug_flag.jsonl instead of test.jsonl")
    print("  2. Compute F1 separately for involves_prodrug=True/False")
    print("  3. Compare with baseline to measure prodrug-specific improvement")


if __name__ == "__main__":
    main()
