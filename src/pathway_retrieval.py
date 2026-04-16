"""
Pathway-aware RAG retrieval for DDI teacher generation.

Replaces Tanimoto-based structural retrieval with mechanism-based retrieval:
instead of finding drug pairs that look structurally similar, we find pairs
that share the same biological pathway relationship (same CYP enzyme, same
transporter, same pharmacodynamic target, same action type).

The core insight: DDIs are caused by shared biological bottlenecks, not
structural similarity. CYP3A4 inhibition by fluconazole causes the same
interaction pattern regardless of what fluconazole looks like. Two drugs
with identical Morgan fingerprints can interact via completely different
mechanisms.

Output format: identical to retrieved_examples_train.json produced by
precompute_retrievals() in data_preparation.py. This means it can be
used as a drop-in replacement with zero changes to teacher_generation.py.

Usage (run on login node before teacher generation):
    python scripts/build_pathway_retrievals.py \
        --split train \
        --out data/processed/retrieved_examples_train_pathway.json

To use in teacher generation, either:
    1. Rename the output to retrieved_examples_train.json, OR
    2. Pass --retrieval-file flag to teacher_generation.py (if you add that arg)
"""

import json
import re
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Enzyme / transporter / target parsing
# ---------------------------------------------------------------------------
# DrugBank profile entries look like:
#   "CYP3A4 (CYP3A4): substrate, inhibitor"
#   "P-glycoprotein (ABCB1): substrate"
#   "Vitamin K epoxide reductase (VKORC1): inhibitor"

_ENTRY_RE = re.compile(
    r"^(?P<name>[^(]+?)"                # everything before the first (
    r"(?:\s*\((?P<gene>[^)]+)\))?"      # optional (GENE_NAME)
    r"(?:\s*:\s*(?P<actions>.+))?$"     # optional : action1, action2
)

# Normalise action strings to canonical forms
_ACTION_NORM = {
    "substrate":       "substrate",
    "inhibitor":       "inhibitor",
    "inducer":         "inducer",
    "activator":       "activator",
    "binder":          "binder",
    "cofactor":        "cofactor",
    "transporter":     "transporter",
    "agonist":         "agonist",
    "antagonist":      "antagonist",
    "partial agonist": "partial_agonist",
}

# Map inhibitor/inducer/substrate to a short role string used for matching
_ROLE_PRIORITY = {
    "inhibitor": "inhibitor",
    "inducer":   "inducer",
    "substrate": "substrate",
    "activator": "activator",
    "agonist":   "agonist",
    "antagonist":"antagonist",
}


def _parse_entry(raw: str) -> dict:
    """Parse a single enzyme/transporter/target string into components.

    Always returns a dict with keys: name, gene, canonical, actions.
    Returns canonical='' for empty or completely unparseable entries so
    callers can safely skip them with: if not parsed['canonical']: continue
    """
    raw = raw.strip()

    # Empty string — return safe empty dict, caller will skip
    if not raw:
        return {"name": "", "gene": "", "canonical": "", "actions": []}

    m = _ENTRY_RE.match(raw)

    # Regex failed — use the raw string itself as the canonical name
    # rather than crashing. This handles unusual DrugBank formatting.
    if not m:
        return {"name": raw, "gene": "", "canonical": raw.upper(), "actions": []}

    name = m.group("name").strip()
    gene = (m.group("gene") or "").strip()
    actions_raw = (m.group("actions") or "").strip()

    actions = []
    if actions_raw:
        for a in re.split(r"[,;]", actions_raw):
            a = a.strip().lower()
            normed = _ACTION_NORM.get(a, a)
            if normed:
                actions.append(normed)

    # Use gene name as canonical identifier when available (more precise)
    canonical = gene if gene else name
    return {"name": name, "gene": gene, "canonical": canonical.upper(), "actions": actions}


def _get_role(actions: list) -> str:
    """Return the most specific pharmacological role from an action list."""
    for role in ("inhibitor", "inducer", "activator", "agonist", "antagonist", "substrate"):
        if role in actions:
            return role
    return actions[0] if actions else "unknown"


def _extract_pathway_nodes(profile: dict) -> dict:
    """Extract pathway nodes from a drug profile.

    Returns a dict with three keys:
      enzymes    : {canonical_name -> role}
      transporters: {canonical_name -> role}
      targets    : {canonical_name -> role}
    """
    result = {"enzymes": {}, "transporters": {}, "targets": {}}

    for field in ("enzymes", "transporters", "targets"):
        for raw in profile.get(field, []):
            parsed = _parse_entry(raw)
            # .get() with default handles any edge case where canonical
            # key is missing — should never happen now but belt-and-suspenders
            canonical = parsed.get("canonical", "")
            if not canonical:
                continue
            role = _get_role(parsed["actions"])
            result[field][canonical] = role

    return result


# ---------------------------------------------------------------------------
# Interaction type characterisation
# ---------------------------------------------------------------------------

# When a drug pair interacts via a shared enzyme, the typical pattern is:
#   drug_a: substrate of enzyme X
#   drug_b: inhibitor or inducer of enzyme X
# This function identifies the "interaction signature" for a pair.

def _pair_signature(nodes_a: dict, nodes_b: dict) -> list[dict]:
    """Return all pathway overlaps between drug A and drug B.

    Each overlap is a dict:
      field    : "enzymes" | "transporters" | "targets"
      entity   : canonical entity name (e.g. "CYP3A4")
      role_a   : role of drug A (e.g. "substrate")
      role_b   : role of drug B (e.g. "inhibitor")
      weight   : importance weight for scoring
    """
    overlaps = []

    # Field weights: enzymes most important for DDI, then transporters, then targets
    field_weights = {"enzymes": 3.0, "transporters": 2.0, "targets": 1.0}

    for field in ("enzymes", "transporters", "targets"):
        a_entities = nodes_a[field]
        b_entities = nodes_b[field]
        shared = set(a_entities.keys()) & set(b_entities.keys())

        for entity in shared:
            role_a = a_entities[entity]
            role_b = b_entities[entity]
            w = field_weights[field]

            # Boost weight for pharmacokinetic interactions
            # (substrate+inhibitor or substrate+inducer is the classic DDI pattern)
            if {role_a, role_b} == {"substrate", "inhibitor"}:
                w *= 2.0
            elif {role_a, role_b} == {"substrate", "inducer"}:
                w *= 1.8
            elif role_a == role_b == "substrate":
                w *= 1.2  # competition for same enzyme

            overlaps.append({
                "field": field,
                "entity": entity,
                "role_a": role_a,
                "role_b": role_b,
                "weight": w,
            })

    return overlaps


def _signature_score(overlaps: list[dict]) -> float:
    """Compute a scalar similarity score from pathway overlaps."""
    if not overlaps:
        return 0.0
    return sum(o["weight"] for o in overlaps)


# ---------------------------------------------------------------------------
# Building the pathway index
# ---------------------------------------------------------------------------

def build_pathway_index(profiles: dict) -> dict:
    """Pre-extract pathway nodes for every drug in the profiles.

    Returns {drug_id -> {enzymes: {entity->role}, transporters: ..., targets: ...}}
    """
    index = {}
    for drug_id, profile in profiles.items():
        index[drug_id] = _extract_pathway_nodes(profile)
    return index


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def compute_pathway_retrievals(
    split_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    profiles: dict,
    top_k: int = 5,
    min_diverse_classes: int = 2,
    seed: int = 42,
    batch_size: int = 1000,
) -> dict:
    """Compute pathway-aware retrieved examples for every row in split_df.

    For each query pair (drug1, drug2), scores every candidate pair by
    pathway overlap and returns the top_k candidates with at least
    min_diverse_classes distinct interaction classes represented.

    Arguments:
        split_df           : DataFrame of pairs to retrieve for (query set)
        candidate_df       : DataFrame of pairs to retrieve from (candidate pool)
        profiles           : drug_profiles dict {drug_id -> profile}
        top_k              : number of examples to retrieve per query
        min_diverse_classes: minimum number of distinct classes in retrieved set
        seed               : random seed for tie-breaking
        batch_size         : pairs to process per batch (memory control)

    Returns:
        dict {original_df_index -> [list of candidate_df original indices]}
        Same format as precompute_retrievals() in data_preparation.py.
    """
    rng = np.random.RandomState(seed)

    print(f"Building pathway index for {len(profiles):,} drugs...")
    pathway_index = build_pathway_index(profiles)

    # Build arrays of candidate data
    cand_indices = np.array(candidate_df.index.tolist())
    cand_labels = np.array(candidate_df["label"].tolist())
    cand_d1_ids = candidate_df["drug1_id"].tolist()
    cand_d2_ids = candidate_df["drug2_id"].tolist()

    n_cand = len(candidate_df)
    n_query = len(split_df)

    # Pre-extract pathway nodes for all candidate drugs
    print(f"Pre-extracting pathway nodes for {n_cand:,} candidate pairs...")
    cand_nodes_d1 = [pathway_index.get(did, {"enzymes": {}, "transporters": {}, "targets": {}})
                     for did in cand_d1_ids]
    cand_nodes_d2 = [pathway_index.get(did, {"enzymes": {}, "transporters": {}, "targets": {}})
                     for did in cand_d2_ids]

    # Track coverage stats
    n_with_pathway = 0
    n_no_pathway = 0
    total_retrieved = 0

    retrievals = {}

    print(f"Computing pathway retrievals for {n_query:,} query pairs...")
    for batch_start in range(0, n_query, batch_size):
        batch_end = min(batch_start + batch_size, n_query)
        batch = split_df.iloc[batch_start:batch_end]

        for local_i, (orig_idx, row) in enumerate(batch.iterrows()):
            q_d1 = row["drug1_id"]
            q_d2 = row["drug2_id"]
            q_nodes_d1 = pathway_index.get(q_d1, {"enzymes": {}, "transporters": {}, "targets": {}})
            q_nodes_d2 = pathway_index.get(q_d2, {"enzymes": {}, "transporters": {}, "targets": {}})

            # Score every candidate pair
            scores = np.zeros(n_cand, dtype=np.float32)
            for ci in range(n_cand):
                # Skip self
                if cand_indices[ci] == orig_idx:
                    continue

                c_nodes_d1 = cand_nodes_d1[ci]
                c_nodes_d2 = cand_nodes_d2[ci]

                # Score both orientations (query_d1 vs cand_d1, and query_d1 vs cand_d2)
                overlaps_fwd = _pair_signature(q_nodes_d1, c_nodes_d1)
                overlaps_fwd += _pair_signature(q_nodes_d2, c_nodes_d2)
                score_fwd = _signature_score(overlaps_fwd)

                overlaps_rev = _pair_signature(q_nodes_d1, c_nodes_d2)
                overlaps_rev += _pair_signature(q_nodes_d2, c_nodes_d1)
                score_rev = _signature_score(overlaps_rev)

                scores[ci] = max(score_fwd, score_rev)

            has_any_pathway = scores.max() > 0
            if has_any_pathway:
                n_with_pathway += 1
            else:
                n_no_pathway += 1

            # Get top candidates with diversity constraint
            top_n = min(top_k * 20, n_cand)
            top_positions = np.argpartition(scores, -top_n)[-top_n:]
            top_positions = top_positions[np.argsort(-scores[top_positions])]

            selected = []
            classes_seen = set()

            for cp in top_positions:
                if len(selected) >= top_k:
                    break
                if scores[cp] <= 0:
                    break  # no more pathway overlap
                lbl = cand_labels[cp]
                # Enforce diversity: allow adding same class only if we already
                # have enough diverse ones
                if (len(classes_seen) < min_diverse_classes
                        or lbl not in classes_seen
                        or len(selected) >= top_k - 1):
                    selected.append(int(cand_indices[cp]))
                    classes_seen.add(lbl)

            # Fallback: if we couldn't fill top_k with pathway matches,
            # add highest-scoring remaining candidates regardless of diversity
            if len(selected) < top_k:
                for cp in top_positions:
                    if int(cand_indices[cp]) not in selected:
                        selected.append(int(cand_indices[cp]))
                        if len(selected) >= top_k:
                            break

            retrievals[int(orig_idx)] = selected
            total_retrieved += len(selected)

        completed = batch_end
        if completed % 10000 < batch_size or completed == n_query:
            pct = 100 * completed / n_query
            print(f"  Pathway retrieval: {completed:,}/{n_query:,} ({pct:.1f}%)  "
                  f"| with_pathway={n_with_pathway:,}  no_pathway={n_no_pathway:,}")

    print(f"\nPathway retrieval complete:")
    print(f"  Pairs with pathway overlap: {n_with_pathway:,}/{n_query:,} "
          f"({100*n_with_pathway/n_query:.1f}%)")
    print(f"  Pairs with no overlap (zero score): {n_no_pathway:,}")
    print(f"  Avg retrieved per pair: {total_retrieved/n_query:.2f}")
    print(f"  Note: pairs with no pathway data fall back to top Tanimoto "
          f"candidates if you combine retrievers.")

    return retrievals


# ---------------------------------------------------------------------------
# Coverage diagnostic
# ---------------------------------------------------------------------------

def pathway_coverage_report(profiles: dict, split_df: pd.DataFrame) -> dict:
    """Report what fraction of drug pairs have usable pathway data.

    A drug is 'pathway-rich' if it has at least one enzyme, transporter,
    or target annotation. A pair is 'pathway-matchable' if both drugs
    are pathway-rich.
    """
    pathway_index = build_pathway_index(profiles)

    def is_rich(drug_id):
        nodes = pathway_index.get(drug_id, {})
        return any(nodes.get(f) for f in ("enzymes", "transporters", "targets"))

    all_ids = set(split_df["drug1_id"]) | set(split_df["drug2_id"])
    rich_ids = {did for did in all_ids if is_rich(did)}

    pair_both_rich = sum(
        1 for _, row in split_df.iterrows()
        if row["drug1_id"] in rich_ids and row["drug2_id"] in rich_ids
    )
    pair_one_rich = sum(
        1 for _, row in split_df.iterrows()
        if (row["drug1_id"] in rich_ids) != (row["drug2_id"] in rich_ids)
    )
    pair_neither = len(split_df) - pair_both_rich - pair_one_rich

    # Enzyme coverage specifically (most important for DDI)
    def has_enzymes(drug_id):
        nodes = pathway_index.get(drug_id, {})
        return bool(nodes.get("enzymes"))

    pair_both_enzymes = sum(
        1 for _, row in split_df.iterrows()
        if has_enzymes(row["drug1_id"]) and has_enzymes(row["drug2_id"])
    )

    return {
        "total_unique_drugs": len(all_ids),
        "drugs_with_pathway_data": len(rich_ids),
        "drug_coverage_pct": round(100 * len(rich_ids) / len(all_ids), 1),
        "total_pairs": len(split_df),
        "pairs_both_drugs_rich": pair_both_rich,
        "pairs_one_drug_rich": pair_one_rich,
        "pairs_neither_rich": pair_neither,
        "full_pair_coverage_pct": round(100 * pair_both_rich / len(split_df), 1),
        "pairs_both_have_enzymes": pair_both_enzymes,
        "enzyme_pair_coverage_pct": round(100 * pair_both_enzymes / len(split_df), 1),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build pathway-aware retrieved examples for teacher generation"
    )
    parser.add_argument(
        "--split", choices=["train", "test"], default="train",
        help="Which split to compute retrievals for"
    )
    parser.add_argument(
        "--profiles", default="data/processed/drug_profiles.json",
        help="Path to drug_profiles.json"
    )
    parser.add_argument(
        "--data-dir", default="data/processed",
        help="Directory containing train.jsonl and test.jsonl"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: data/processed/retrieved_examples_{split}_pathway.json)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of examples to retrieve per pair"
    )
    parser.add_argument(
        "--min-diverse", type=int, default=2,
        help="Minimum distinct interaction classes in retrieved set"
    )
    parser.add_argument(
        "--pilot", type=int, default=0,
        help="Run on first N pairs only (0 = full)"
    )
    parser.add_argument(
        "--coverage-only", action="store_true",
        help="Only print coverage report, do not compute retrievals"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = f"data/processed/retrieved_examples_{args.split}_pathway.json"

    print(f"Loading drug profiles from {args.profiles}...")
    with open(args.profiles) as f:
        profiles = json.load(f)
    print(f"  Loaded {len(profiles):,} profiles")

    split_path = os.path.join(args.data_dir, f"{args.split}.jsonl")
    print(f"Loading {args.split} split from {split_path}...")
    split_df = pd.read_json(split_path, lines=True)
    print(f"  Loaded {len(split_df):,} pairs")

    if args.pilot > 0:
        split_df = split_df.head(args.pilot)
        print(f"  PILOT MODE: using first {len(split_df):,} pairs")

    # Coverage report
    print("\n=== Pathway Coverage Report ===")
    cov = pathway_coverage_report(profiles, split_df)
    for k, v in cov.items():
        print(f"  {k}: {v}")

    if args.coverage_only:
        return

    # Compute retrievals
    print(f"\n=== Computing Pathway Retrievals ===")
    print(f"  top_k={args.top_k}, min_diverse_classes={args.min_diverse}")

    retrievals = compute_pathway_retrievals(
        split_df=split_df,
        candidate_df=split_df,    # retrieve from same split
        profiles=profiles,
        top_k=args.top_k,
        min_diverse_classes=args.min_diverse,
        seed=args.seed,
    )

    # Save in same format as original retrieved_examples_train.json
    # Keys are string integers to match the original format
    out_serializable = {str(k): v for k, v in retrievals.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_serializable, f)
    print(f"\nSaved {len(retrievals):,} retrievals to {args.out}")

    # Quick sanity check
    n_empty = sum(1 for v in retrievals.values() if not v)
    n_full = sum(1 for v in retrievals.values() if len(v) == args.top_k)
    print(f"  Empty (no candidates): {n_empty:,}")
    print(f"  Full ({args.top_k} examples): {n_full:,}")
    print(f"  Partial: {len(retrievals) - n_empty - n_full:,}")

    print("\nDone. To use in teacher generation:")
    print(f"  cp {args.out} data/processed/retrieved_examples_train.json")
    print("  (or modify teacher_generation.py to accept --retrieval-file flag)")


if __name__ == "__main__":
    main()
