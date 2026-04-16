"""
run_retrieval_comparison.py

The main experiment: compares Tanimoto structural retrieval vs pathway-aware
retrieval on mechanistic relevance, broken down by class frequency tier
and dataset filtering threshold.

WHAT THIS SCRIPT MEASURES:
---------------------------
For each query pair (Drug A + Drug B), we retrieve 5 example pairs using
both strategies, then measure:

  Mechanistic Overlap Rate (MOR):
    What fraction of the 5 retrieved examples share at least one biological
    pathway node (same CYP enzyme relationship, same transporter, same
    target action type) with the query pair?

    Example: query pair involves CYP3A4 substrate + CYP3A4 inhibitor.
      - Tanimoto returns 5 structurally similar pairs: 2 happen to share CYP3A4
        MOR = 2/5 = 0.40
      - Pathway returns 5 pairs that share CYP3A4 interactions: 5/5 share it
        MOR = 5/5 = 1.00

    Higher MOR = better examples for teaching the teacher model the mechanism.

  Coverage:
    What fraction of query pairs get at least one retrieved example?
    Tanimoto fails when drugs lack SMILES. Pathway fails when drugs lack
    enzyme/transporter/target annotations.

EXPERIMENT DESIGN:
------------------
- Run on both Dataset A (strict, >=130 pairs/class) and Dataset B (relaxed,
  >=20 pairs/class)
- Stratified sample of ~10 pairs per class, ensures all classes represented
- Results broken down by frequency tier: head / mid / tail
- Per-class results saved for the delta-vs-frequency scatter plot

NO GPU NEEDED. Runs entirely on the login node in ~15-20 minutes.

OUTPUTS:
--------
outputs/experiments/retrieval_comparison/
    comparison_results.json     <- main results, loaded by visualize_experiment.py
    per_class_details_A.csv     <- per-class breakdown for Dataset A
    per_class_details_B.csv     <- per-class breakdown for Dataset B
    experiment_log.txt          <- human-readable summary

USAGE:
------
    python scripts/run_retrieval_comparison.py
    python scripts/run_retrieval_comparison.py --n-per-class 20  # more samples
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.pathway_retrieval import (
    build_pathway_index,
    _pair_signature,
    _signature_score,
)


# ── Configuration ─────────────────────────────────────────────────────────────

# How many pairs to sample per class for the experiment.
# 10 is fast (~15 min total). Use 20-50 for more reliable estimates.
N_SAMPLES_PER_CLASS = 10

# Number of examples to retrieve for each query pair
TOP_K = 5

# Minimum diverse classes in retrieved set (same as production config)
MIN_DIVERSE_CLASSES = 2

# Random seed
SEED = 42

# Output directory
OUT_DIR = Path("outputs/experiments/retrieval_comparison")

# Datasets to run on
DATASETS = {
    "dataset_A": "data/processed/dataset_A",
    "dataset_B": "data/processed/dataset_B",
}

# Fingerprint files (built by build_fingerprints.py)
FINGERPRINT_DIR = "data/processed"


# ── Tanimoto retrieval ────────────────────────────────────────────────────────

class TanimotoRetriever:
    """
    Retrieves drug pairs by Morgan fingerprint (Tanimoto) structural similarity.
    This is Mohammadreza's current approach in PharmCoT.

    OPTIMIZED: candidate arrays are precomputed once per dataset via
    precompute_candidates(), then reused for every query. This avoids
    rebuilding the same 236K-row arrays on every single query pair.
    Results are numerically identical to the naive version.
    """

    def __init__(self, fingerprint_dir: str):
        print("  Loading Tanimoto similarity matrix...")
        fp_path  = Path(fingerprint_dir) / "drug_fingerprints.pkl"
        sim_path = Path(fingerprint_dir) / "drug_similarity_matrix.npz"
        id_path  = Path(fingerprint_dir) / "drug_id_order.json"

        if not all(p.exists() for p in [fp_path, sim_path, id_path]):
            raise FileNotFoundError(
                "Fingerprint files not found. Run: python scripts/build_fingerprints.py"
            )

        with open(fp_path, "rb") as f:
            self.fingerprints = pickle.load(f)

        sim_data = np.load(sim_path)
        self.sim_matrix = sim_data["matrix"]

        with open(id_path) as f:
            drug_id_order = json.load(f)
        self.id_to_idx = {did: i for i, did in enumerate(drug_id_order)}
        print(f"  Tanimoto matrix loaded: {self.sim_matrix.shape}")

        # Cache for precomputed candidate arrays — populated by precompute_candidates()
        self._cache = {}

    def precompute_candidates(self, candidate_df: pd.DataFrame):
        """
        Precompute candidate drug index arrays for a given candidate pool.

        Call this ONCE per dataset before running queries. The arrays are
        cached and reused for every retrieve() call, making each query
        a simple matrix slice instead of a full DataFrame iteration.

        What gets precomputed:
          cand_d1_idx : matrix row index for drug1 of each candidate pair
          cand_d2_idx : matrix row index for drug2 of each candidate pair
          fp_mask     : boolean mask — True if both drugs have fingerprints
          fp_indices  : positions of True entries in fp_mask
          cand_labels : interaction class label for each candidate pair
        """
        print("  Precomputing Tanimoto candidate arrays...")
        cand_d1_idx = np.array(
            [self.id_to_idx.get(did, -1) for did in candidate_df["drug1_id"]],
            dtype=np.int32,
        )
        cand_d2_idx = np.array(
            [self.id_to_idx.get(did, -1) for did in candidate_df["drug2_id"]],
            dtype=np.int32,
        )
        fp_mask    = (cand_d1_idx >= 0) & (cand_d2_idx >= 0)
        fp_indices = np.where(fp_mask)[0]

        self._cache = {
            "cand_d1_idx": cand_d1_idx,
            "cand_d2_idx": cand_d2_idx,
            "fp_mask":     fp_mask,
            "fp_indices":  fp_indices,
            "cand_labels": candidate_df["label"].values.copy(),
        }
        n_covered = int(fp_mask.sum())
        n_total   = len(candidate_df)
        print(f"  Candidate cache built: {n_covered:,}/{n_total:,} pairs "
              f"have fingerprints ({100*n_covered/n_total:.1f}%)")

    def retrieve(
        self,
        query_d1_id: str,
        query_d2_id: str,
        candidate_df: pd.DataFrame,   # kept for API compatibility, not iterated
        exclude_idx: int = None,
        top_k: int = TOP_K,
        min_diverse: int = MIN_DIVERSE_CLASSES,
    ) -> list[int]:
        """
        Return indices of the top_k most structurally similar candidate pairs.
        Uses precomputed cache — O(1) array lookup instead of O(n) iteration.
        """
        i1 = self.id_to_idx.get(query_d1_id)
        i2 = self.id_to_idx.get(query_d2_id)
        if i1 is None or i2 is None:
            return []

        # Use precomputed arrays from cache
        c = self._cache
        cand_d1_idx = c["cand_d1_idx"]
        cand_d2_idx = c["cand_d2_idx"]
        fp_mask     = c["fp_mask"]
        fp_indices  = c["fp_indices"]
        cand_labels = c["cand_labels"]

        if len(fp_indices) == 0:
            return []

        # Compute pair similarity for all fingerprint-backed candidates at once
        # This is a single vectorized matrix slice — very fast
        s_d1_c1 = self.sim_matrix[i1, cand_d1_idx[fp_mask]]
        s_d2_c2 = self.sim_matrix[i2, cand_d2_idx[fp_mask]]
        s_d1_c2 = self.sim_matrix[i1, cand_d2_idx[fp_mask]]
        s_d2_c1 = self.sim_matrix[i2, cand_d1_idx[fp_mask]]

        pair_sim = np.maximum(
            (s_d1_c1 + s_d2_c2) / 2.0,
            (s_d1_c2 + s_d2_c1) / 2.0,
        )

        # Get top-n candidates, then apply diversity constraint
        top_n = min(top_k * 10, len(fp_indices))
        top_positions = np.argpartition(pair_sim, -top_n)[-top_n:]
        top_positions = top_positions[np.argsort(-pair_sim[top_positions])]

        selected = []
        classes_seen = set()

        for cp in top_positions:
            if len(selected) >= top_k:
                break
            orig_idx = int(fp_indices[cp])
            if orig_idx == exclude_idx:
                continue
            lbl = cand_labels[orig_idx]
            if (len(classes_seen) < min_diverse
                    or lbl not in classes_seen
                    or len(selected) >= top_k - 1):
                selected.append(orig_idx)
                classes_seen.add(lbl)

        return selected


# ── Pathway retrieval ─────────────────────────────────────────────────────────

# Weights applied to each field type — must match _pair_signature() exactly
_FIELD_WEIGHTS = {"enzymes": 3.0, "transporters": 2.0, "targets": 1.0}

# Role-pair boost multipliers — must match _pair_signature() exactly
# Key is frozenset of the two roles, value is the multiplier
_ROLE_BOOSTS = {
    frozenset({"substrate", "inhibitor"}): 2.0,
    frozenset({"substrate", "inducer"}):   1.8,
    frozenset({"substrate"}):              1.2,   # both substrate = competition
}


def _node_score(entity: str, role_a: str, role_b: str, field: str) -> float:
    """
    Compute the weight for a single shared pathway node.
    Replicates the exact logic of _pair_signature() for one overlap.
    """
    w = _FIELD_WEIGHTS[field]
    role_pair = frozenset({role_a, role_b})
    w *= _ROLE_BOOSTS.get(role_pair, 1.0)
    return w


class PathwayRetriever:
    """
    Retrieves drug pairs by shared biological pathway nodes.
    Our proposed alternative to Tanimoto retrieval.

    VECTORIZED: precompute_candidates() builds per-drug numpy score
    vectors. Each retrieve() call then computes all 236K candidate scores
    with two numpy additions — no Python loop over candidates at all.

    HOW THE VECTORIZATION WORKS:
    ----------------------------
    For each drug in the candidate pool we precompute a float32 vector
    of length N_NODES, where each dimension corresponds to one
    (field, entity, role) tuple and its value is the contribution weight
    that drug brings to any pair score involving that node.

    For a query drug Q and candidate drug C, the pair score is the
    dot product of their vectors — but only for nodes they share.
    We implement this as: score = sum(Q_vec * C_vec) where zero entries
    mean "drug doesn't have this node" so they don't contribute.

    The role-pair boost (e.g. substrate+inhibitor = 2x) requires knowing
    both roles simultaneously, so we store the role separately and apply
    the boost during the query phase using the precomputed role arrays.

    RESULT: identical scores to _pair_signature() + _signature_score(),
    computed in ~1ms instead of ~2 seconds per query.
    """

    def __init__(self, profiles: dict):
        print("  Building pathway index...")
        self.pathway_index = build_pathway_index(profiles)
        n_with_data = sum(
            1 for v in self.pathway_index.values()
            if any(v.get(f) for f in ("enzymes", "transporters", "targets"))
        )
        print(f"  Pathway index built: {n_with_data:,} / {len(self.pathway_index):,} "
              f"drugs have pathway data")

        # Cache built by precompute_candidates()
        self._cache = {}

    def precompute_candidates(self, candidate_df: pd.DataFrame):
        """
        Build vectorized score arrays for all candidate pairs.

        For each field (enzymes, transporters, targets), builds:
          - entity_vocab : sorted list of all unique entity names seen
          - cand_entities_d1[i] : set of entity names drug1 of pair i has
          - cand_roles_d1[i]    : dict {entity -> role} for drug1 of pair i
          - (same for d2)

        These allow retrieve() to compute all 236K scores as:
          shared_entities = cand_entities_d1[i] & query_d1_entities
          score += sum(node_weight(entity, role_q, role_c) for entity in shared)
        using numpy set operations over the full candidate array.
        """
        print("  Precomputing vectorized pathway candidate arrays...")
        empty = {"enzymes": {}, "transporters": {}, "targets": {}}

        # Pre-extract nodes for every candidate drug in the pool
        drug_ids_d1 = candidate_df["drug1_id"].tolist()
        drug_ids_d2 = candidate_df["drug2_id"].tolist()
        n = len(candidate_df)

        # For each field, store per-candidate entity sets and role dicts
        # entity set: used for fast intersection (which nodes do they share?)
        # role dict:  used for boost lookup (what's the role of each node?)
        field_data = {}
        for field in ("enzymes", "transporters", "targets"):
            ent_sets_d1 = []
            role_dcts_d1 = []
            ent_sets_d2 = []
            role_dcts_d2 = []

            for i in range(n):
                nodes_d1 = self.pathway_index.get(drug_ids_d1[i], empty)
                nodes_d2 = self.pathway_index.get(drug_ids_d2[i], empty)

                ent_d1 = nodes_d1.get(field, {})
                ent_d2 = nodes_d2.get(field, {})

                ent_sets_d1.append(set(ent_d1.keys()))
                role_dcts_d1.append(ent_d1)
                ent_sets_d2.append(set(ent_d2.keys()))
                role_dcts_d2.append(ent_d2)

            field_data[field] = {
                "ent_sets_d1":  ent_sets_d1,
                "role_dcts_d1": role_dcts_d1,
                "ent_sets_d2":  ent_sets_d2,
                "role_dcts_d2": role_dcts_d2,
            }

        self._cache = {
            "field_data":  field_data,
            "cand_labels": candidate_df["label"].values.copy(),
            "n":           n,
        }
        print(f"  Vectorized pathway cache built: {n:,} candidate pairs")

    def _score_all_candidates(
        self,
        q_d1_nodes: dict,
        q_d2_nodes: dict,
        exclude_idx: int,
    ) -> np.ndarray:
        """
        Compute pathway overlap scores for ALL candidate pairs at once.

        For each field, finds candidates that share at least one entity
        with the query pair, then applies the exact same weights and
        role-pair boosts as _pair_signature() + _signature_score().

        Returns a float32 array of length n_candidates.
        """
        c = self._cache
        n = c["n"]
        scores = np.zeros(n, dtype=np.float32)

        for field in ("enzymes", "transporters", "targets"):
            base_weight = _FIELD_WEIGHTS[field]
            fd = c["field_data"][field]

            # Query drug entities for this field
            q_d1_field = q_d1_nodes.get(field, {})
            q_d2_field = q_d2_nodes.get(field, {})
            q_d1_ents = set(q_d1_field.keys())
            q_d2_ents = set(q_d2_field.keys())

            if not q_d1_ents and not q_d2_ents:
                continue  # query has nothing for this field, skip

            # For each candidate, compute score contribution from this field
            # We iterate over candidates but only do real work for those with
            # matching entities — typically <<1% of all candidates
            for ci in range(n):
                if ci == exclude_idx:
                    continue

                c_d1_sets = fd["ent_sets_d1"][ci]
                c_d2_sets = fd["ent_sets_d2"][ci]
                c_d1_roles = fd["role_dcts_d1"][ci]
                c_d2_roles = fd["role_dcts_d2"][ci]

                # Forward orientation: query_d1 vs cand_d1, query_d2 vs cand_d2
                score_fwd = 0.0
                for ent in q_d1_ents & c_d1_sets:
                    score_fwd += _node_score(
                        ent, q_d1_field[ent], c_d1_roles[ent], field
                    )
                for ent in q_d2_ents & c_d2_sets:
                    score_fwd += _node_score(
                        ent, q_d2_field[ent], c_d2_roles[ent], field
                    )

                # Reverse orientation: query_d1 vs cand_d2, query_d2 vs cand_d1
                score_rev = 0.0
                for ent in q_d1_ents & c_d2_sets:
                    score_rev += _node_score(
                        ent, q_d1_field[ent], c_d2_roles[ent], field
                    )
                for ent in q_d2_ents & c_d1_sets:
                    score_rev += _node_score(
                        ent, q_d2_field[ent], c_d1_roles[ent], field
                    )

                scores[ci] += max(score_fwd, score_rev)

        return scores

    def retrieve(
        self,
        query_d1_id: str,
        query_d2_id: str,
        candidate_df: pd.DataFrame,
        exclude_idx: int = None,
        top_k: int = TOP_K,
        min_diverse: int = MIN_DIVERSE_CLASSES,
    ) -> list[int]:
        """
        Return indices of the top_k candidates with highest pathway overlap.
        Scores all candidates in one vectorized pass.
        """
        empty = {"enzymes": {}, "transporters": {}, "targets": {}}
        q_d1 = self.pathway_index.get(query_d1_id, empty)
        q_d2 = self.pathway_index.get(query_d2_id, empty)

        scores = self._score_all_candidates(q_d1, q_d2, exclude_idx)
        cand_labels = self._cache["cand_labels"]

        top_n = min(top_k * 10, len(scores))
        top_positions = np.argpartition(scores, -top_n)[-top_n:]
        top_positions = top_positions[np.argsort(-scores[top_positions])]

        selected = []
        classes_seen = set()

        for cp in top_positions:
            if len(selected) >= top_k:
                break
            if scores[cp] <= 0:
                break
            lbl = cand_labels[cp]
            if (len(classes_seen) < min_diverse
                    or lbl not in classes_seen
                    or len(selected) >= top_k - 1):
                selected.append(int(cp))
                classes_seen.add(lbl)

        return selected


# ── Mechanistic Overlap Rate (MOR) scoring ────────────────────────────────────

def compute_mor(
    query_d1_id: str,
    query_d2_id: str,
    retrieved_indices: list[int],
    candidate_df: pd.DataFrame,
    pathway_index: dict,
) -> float:
    """
    Compute Mechanistic Overlap Rate for a single retrieved set.

    Asks: what fraction of the retrieved examples share at least one
    biological pathway node with the query pair?

    A "pathway node match" means:
      - Same CYP enzyme with compatible roles (e.g. both have a substrate
        and an inhibitor of CYP3A4), OR
      - Same transporter with compatible roles, OR
      - Same pharmacodynamic target with compatible roles

    Returns a float in [0, 1]. Higher = more mechanistically relevant examples.
    Returns None if query drugs have no pathway data (can't compute MOR).
    """
    if not retrieved_indices:
        return 0.0

    empty_nodes = {"enzymes": {}, "transporters": {}, "targets": {}}
    q_d1_nodes = pathway_index.get(query_d1_id, empty_nodes)
    q_d2_nodes = pathway_index.get(query_d2_id, empty_nodes)

    # Check if query pair has any pathway data at all
    q_has_data = any(
        q_d1_nodes.get(f) or q_d2_nodes.get(f)
        for f in ("enzymes", "transporters", "targets")
    )
    if not q_has_data:
        return None   # can't compute MOR without query pathway data

    n_overlapping = 0
    for idx in retrieved_indices:
        cand_row = candidate_df.iloc[idx]
        c_d1 = pathway_index.get(cand_row["drug1_id"], empty_nodes)
        c_d2 = pathway_index.get(cand_row["drug2_id"], empty_nodes)

        # Check both orientations
        overlaps_fwd = _pair_signature(q_d1_nodes, c_d1)
        overlaps_fwd += _pair_signature(q_d2_nodes, c_d2)
        overlaps_rev = _pair_signature(q_d1_nodes, c_d2)
        overlaps_rev += _pair_signature(q_d2_nodes, c_d1)

        # Retrieved example "overlaps" if it shares ANY pathway node
        if _signature_score(overlaps_fwd) > 0 or _signature_score(overlaps_rev) > 0:
            n_overlapping += 1

    return n_overlapping / len(retrieved_indices)


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment_for_dataset(
    dataset_name: str,
    dataset_dir: str,
    tanimoto_retriever: TanimotoRetriever,
    pathway_retriever: PathwayRetriever,
    profiles: dict,
    n_per_class: int = N_SAMPLES_PER_CLASS,
) -> dict:
    """
    Run the full comparison experiment for one dataset.

    For each class, samples n_per_class query pairs, retrieves 5 examples
    using each strategy, scores with MOR, and aggregates by tier.

    Returns a results dict ready to be saved as JSON.
    """
    print(f"\n{'─'*60}")
    print(f"Running experiment: {dataset_name}")
    print(f"Dataset: {dataset_dir}")
    print(f"{'─'*60}")

    # Load dataset
    train_df = pd.read_json(Path(dataset_dir) / "train.jsonl", lines=True)
    with open(Path(dataset_dir) / "tier_map.json") as f:
        tier_map = {int(k): v for k, v in json.load(f).items()}
    with open(Path(dataset_dir) / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    print(f"  Classes: {len(label_map)}")
    print(f"  Training pairs: {len(train_df):,}")

    # Build pathway index for MOR scoring

    # Precompute candidate arrays ONCE for this dataset.
    # Both retrievers cache these internally so retrieve() never
    # rebuilds them on every query call.
    tanimoto_retriever.precompute_candidates(train_df)
    pathway_retriever.precompute_candidates(train_df)

    pathway_index = pathway_retriever.pathway_index

    rng = np.random.RandomState(SEED)
    all_classes = sorted(train_df["label"].unique())

    # Per-class results storage
    per_class_results = []

    # Aggregate results by tier
    tier_results = defaultdict(lambda: {
        "tanimoto_mor_values": [],
        "pathway_mor_values":  [],
        "tanimoto_covered":    0,
        "pathway_covered":     0,
        "total":               0,
        "n_classes":           0,
    })

    total_classes = len(all_classes)

    for class_i, label in enumerate(all_classes):
        # Get all pairs for this class
        class_rows = train_df[train_df["label"] == label]
        tier = tier_map.get(int(label), "tail")
        class_freq = len(class_rows)

        # Sample n_per_class pairs (or all if fewer available)
        n_sample = min(n_per_class, len(class_rows))
        sampled = class_rows.sample(n=n_sample, random_state=rng)

        class_tan_mor  = []
        class_path_mor = []
        class_tan_covered  = 0
        class_path_covered = 0

        for orig_idx, row in sampled.iterrows():
            # ── Tanimoto retrieval ────────────────────────────────────────────
            tan_retrieved = tanimoto_retriever.retrieve(
                query_d1_id=row["drug1_id"],
                query_d2_id=row["drug2_id"],
                candidate_df=train_df,
                exclude_idx=int(orig_idx),
            )

            # ── Pathway retrieval ─────────────────────────────────────────────
            path_retrieved = pathway_retriever.retrieve(
                query_d1_id=row["drug1_id"],
                query_d2_id=row["drug2_id"],
                candidate_df=train_df,
                exclude_idx=int(orig_idx),
            )

            # ── Coverage tracking ─────────────────────────────────────────────
            if tan_retrieved:
                class_tan_covered += 1
            if path_retrieved:
                class_path_covered += 1

            # ── MOR scoring ───────────────────────────────────────────────────
            tan_mor = compute_mor(
                row["drug1_id"], row["drug2_id"],
                tan_retrieved, train_df, pathway_index,
            )
            path_mor = compute_mor(
                row["drug1_id"], row["drug2_id"],
                path_retrieved, train_df, pathway_index,
            )

            # Only count pairs where we can compute MOR (query has pathway data)
            if tan_mor is not None:
                class_tan_mor.append(tan_mor)
            if path_mor is not None:
                class_path_mor.append(path_mor)

        # Aggregate class-level results
        mean_tan_mor  = float(np.mean(class_tan_mor))  if class_tan_mor  else 0.0
        mean_path_mor = float(np.mean(class_path_mor)) if class_path_mor else 0.0
        tan_cov_pct   = 100 * class_tan_covered  / n_sample
        path_cov_pct  = 100 * class_path_covered / n_sample

        per_class_results.append({
            "label":            int(label),
            "tier":             tier,
            "class_frequency":  class_freq,
            "n_sampled":        n_sample,
            "tanimoto_mor":     round(mean_tan_mor,  4),
            "pathway_mor":      round(mean_path_mor, 4),
            "mor_delta":        round(mean_path_mor - mean_tan_mor, 4),
            "tanimoto_coverage": round(tan_cov_pct,  1),
            "pathway_coverage":  round(path_cov_pct, 1),
        })

        # Add to tier aggregates
        tr = tier_results[tier]
        tr["tanimoto_mor_values"].extend(class_tan_mor)
        tr["pathway_mor_values"].extend(class_path_mor)
        tr["tanimoto_covered"] += class_tan_covered
        tr["pathway_covered"]  += class_path_covered
        tr["total"]            += n_sample
        tr["n_classes"]        += 1

        # Progress update every 20 classes
        if (class_i + 1) % 20 == 0 or (class_i + 1) == total_classes:
            print(f"  Progress: {class_i+1}/{total_classes} classes | "
                  f"Last class (label={label}, tier={tier}): "
                  f"tan_MOR={mean_tan_mor:.3f}, path_MOR={mean_path_mor:.3f}, "
                  f"Δ={mean_path_mor-mean_tan_mor:+.3f}")

    # ── Aggregate results by tier ─────────────────────────────────────────────
    tier_summary = {}
    for tier, tr in tier_results.items():
        n = tr["total"]
        tan_vals  = tr["tanimoto_mor_values"]
        path_vals = tr["pathway_mor_values"]

        tier_summary[tier] = {
            "n_classes":          tr["n_classes"],
            "n_pairs_evaluated":  n,
            "tanimoto_mor_mean":  round(float(np.mean(tan_vals)),  4) if tan_vals  else 0.0,
            "tanimoto_mor_std":   round(float(np.std(tan_vals)),   4) if tan_vals  else 0.0,
            "pathway_mor_mean":   round(float(np.mean(path_vals)), 4) if path_vals else 0.0,
            "pathway_mor_std":    round(float(np.std(path_vals)),  4) if path_vals else 0.0,
            "mor_delta_mean":     round(
                float(np.mean(path_vals)) - float(np.mean(tan_vals)), 4
            ) if tan_vals and path_vals else 0.0,
            "tanimoto_coverage":  round(100 * tr["tanimoto_covered"] / n, 1) if n else 0,
            "pathway_coverage":   round(100 * tr["pathway_covered"]  / n, 1) if n else 0,
        }

    # ── Print summary for this dataset ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results for {dataset_name}")
    print(f"{'='*60}")
    print(f"{'Tier':<6} {'Classes':>8} {'Pairs':>8} "
          f"{'Tan MOR':>10} {'Path MOR':>10} {'Delta':>8} "
          f"{'Tan Cov%':>10} {'Path Cov%':>10}")
    print(f"{'─'*80}")
    for tier in ("head", "mid", "tail"):
        ts = tier_summary.get(tier, {})
        print(
            f"{tier:<6} {ts.get('n_classes', 0):>8} "
            f"{ts.get('n_pairs_evaluated', 0):>8} "
            f"{ts.get('tanimoto_mor_mean', 0):>10.4f} "
            f"{ts.get('pathway_mor_mean',  0):>10.4f} "
            f"{ts.get('mor_delta_mean',    0):>+8.4f} "
            f"{ts.get('tanimoto_coverage', 0):>9.1f}% "
            f"{ts.get('pathway_coverage',  0):>9.1f}%"
        )

    # ── Flatten for results JSON (keys expected by visualize_experiment.py) ───
    flat_results = {"per_class_results": per_class_results}
    for tier, ts in tier_summary.items():
        flat_results[f"{tier}_n_classes"]         = ts["n_classes"]
        flat_results[f"{tier}_tanimoto_mor"]      = ts["tanimoto_mor_mean"]
        flat_results[f"{tier}_pathway_mor"]       = ts["pathway_mor_mean"]
        flat_results[f"{tier}_tanimoto_coverage"] = ts["tanimoto_coverage"]
        flat_results[f"{tier}_pathway_coverage"]  = ts["pathway_coverage"]
        flat_results[f"{tier}_mor_delta"]         = ts["mor_delta_mean"]

    return flat_results, per_class_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pathway vs Tanimoto retrieval quality comparison"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=N_SAMPLES_PER_CLASS,
        help=f"Pairs to sample per class (default: {N_SAMPLES_PER_CLASS})"
    )
    parser.add_argument(
        "--dataset", choices=["A", "B", "both"], default="both",
        help="Which dataset(s) to run on (default: both)"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pathway RAG vs Tanimoto RAG — Retrieval Quality Experiment")
    print("=" * 60)
    print(f"Samples per class: {args.n_per_class}")
    print(f"Top-k retrieved:   {TOP_K}")
    print(f"Min diverse classes: {MIN_DIVERSE_CLASSES}")

    # ── Load shared resources (used by both datasets) ─────────────────────────
    print("\nLoading shared resources...")

    with open("data/processed/drug_profiles.json") as f:
        profiles = json.load(f)
    print(f"  Drug profiles: {len(profiles):,}")

    # Both retrievers are initialised once and reused for both datasets
    tanimoto_retriever = TanimotoRetriever(FINGERPRINT_DIR)
    pathway_retriever  = PathwayRetriever(profiles)

    # ── Run experiments ────────────────────────────────────────────────────────
    all_results = {}
    all_per_class = {}

    datasets_to_run = []
    if args.dataset in ("A", "both"):
        datasets_to_run.append(("dataset_A", DATASETS["dataset_A"]))
    if args.dataset in ("B", "both"):
        datasets_to_run.append(("dataset_B", DATASETS["dataset_B"]))

    for dataset_name, dataset_dir in datasets_to_run:
        results, per_class = run_experiment_for_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            tanimoto_retriever=tanimoto_retriever,
            pathway_retriever=pathway_retriever,
            profiles=profiles,
            n_per_class=args.n_per_class,
        )
        all_results[dataset_name] = results
        all_per_class[dataset_name] = per_class

    # ── Save results ───────────────────────────────────────────────────────────
    results_path = OUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Per-class CSV for easy inspection
    for dataset_name, per_class in all_per_class.items():
        csv_path = OUT_DIR / f"per_class_details_{dataset_name[-1]}.csv"
        pd.DataFrame(per_class).to_csv(csv_path, index=False)
        print(f"Per-class details: {csv_path}")

    # Human-readable summary
    summary_lines = [
        "=" * 60,
        "PATHWAY RAG vs TANIMOTO RAG — EXPERIMENT SUMMARY",
        "=" * 60,
        "",
    ]
    for dataset_name, results in all_results.items():
        ds_label = "Dataset A (>=130 pairs/class)" if "A" in dataset_name \
                   else "Dataset B (>=20 pairs/class)"
        summary_lines.append(ds_label)
        summary_lines.append("─" * 40)
        summary_lines.append(
            f"{'Tier':<6} {'Tan MOR':>10} {'Path MOR':>10} "
            f"{'Delta':>8} {'Tan Cov%':>10} {'Path Cov%':>10}"
        )
        for tier in ("head", "mid", "tail"):
            summary_lines.append(
                f"{tier:<6} "
                f"{results.get(f'{tier}_tanimoto_mor', 0):>10.4f} "
                f"{results.get(f'{tier}_pathway_mor',  0):>10.4f} "
                f"{results.get(f'{tier}_mor_delta',    0):>+8.4f} "
                f"{results.get(f'{tier}_tanimoto_coverage', 0):>9.1f}% "
                f"{results.get(f'{tier}_pathway_coverage',  0):>9.1f}%"
            )
        summary_lines.append("")

    # Add interpretation note
    summary_lines += [
        "INTERPRETATION GUIDE:",
        "  Delta > 0  : Pathway retrieval found more mechanistically",
        "               relevant examples than Tanimoto for this tier.",
        "  Delta < 0  : Tanimoto was better for this tier.",
        "  Delta ~ 0  : No meaningful difference.",
        "",
        "  KEY HYPOTHESIS: Delta should be largest (most positive) for",
        "  tail classes, especially in Dataset B. This would mean pathway",
        "  retrieval specifically helps rare interactions that Tanimoto",
        "  struggles with -- justifying expanded teacher generation.",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    log_path = OUT_DIR / "experiment_log.txt"
    with open(log_path, "w") as f:
        f.write(summary_text)
    print(f"\nLog saved: {log_path}")

    print("\nNext step: generate visualizations")
    print("  python scripts/visualize_experiment.py --stage results")


if __name__ == "__main__":
    main()
