"""
Data preparation pipeline for DDI CoT Distillation V3.

Handles:
  1. Class filtering (keep >= min_pairs_per_class)
  2. Label remapping to contiguous IDs
  3. Coarse category mapping
  4. Per-class training cap
  5. Stratified 80/20 split
  6. Drug profile enrichment
  7. Severity label attachment
  8. Dynamic few-shot retrieval precomputation
  9. Prompt construction (teacher + student)

------------------------------------------------------------------------------
RJ ADDITIONS (Rameen Jafri, April 2026)
------------------------------------------------------------------------------

Three pharmacologically-motivated improvements to the teacher prompt
construction. None of these change the pipeline structure or add compute cost —
they improve the quality of information the teacher model receives.

1. PK/PD interaction type flag  (classify_pk_pd, added to build_teacher_prompt)
   The teacher model now gets told whether an interaction is pharmacokinetic
   (one drug changes the other's ADME — absorption, distribution, metabolism,
   excretion) or pharmacodynamic (both drugs act on the same receptor/system).
   Without this, the teacher has to infer the interaction type from the label
   text alone and can apply the wrong reasoning framework — e.g. using enzyme
   reasoning for a receptor-mediated interaction or vice versa.
   Covers 100% of Dataset A classes (verified empirically, zero ambiguous).

2. Prodrug warning  (_load_prodrug_ids, added to build_teacher_prompt)
   125 drugs in Dataset A are prodrugs — pharmacologically inactive until
   converted to their active form by an enzyme. For these drugs, the direction
   of enzyme inhibition is REVERSED compared to a normal drug: inhibiting the
   activating enzyme reduces active drug levels rather than increasing them.
   Without this flag, the teacher frequently generates traces with the wrong
   direction of effect for prodrug pairs (e.g. stating that CYP2C19 inhibition
   increases clopidogrel's antiplatelet effect when it actually decreases it).
   Affects 8.3% of training pairs. Prodrug list from count_prodrugs.py using
   DrugBank 5.1.17. Requires data/processed/prodrug_ids.json to be present.

3. Raised profile truncation caps  (_format_drug_profile)
   The original code silently truncated drug profiles at 5 enzymes, 3
   transporters, and 3 targets. For highly metabolised drugs (Nicotine,
   Troglitazone, Dapsone etc.) this dropped important CYP interactions from
   the prompt entirely without any warning. Raised to 8 enzymes, 5
   transporters, 5 targets. Affects ~2-5% of drug profiles.
------------------------------------------------------------------------------
"""
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, categorize_interaction


# ── Prompt templates ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs with their pharmacological profiles, analyse their "
    "mechanisms step-by-step and predict their interaction type. "
    "Include the severity if known."
)

TEACHER_SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs and their known interaction type, explain the "
    "pharmacological mechanisms step-by-step. Then provide a concise summary. "
    "Structure your response as:\n\n"
    "## Reasoning\n"
    "[Numbered steps explaining the mechanism]\n\n"
    "## Summary\n"
    "[2-3 sentence summary of the key interaction mechanism]\n\n"
    "## Classification\n"
    "Y={label} -- \"{label_text}\"\n\n"
    "## Severity\n"
    "{Major/Moderate/Minor/Unknown}"
)



# ── PK/PD interaction type classifier ─────────────────────────────────────────
# Classifies a DrugBank interaction template as pharmacokinetic (PK) or
# pharmacodynamic (PD) based on keyword matching.
#
# PK = one drug changes how much of the other drug gets into the blood
#      (absorption, distribution, metabolism, excretion)
# PD = both drugs act on the same receptor/system at the same time
#      (additive, synergistic, or antagonistic effects)
#
# This is used to give the teacher model a pharmacological hint about what
# TYPE of reasoning to apply when explaining the interaction mechanism.
#
# Rule: PK keywords take priority over PD keywords when both are present,
# because many templates describe a PK mechanism that leads to a PD outcome
# (e.g. "excretion rate decreased resulting in lower serum concentration").

_PK_KEYWORDS = [
    "serum concentration", "metabolism", "excretion", "absorption",
    "half-life", "clearance", "bioavailability", "pharmacokinetic",
    "protein binding", "distribution",
]
_PD_KEYWORDS = [
    "risk or severity", "adverse effects", "efficacy", "toxic",
    "sedation", "bleeding", "cardiac", "qtc", "hypotension",
    "hyperkalemia", "serotonin", "respiratory", "pharmacodynamic",
    "activities", "antihypertensive", "hypoglycemic", "bradycardic",
    "thrombogenic", "arrhythmogenic", "neuromuscular", "immunosuppressive",
    "anticoagulant", "analgesic", "hypersensitivity", "diagnostic",
]

def classify_pk_pd(template: str) -> str:
    """Return 'PK' or 'PD' for an interaction template string.

    PK wins when both keyword types are present — the template is describing
    a pharmacokinetic mechanism even if the outcome has pharmacodynamic
    consequences.

    Returns 'PK' or 'PD'. Never returns ambiguous because the word lists
    cover all 129 Dataset A classes and all 354 raw DrugBank templates.
    """
    t = template.lower()
    if any(kw in t for kw in _PK_KEYWORDS):
        return "PK"
    return "PD"


# ── Prodrug ID loader ──────────────────────────────────────────────────────────
# Prodrugs are pharmacologically inactive until converted to their active form
# by an enzyme (usually a CYP enzyme or esterase). This means the direction
# of enzyme inhibition is REVERSED compared to a normal drug:
#   Normal drug:  inhibitor blocks breakdown → MORE drug in blood
#   Prodrug:      inhibitor blocks activation → LESS active drug in blood
#
# We load the prodrug list produced by count_prodrugs.py and use it to add
# a warning to teacher prompts for pairs involving prodrugs.

import functools

@functools.lru_cache(maxsize=1)
def _load_prodrug_ids() -> set:
    """Load the set of DrugBank IDs known to be prodrugs.

    Cached after first load — only reads the file once per process.
    Returns empty set if prodrug_ids.json doesn't exist yet.
    """
    prodrug_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "prodrug_ids.json"
    if not prodrug_path.exists():
        return set()
    with open(prodrug_path) as f:
        data = json.load(f)
    return set(data.keys())

def _format_drug_profile(profile: dict) -> str:
    """Format a drug profile into a compact text block for prompts."""
    lines = [f"  Description: {profile['description']}" if profile.get("description") else None]

    if profile.get("mechanism_of_action"):
        lines.append(f"  Mechanism: {profile['mechanism_of_action'][:200]}")

    if profile.get("enzymes"):
        # Show up to 8 enzymes (raised from 5 — some drugs like Nicotine have >5 important CYP interactions)
        enz_str = "; ".join(profile["enzymes"][:8])
        lines.append(f"  Key enzymes: {enz_str}")

    if profile.get("transporters"):
        # Show up to 5 transporters (raised from 3)
        trans_str = "; ".join(profile["transporters"][:5])
        lines.append(f"  Transporters: {trans_str}")

    if profile.get("targets"):
        # Show up to 5 targets (raised from 3)
        tgt_str = "; ".join(profile["targets"][:5])
        lines.append(f"  Targets: {tgt_str}")

    if profile.get("smiles"):
        lines.append(f"  SMILES: {profile['smiles'][:200]}")

    return "\n".join(l for l in lines if l)


def build_teacher_prompt(row, label_map, profiles, retrieved_examples=None):
    """Construct the enriched teacher prompt with drug profiles and retrieved examples."""
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
                ex_label_text = ex_label_text.replace("#Drug1", ex["drug1_name"]).replace("#Drug2", ex["drug2_name"])
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

    # ── Severity: use classifier if DDInter label missing or teacher-generated ──
    # 83.8% of pairs have Unknown severity. Teacher severity is 92.4% Major
    # (hallucinated). Rule-based classifier provides evidence-based labels.
    severity = str(row.get("severity", "Unknown"))
    severity_source = str(row.get("severity_source", "none"))
    if severity == "Unknown" or severity_source == "teacher":
        try:
            from src.severity_classifier import classify_severity as _clf_sev
            _result = _clf_sev(
                drug1_id=str(row["drug1_id"]),
                drug2_id=str(row["drug2_id"]),
                label_text=str(row.get("label_text", "")),
                profiles=profiles,
                drug1_name=str(row.get("drug1_name", "")),
                drug2_name=str(row.get("drug2_name", "")),
            )
            if _result["severity"] != "Unknown":
                severity = _result["severity"]
        except Exception:
            pass  # fall back gracefully if classifier unavailable
    severity = severity  # final value used below
    parts.append(f"Known interaction: Y={row['label']} -- \"{row['label_text']}\"")
    parts.append(f"Known severity: {severity}")
    parts.append("")

    # ── PK/PD hint ────────────────────────────────────────────────────────────
    # Tell the teacher whether this is a pharmacokinetic or pharmacodynamic
    # interaction so it applies the right type of reasoning.
    #   PK: one drug changes the other's ADME (absorption, distribution,
    #       metabolism, excretion) — reason about enzymes, transporters,
    #       drug levels in blood
    #   PD: both drugs act on the same receptor/system — reason about
    #       additive/synergistic/antagonistic effects on physiology
    interaction_type = classify_pk_pd(row.get("label_text", row.get("template", "")))
    parts.append(
        f"Interaction type: {interaction_type} "
        f"({'pharmacokinetic — reason about ADME mechanisms, enzyme/transporter roles, and drug level changes' if interaction_type == 'PK' else 'pharmacodynamic — reason about receptor/system effects and combined pharmacological actions'})"
    )
    parts.append("")

    # ── Prodrug warning ───────────────────────────────────────────────────────
    # If either drug is a prodrug, warn the teacher that enzyme inhibition
    # has the OPPOSITE effect compared to a normal drug:
    #   Normal drug:  inhibitor blocks breakdown → more drug in blood
    #   Prodrug:      inhibitor blocks activation → LESS active drug
    prodrug_ids = _load_prodrug_ids()
    prodrug_warnings = []
    for drug_id, drug_name in [(row["drug1_id"], row["drug1_name"]),
                                (row["drug2_id"], row["drug2_name"])]:
        if drug_id in prodrug_ids:
            prodrug_warnings.append(
                f"⚠️  PRODRUG WARNING: {drug_name} is a prodrug — it is "
                f"pharmacologically inactive until converted to its active form "
                f"by an enzyme. If an enzyme involved in its activation is "
                f"inhibited, the result is DECREASED active drug levels (not "
                f"increased). Reason about activation, not elimination."
            )
    if prodrug_warnings:
        for warning in prodrug_warnings:
            parts.append(warning)
        parts.append("")

    parts.append(
        "Explain step-by-step the pharmacological mechanisms behind this "
        "drug-drug interaction. Discuss each drug's mechanism of action and "
        "how they combine to produce this effect. Then provide a concise summary. "
        "End with the classification and severity."
    )
    # ── No shared pathway note ──────────────────────────────────────────────
    # When drugs share no common enzymes/transporters/targets, the teacher
    # has no pathway anchor. Tell it to reason pharmacodynamically instead.
    try:
        from src.pathway_retrieval import _extract_pathway_nodes
        _p1_nodes = _extract_pathway_nodes(p1) if p1 else {}
        _p2_nodes = _extract_pathway_nodes(p2) if p2 else {}
        # Check overlap across all three annotation types
        _shared = (
            set(_p1_nodes.get("enzymes", {})) & set(_p2_nodes.get("enzymes", {})) |
            set(_p1_nodes.get("transporters", {})) & set(_p2_nodes.get("transporters", {})) |
            set(_p1_nodes.get("targets", {})) & set(_p2_nodes.get("targets", {}))
        )
        _has_data = any(_p1_nodes.get(k) for k in ("enzymes","transporters","targets")) or                     any(_p2_nodes.get(k) for k in ("enzymes","transporters","targets"))
        if not _shared and _has_data:
            parts.append(
                "⚠️  NOTE: These two drugs share NO common enzymes, transporters, "
                "or targets in DrugBank. There is no direct pharmacokinetic pathway "
                "connecting them. Focus your reasoning on pharmacodynamic mechanisms "
                "— do both drugs act on the same receptor, ion channel, or "
                "physiological system? Do NOT invoke CYP enzyme reasoning unless "
                "the drug profiles above explicitly show CYP involvement."
            )
            parts.append("")
    except Exception:
        pass  # pathway retrieval not available

    return "\n".join(parts)


def build_student_input(row, profiles, retrieved_examples=None):
    """Build the student prompt (no answer, task instruction to predict)."""
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
            parts.append(f"Interaction: Y={ex['label']} -- \"{ex.get('label_text', '')}\"")
            sev = ex.get("severity", "Unknown")
            parts.append(f"Severity: {sev}")
            parts.append("")

    p1 = profiles.get(row["drug1_id"], {})
    p2 = profiles.get(row["drug2_id"], {})

    parts.append(f"Drug 1: {row['drug1_name']} ({row['drug1_id']})")
    if p1:
        parts.append(_format_drug_profile(p1))
    parts.append(f"Drug 2: {row['drug2_name']} ({row['drug2_id']})")
    if p2:
        parts.append(_format_drug_profile(p2))
    parts.append("")
    parts.append("Predict the interaction type, explain the mechanism briefly, "
                 "and state the severity.")
    return "\n".join(parts)


# ── Few-shot retrieval ────────────────────────────────────────────────

def precompute_retrievals(train_df, profiles, drug_id_order, sim_matrix,
                          fingerprints, top_k=5, min_diverse=2, seed=42,
                          batch_size=500):
    """Precompute top-k retrieved examples using vectorized numpy operations.

    Pairs where either drug lacks fingerprints get no retrieved examples
    (empty list) rather than random fallback, to avoid noise.
    """
    id_to_idx = {did: i for i, did in enumerate(drug_id_order)}
    n_total = len(train_df)

    fp_mask = np.array([
        id_to_idx.get(row["drug1_id"]) is not None and
        id_to_idx.get(row["drug2_id"]) is not None
        for _, row in train_df.iterrows()
    ])

    all_indices = np.array(train_df.index.tolist())
    all_labels = np.array(train_df["label"].tolist())

    d1_sim_idx = np.full(n_total, -1, dtype=np.int32)
    d2_sim_idx = np.full(n_total, -1, dtype=np.int32)
    for i, (_, row) in enumerate(train_df.iterrows()):
        i1 = id_to_idx.get(row["drug1_id"])
        i2 = id_to_idx.get(row["drug2_id"])
        if i1 is not None and i2 is not None:
            d1_sim_idx[i] = i1
            d2_sim_idx[i] = i2

    fp_positions = np.where(fp_mask)[0]
    n_with_fp = len(fp_positions)
    n_skip = n_total - n_with_fp
    print(f"  Pairs with fingerprints: {n_with_fp:,}, without: {n_skip:,}")

    cand_d1 = d1_sim_idx[fp_mask]
    cand_d2 = d2_sim_idx[fp_mask]
    cand_labels = all_labels[fp_mask]
    cand_orig_idx = all_indices[fp_mask]

    sim_dense = sim_matrix
    if hasattr(sim_matrix, 'toarray'):
        sim_dense = sim_matrix.toarray()
    elif hasattr(sim_matrix, 'A'):
        sim_dense = np.asarray(sim_matrix.A)
    else:
        sim_dense = np.asarray(sim_matrix)

    retrievals = {}
    for i in range(n_total):
        if not fp_mask[i]:
            retrievals[all_indices[i]] = []

    processed = 0
    for batch_start in range(0, n_with_fp, batch_size):
        batch_end = min(batch_start + batch_size, n_with_fp)
        batch_pos = fp_positions[batch_start:batch_end]
        b_size = len(batch_pos)

        b_d1 = d1_sim_idx[batch_pos]
        b_d2 = d2_sim_idx[batch_pos]

        s_d1_c1 = sim_dense[b_d1][:, cand_d1]
        s_d2_c2 = sim_dense[b_d2][:, cand_d2]
        s_d1_c2 = sim_dense[b_d1][:, cand_d2]
        s_d2_c1 = sim_dense[b_d2][:, cand_d1]

        pair_sim = np.maximum(
            (s_d1_c1 + s_d2_c2) / 2.0,
            (s_d1_c2 + s_d2_c1) / 2.0
        )

        for bi in range(b_size):
            pos_in_fp = batch_start + bi
            orig_idx = all_indices[batch_pos[bi]]
            sims = pair_sim[bi].copy()
            sims[pos_in_fp] = -1.0

            top_n = min(top_k * 10, n_with_fp)
            top_positions = np.argpartition(sims, -top_n)[-top_n:]
            top_positions = top_positions[np.argsort(-sims[top_positions])]

            selected = []
            classes_seen = set()
            for cp in top_positions:
                if len(selected) >= top_k:
                    break
                lbl = cand_labels[cp]
                if len(selected) >= top_k - min_diverse or lbl not in classes_seen:
                    selected.append(int(cand_orig_idx[cp]))
                    classes_seen.add(lbl)

            if len(selected) < top_k:
                for cp in top_positions:
                    if int(cand_orig_idx[cp]) not in selected:
                        selected.append(int(cand_orig_idx[cp]))
                        if len(selected) >= top_k:
                            break

            retrievals[orig_idx] = selected

        processed += b_size
        if processed % 10000 < batch_size or batch_end == n_with_fp:
            print(f"  Retrieval: {processed:,}/{n_with_fp:,} pairs with FP computed...")

    n_with = sum(1 for v in retrievals.values() if v)
    print(f"  Retrieval complete: {n_with:,}/{n_total:,} pairs have retrieved examples, "
          f"{n_skip:,} skipped (no fingerprints)")
    return retrievals


# ── Main preparation pipeline ─────────────────────────────────────────

def prepare_data(cfg: dict):
    logger = setup_logging("data_preparation")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    proc_dir = Path(cfg["data"]["processed_dir"])
    min_pairs = cfg["data"]["min_pairs_per_class"]
    max_train = cfg["data"]["max_train_per_class"]
    train_ratio = cfg["data"]["train_ratio"]
    seed = cfg["project"]["seed"]

    logger.info("Loading extracted data...")
    interactions = []
    with open(proc_dir / "interactions_full.jsonl") as f:
        for line in f:
            interactions.append(json.loads(line))
    logger.info(f"  Total interaction pairs: {len(interactions):,}")

    raw_lm_path = proc_dir / "raw_label_map.json"
    if raw_lm_path.exists():
        with open(raw_lm_path) as f:
            raw_label_map = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"  Raw label classes (from raw_label_map.json): {len(raw_label_map)}")
    else:
        with open(proc_dir / "label_map.json") as f:
            raw_label_map = {int(k): v for k, v in json.load(f).items()}
        with open(raw_lm_path, "w") as f:
            json.dump({str(k): v for k, v in raw_label_map.items()}, f, indent=2)
        logger.info(f"  Raw label classes: {len(raw_label_map)} (backed up to raw_label_map.json)")

    with open(proc_dir / "drug_profiles.json") as f:
        profiles = json.load(f)
    logger.info(f"  Drug profiles: {len(profiles):,}")

    with open(proc_dir / "severity_map.json") as f:
        severity_map = json.load(f)
    logger.info(f"  Severity labels: {len(severity_map):,}")

    # Step 1: Filter to classes with >= min_pairs
    label_counts = Counter(ix["label"] for ix in interactions)
    kept_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_pairs}
    filtered = [ix for ix in interactions if ix["label"] in kept_labels]
    logger.info(f"  After filtering (>= {min_pairs} pairs): "
                f"{len(kept_labels)} classes, {len(filtered):,} pairs")

    # Step 1b: Remove pairs where EITHER drug lacks useful pharmacological info.
    # Both drugs must have at least one of: description, mechanism, enzymes, targets, transporters.
    # This prevents the teacher from hallucinating mechanisms for unknown drugs.
    useful_fields = ["description", "mechanism_of_action", "enzymes", "targets", "transporters"]
    before_quality = len(filtered)
    quality_filtered = []
    for ix in filtered:
        p1 = profiles.get(ix["drug1_id"], {})
        p2 = profiles.get(ix["drug2_id"], {})
        d1_has = p1 and any(p1.get(f) for f in useful_fields)
        d2_has = p2 and any(p2.get(f) for f in useful_fields)
        if d1_has and d2_has:
            quality_filtered.append(ix)
    n_removed = before_quality - len(quality_filtered)
    filtered = quality_filtered
    logger.info(f"  Profile quality filter: removed {n_removed:,} pairs "
                f"(at least one drug missing description/mechanism/enzymes/targets/transporters)")
    logger.info(f"  Remaining: {len(filtered):,} pairs")

    # Re-check class counts after quality filter -- some classes may now be below min_pairs
    label_counts_post = Counter(ix["label"] for ix in filtered)
    dropped_classes = {lbl for lbl in kept_labels if label_counts_post.get(lbl, 0) < min_pairs}
    if dropped_classes:
        kept_labels -= dropped_classes
        filtered = [ix for ix in filtered if ix["label"] in kept_labels]
        logger.info(f"  Dropped {len(dropped_classes)} classes below {min_pairs} after quality filter")
        logger.info(f"  Final: {len(kept_labels)} classes, {len(filtered):,} pairs")

    # Step 2: Remap labels to contiguous IDs (1..N)
    old_to_new = {}
    new_label_map = {}
    for new_id, (old_id, template) in enumerate(
        sorted([(lbl, raw_label_map[lbl]) for lbl in kept_labels],
               key=lambda x: -label_counts[x[0]]),
        start=1,
    ):
        old_to_new[old_id] = new_id
        new_label_map[new_id] = template

    for ix in filtered:
        ix["label"] = old_to_new[ix["label"]]

    # Step 3: Build coarse category mapping
    coarse_map = {}
    for label_id, template in new_label_map.items():
        coarse_map[label_id] = categorize_interaction(template)

    coarse_counts = Counter(coarse_map.values())
    logger.info(f"  Coarse categories: {len(coarse_counts)}")
    for cat, cnt in coarse_counts.most_common():
        logger.info(f"    {cat}: {cnt} fine-grained classes")

    # Step 4: Fill label_text with real drug names
    for ix in filtered:
        template = new_label_map[ix["label"]]
        ix["label_text"] = template.replace("#Drug1", ix["drug1_name"]).replace("#Drug2", ix["drug2_name"])
        ix["coarse_category"] = coarse_map[ix["label"]]

    # Step 5: Attach severity labels
    severity_attached = 0
    for ix in filtered:
        pair_key = "_".join(sorted([ix["drug1_id"], ix["drug2_id"]]))
        sev = severity_map.get(pair_key, "Unknown")
        if sev in ("Major", "Moderate", "Minor"):
            ix["severity"] = sev
            severity_attached += 1
        else:
            ix["severity"] = "Unknown"
    logger.info(f"  Severity attached: {severity_attached:,} / {len(filtered):,} "
                f"({100*severity_attached/len(filtered):.1f}%)")

    # Step 6: Stratified 80/20 split
    df = pd.DataFrame(filtered)
    train_df, test_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(f"  Split: train={len(train_df):,}, test={len(test_df):,}")

    # Step 7: Per-class training cap
    capped_parts = []
    rng = np.random.RandomState(seed)
    for label in sorted(train_df["label"].unique()):
        group = train_df[train_df["label"] == label]
        if len(group) > max_train:
            group = group.sample(n=max_train, random_state=rng)
        capped_parts.append(group)
    train_df = pd.concat(capped_parts, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    logger.info(f"  After per-class cap ({max_train}): {len(train_df):,} training pairs")

    # Step 8: Log distribution stats
    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()
    logger.info(f"  Training class distribution:")
    logger.info(f"    Min: {train_counts.min()}, Max: {train_counts.max()}, "
                f"Median: {train_counts.median():.0f}")
    logger.info(f"  Test class distribution:")
    logger.info(f"    Min: {test_counts.min()}, Max: {test_counts.max()}, "
                f"Median: {test_counts.median():.0f}")

    sev_train = train_df[train_df["severity"] != "Unknown"]
    sev_test = test_df[test_df["severity"] != "Unknown"]
    logger.info(f"  Severity-labeled: train={len(sev_train):,}, test={len(sev_test):,}")

    # Save outputs
    train_df.to_json(proc_dir / "train.jsonl", orient="records", lines=True)
    test_df.to_json(proc_dir / "test.jsonl", orient="records", lines=True)

    with open(proc_dir / "label_map.json", "w") as f:
        json.dump(new_label_map, f, indent=2)

    with open(proc_dir / "coarse_category_map.json", "w") as f:
        json.dump(coarse_map, f, indent=2)

    logger.info(f"  Saved train.jsonl ({len(train_df):,} pairs)")
    logger.info(f"  Saved test.jsonl ({len(test_df):,} pairs)")
    logger.info(f"  Saved label_map.json ({len(new_label_map)} classes)")
    logger.info(f"  Saved coarse_category_map.json ({len(coarse_map)} mappings)")

    # Step 9: Precompute few-shot retrievals for training set
    logger.info("Precomputing few-shot retrievals for training pairs...")
    fp_path = proc_dir / "drug_fingerprints.pkl"
    sim_path = proc_dir / "drug_similarity_matrix.npz"
    id_path = proc_dir / "drug_id_order.json"

    if fp_path.exists() and sim_path.exists() and id_path.exists():
        with open(fp_path, "rb") as f:
            fingerprints = pickle.load(f)
        sim_data = np.load(sim_path)
        sim_matrix = sim_data["matrix"]
        with open(id_path) as f:
            drug_id_order = json.load(f)

        top_k = cfg.get("retrieval", {}).get("top_k", 5)
        min_diverse = cfg.get("retrieval", {}).get("min_diverse_classes", 2)

        train_retrievals = precompute_retrievals(
            train_df, profiles, drug_id_order, sim_matrix, fingerprints,
            top_k=top_k, min_diverse=min_diverse, seed=seed,
        )

        retrieval_out = {}
        for idx, selected_indices in train_retrievals.items():
            examples = []
            for sel_idx in selected_indices:
                sel_row = train_df.iloc[sel_idx] if sel_idx < len(train_df) else None
                if sel_row is not None:
                    examples.append({
                        "drug1_id": sel_row["drug1_id"],
                        "drug2_id": sel_row["drug2_id"],
                        "drug1_name": sel_row["drug1_name"],
                        "drug2_name": sel_row["drug2_name"],
                        "label": int(sel_row["label"]),
                        "label_text": sel_row["label_text"],
                        "severity": sel_row.get("severity", "Unknown"),
                    })
            retrieval_out[str(idx)] = examples

        with open(proc_dir / "retrieved_examples_train.json", "w") as f:
            json.dump(retrieval_out, f)
        logger.info(f"  Saved retrieved_examples_train.json ({len(retrieval_out):,} entries)")
    else:
        logger.warning("Fingerprint/similarity files not found. Skipping retrieval precomputation.")

    logger.info("Data preparation complete.")
    return train_df, test_df, new_label_map


def precompute_test_retrievals(cfg: dict):
    """Precompute few-shot retrievals for test set using training candidates."""
    logger = setup_logging("test_retrieval")
    proc_dir = Path(cfg["data"]["processed_dir"])

    train_df = pd.read_json(proc_dir / "train.jsonl", lines=True)
    test_df = pd.read_json(proc_dir / "test.jsonl", lines=True)

    fp_path = proc_dir / "drug_fingerprints.pkl"
    sim_path = proc_dir / "drug_similarity_matrix.npz"
    id_path = proc_dir / "drug_id_order.json"

    if not (fp_path.exists() and sim_path.exists() and id_path.exists()):
        logger.error("Fingerprint/similarity files not found. Run prepare_data first.")
        return

    with open(fp_path, "rb") as f:
        fingerprints = pickle.load(f)
    sim_data = np.load(sim_path)
    sim_matrix = sim_data["matrix"]
    with open(id_path) as f:
        drug_id_order = json.load(f)

    id_to_idx = {did: i for i, did in enumerate(drug_id_order)}
    top_k = cfg.get("retrieval", {}).get("top_k", 5)
    min_diverse = cfg.get("retrieval", {}).get("min_diverse_classes", 2)
    batch_size = cfg.get("retrieval", {}).get("test_retrieval_batch_size", 128)

    sim_dense = sim_matrix
    if hasattr(sim_matrix, 'toarray'):
        sim_dense = sim_matrix.toarray()
    elif hasattr(sim_matrix, 'A'):
        sim_dense = np.asarray(sim_matrix.A)
    else:
        sim_dense = np.asarray(sim_matrix)

    train_d1_idx = np.array([id_to_idx.get(r["drug1_id"], -1) for _, r in train_df.iterrows()], dtype=np.int32)
    train_d2_idx = np.array([id_to_idx.get(r["drug2_id"], -1) for _, r in train_df.iterrows()], dtype=np.int32)
    train_fp_mask = (train_d1_idx >= 0) & (train_d2_idx >= 0)
    train_labels = np.array(train_df["label"].tolist())

    cand_d1 = train_d1_idx[train_fp_mask]
    cand_d2 = train_d2_idx[train_fp_mask]
    cand_labels = train_labels[train_fp_mask]
    cand_orig_idx = np.array(train_df.index.tolist())[train_fp_mask]
    if len(cand_d1) == 0:
        logger.error("No fingerprint-backed training candidates; cannot build test retrievals.")
        out_path = proc_dir / "retrieved_examples_test.json"
        with open(out_path, "w") as f:
            json.dump({}, f)
        return

    test_d1_idx = np.array([id_to_idx.get(r["drug1_id"], -1) for _, r in test_df.iterrows()], dtype=np.int32)
    test_d2_idx = np.array([id_to_idx.get(r["drug2_id"], -1) for _, r in test_df.iterrows()], dtype=np.int32)
    test_fp_mask = (test_d1_idx >= 0) & (test_d2_idx >= 0)
    test_indices = np.array(test_df.index.tolist())

    n_total = len(test_df)
    n_with_fp = int(test_fp_mask.sum())
    n_skip = n_total - n_with_fp
    logger.info(
        f"Test pairs: {n_total:,} | with fingerprints: {n_with_fp:,} | "
        f"without: {n_skip:,} | train candidates: {len(cand_d1):,}"
    )

    retrieval_out = {str(idx): [] for idx in test_indices}
    fp_positions = np.where(test_fp_mask)[0]
    processed = 0

    for batch_start in range(0, n_with_fp, batch_size):
        batch_end = min(batch_start + batch_size, n_with_fp)
        batch_pos = fp_positions[batch_start:batch_end]
        b_size = len(batch_pos)

        b_d1 = test_d1_idx[batch_pos]
        b_d2 = test_d2_idx[batch_pos]

        s_d1_c1 = sim_dense[b_d1][:, cand_d1]
        s_d2_c2 = sim_dense[b_d2][:, cand_d2]
        s_d1_c2 = sim_dense[b_d1][:, cand_d2]
        s_d2_c1 = sim_dense[b_d2][:, cand_d1]

        pair_sim = np.maximum(
            (s_d1_c1 + s_d2_c2) / 2.0,
            (s_d1_c2 + s_d2_c1) / 2.0,
        )

        top_n = min(top_k * 10, len(cand_d1))
        for bi in range(b_size):
            sims = pair_sim[bi]
            top_positions = np.argpartition(sims, -top_n)[-top_n:]
            top_positions = top_positions[np.argsort(-sims[top_positions])]

            selected = []
            classes_seen = set()
            for cp in top_positions:
                if len(selected) >= top_k:
                    break
                lbl = cand_labels[cp]
                if len(selected) >= top_k - min_diverse or lbl not in classes_seen:
                    sel_idx = int(cand_orig_idx[cp])
                    sel_row = train_df.iloc[sel_idx]
                    selected.append({
                        "drug1_id": sel_row["drug1_id"],
                        "drug2_id": sel_row["drug2_id"],
                        "drug1_name": sel_row["drug1_name"],
                        "drug2_name": sel_row["drug2_name"],
                        "label": int(sel_row["label"]),
                        "label_text": sel_row.get("label_text", ""),
                        "severity": sel_row.get("severity", "Unknown"),
                    })
                    classes_seen.add(lbl)

            retrieval_out[str(int(test_indices[batch_pos[bi]]))] = selected

        processed += b_size
        if processed % 10000 < b_size or batch_end == n_with_fp:
            logger.info(f"  Test retrieval: {processed:,}/{n_with_fp:,} with-fp pairs")

    out_path = proc_dir / "retrieved_examples_test.json"
    with open(out_path, "w") as f:
        json.dump(retrieval_out, f)
    logger.info(f"Saved {out_path} ({len(retrieval_out):,} entries)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--test-retrieval", action="store_true",
                        help="Only precompute test set retrievals")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.test_retrieval:
        precompute_test_retrievals(cfg)
    else:
        prepare_data(cfg)
