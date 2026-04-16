"""
run_subset_pilot.py

Compares Tanimoto retrieval vs pathway retrieval on a stratified sample
of 4000 training pairs using Qwen2.5-72B-Instruct as the teacher model.

For each retrieval strategy:
  1. Generates structured CoT traces (with JSONL checkpointing — safe to restart)
  2. Applies hard rejection rules
  3. Scores with grounded factuality
  4. Produces a side-by-side comparison report

This is the core experiment for RJ contribution 1:
  "Does pathway-aware RAG retrieval produce better teacher traces than
   Tanimoto structural similarity retrieval?"

The metric is grounded factuality score — how many named pharmacological
entities (CYPs, transporters, targets) in the trace actually match the
drug profiles in DrugBank. Higher = more factually grounded reasoning.

CHECKPOINTING:
  Both trace files are JSONL-appended. If the job is killed and restarted,
  it resumes from where it stopped. Safe to run across multiple SLURM jobs.

OUTPUTS:
  outputs/teacher_traces/tanimoto_traces.jsonl
  outputs/teacher_traces/pathway_traces.jsonl
  outputs/comparison_report.json
  outputs/comparison_report.txt

USAGE:
  python scripts/run_subset_pilot.py --config configs/config_pilot.yaml
  python scripts/run_subset_pilot.py --config configs/config_pilot.yaml --score-only
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.utils import load_config, setup_logging, set_seed
from src.pathway_retrieval import compute_pathway_retrievals, build_pathway_index
from src.grounded_factuality import score_trace


# ── Stratified sampling ────────────────────────────────────────────────────────

def sample_pairs_stratified(
    train_df: pd.DataFrame,
    tier_map: dict,
    n_total: int = 4000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_total pairs stratified by frequency tier.

    Allocation mirrors the tier distribution in the full dataset:
      head (15 classes, ~43% of pairs)  → ~30% of sample  (overrepresented tail)
      mid  (65 classes, ~42% of pairs)  → ~35% of sample
      tail (49 classes, ~15% of pairs)  → ~35% of sample  (overrepresented tail)

    We deliberately oversample tail classes because:
      1. Our pathway retrieval hypothesis is strongest for rare classes
      2. With proportional sampling, tail would only get ~600 pairs — too few
         to measure per-class differences reliably
    """
    rng = np.random.RandomState(seed)

    tier_allocations = {"head": 1200, "mid": 1400, "tail": 1400}
    parts = []

    for tier, n_tier in tier_allocations.items():
        tier_labels = [l for l, t in tier_map.items() if t == tier]
        tier_df = train_df[train_df["label"].isin(tier_labels)]

        # Sample proportionally from each class within the tier
        n_per_class = max(1, n_tier // len(tier_labels))
        for label in tier_labels:
            class_df = tier_df[tier_df["label"] == label]
            n_sample = min(n_per_class, len(class_df))
            if n_sample > 0:
                parts.append(class_df.sample(n=n_sample, random_state=rng))

    sampled = pd.concat(parts, ignore_index=True)

    # Trim or top-up to exactly n_total
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=rng)
    elif len(sampled) < n_total:
        # Top up from remaining pairs not already sampled
        remaining = train_df[~train_df.index.isin(sampled.index)]
        extra = remaining.sample(
            n=min(n_total - len(sampled), len(remaining)),
            random_state=rng,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    print(f"Sampled {len(sampled):,} pairs")
    tier_counts = sampled["label"].map(tier_map).value_counts()
    for tier in ("head", "mid", "tail"):
        print(f"  {tier}: {tier_counts.get(tier, 0):,} pairs")

    return sampled


# ── Retrieval precomputation ───────────────────────────────────────────────────

def get_tanimoto_retrievals(
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cfg: dict,
    logger,
) -> dict:
    """Load precomputed Tanimoto retrievals for sampled pairs.

    Uses Mohammadreza's precomputed retrieved_examples_train.json directly
    rather than recomputing from scratch. This is the baseline condition.
    Keys in the JSON are strings so we look up by str(idx).
    """
    proc = cfg["data"]["processed_dir"]
    retr_path = os.path.join(proc, "retrieved_examples_train.json")

    if not os.path.exists(retr_path):
        logger.error(f"Tanimoto retrievals not found at {retr_path}")
        return {}

    logger.info(f"Loading precomputed Tanimoto retrievals from {retr_path}...")
    with open(retr_path) as f:
        all_retrievals = json.load(f)

    # Filter to only the sampled pairs — keys are strings
    sampled_keys = {str(idx) for idx in sampled_df.index}
    retrievals = {k: v for k, v in all_retrievals.items() if k in sampled_keys}
    logger.info(f"  Loaded {len(retrievals):,} retrievals for {len(sampled_df):,} sampled pairs")
    return retrievals


def get_pathway_retrievals(
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    profiles: dict,
    cfg: dict,
    logger,
) -> dict:
    """Compute pathway retrievals for sampled pairs."""
    logger.info("Computing pathway retrievals for sampled pairs...")
    retrievals = compute_pathway_retrievals(
        split_df=sampled_df,
        candidate_df=train_df,
        profiles=profiles,
        top_k=cfg.get("retrieval", {}).get("top_k", 5),
        min_diverse_classes=cfg.get("retrieval", {}).get("min_diverse_classes", 2),
        seed=cfg["project"]["seed"],
    )
    return retrievals


# ── Trace generation ───────────────────────────────────────────────────────────

def generate_traces_for_condition(
    condition_name: str,
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    retrievals: dict,
    label_map: dict,
    profiles: dict,
    cfg: dict,
    logger,
    use_rj_prompts: bool = False,
) -> str:
    """
    Generate teacher traces for one condition (Tanimoto or Pathway).

    Returns path to the output JSONL file.
    Fully checkpointed — safe to restart.
    """
    from vllm import LLM, SamplingParams

    if use_rj_prompts:
        from src.data_preparation_rj import TEACHER_SYSTEM_PROMPT, build_teacher_prompt
        logger.info(f"  Using RJ prompts (PK/PD flag + prodrug warnings)")
    else:
        from src.data_preparation import TEACHER_SYSTEM_PROMPT, build_teacher_prompt
        logger.info(f"  Using original prompts")

    out_dir = Path(cfg["project"]["output_dir"]) / "teacher_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_file = str(out_dir / f"{condition_name}_traces.jsonl")

    # Load checkpoint — find already-generated indices
    done_indices = set()
    if os.path.exists(trace_file):
        with open(trace_file) as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["orig_idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"  Resuming: {len(done_indices):,} traces already done")

    remaining = sampled_df[~sampled_df.index.isin(done_indices)]
    logger.info(f"  Total: {len(sampled_df):,} | Remaining: {len(remaining):,}")

    if len(remaining) == 0:
        logger.info(f"  All traces done for {condition_name}")
        return trace_file

    tcfg = cfg["teacher"]
    logger.info(f"  Loading {tcfg['model_name']}...")

    llm = LLM(
        model=tcfg["model_name"],
        tensor_parallel_size=tcfg["tensor_parallel_size"],
        dtype=tcfg["dtype"],
        max_model_len=tcfg.get("max_model_len", 8192),
        gpu_memory_utilization=tcfg.get("gpu_memory_utilization", 0.92),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=tcfg.get("temperature", 0.6),
        top_p=tcfg.get("top_p", 0.95),
        max_tokens=tcfg.get("max_new_tokens", 1536),
    )

    batch_size = tcfg.get("batch_size", 32)

    with open(trace_file, "a") as fout:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining.iloc[batch_start:batch_start + batch_size]

            prompts = []
            batch_meta = []
            for orig_idx, row in batch.iterrows():
                # Get retrieved examples for this pair
                # Retrieve examples — handle two formats:
                # Tanimoto: str key -> list of example dicts (already formatted)
                # Pathway:  int key -> list of int row indices into train_df
                raw = retrievals.get(str(orig_idx),
                        retrievals.get(int(orig_idx), []))
                retr_examples = []
                for item in raw[:5]:
                    try:
                        if isinstance(item, dict):
                            # Tanimoto format - already a dict
                            retr_examples.append(item)
                        else:
                            # Pathway format - integer row index
                            ex_row = train_df.loc[int(item)]
                            retr_examples.append({
                                "drug1_id":   ex_row["drug1_id"],
                                "drug2_id":   ex_row["drug2_id"],
                                "drug1_name": str(ex_row.get("drug1_name", "")),
                                "drug2_name": str(ex_row.get("drug2_name", "")),
                                "label":      int(ex_row["label"]),
                                "label_text": str(ex_row.get("label_text", "")),
                                "severity":   str(ex_row.get("severity", "Unknown")),
                            })
                    except (KeyError, IndexError):
                        continue

                user_msg = build_teacher_prompt(row, label_map, profiles, retr_examples)
                messages = [
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ]
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
                batch_meta.append({
                    "orig_idx":   int(orig_idx),
                    "drug1_id":   row["drug1_id"],
                    "drug2_id":   row["drug2_id"],
                    "drug1_name": str(row.get("drug1_name", "")),
                    "drug2_name": str(row.get("drug2_name", "")),
                    "label":      int(row["label"]),
                    "label_text": str(row.get("label_text", "")),
                    "severity":   str(row.get("severity", "Unknown")),
                    "tier":       row.get("frequency_tier", "mid"),
                })

            outputs = llm.generate(prompts, params)

            for meta, out in zip(batch_meta, outputs):
                text = out.outputs[0].text.strip()
                rec = {**meta, "condition": condition_name, "teacher_cot": text}
                fout.write(json.dumps(rec) + "\n")

            done_so_far = min(batch_start + batch_size, len(remaining))
            logger.info(
                f"  {condition_name}: {done_so_far:,}/{len(remaining):,} "
                f"({100*done_so_far/len(remaining):.1f}%)"
            )

    # Clean up GPU memory before next condition
    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"  Traces saved: {trace_file}")
    return trace_file


# ── Grounded factuality scoring ────────────────────────────────────────────────

def score_traces(
    trace_file: str,
    profiles: dict,
    precision_weight: float = 0.7,
    logger=None,
) -> list[dict]:
    """Score all traces with grounded factuality. Returns list of scored records."""
    scored = []
    with open(trace_file) as f:
        lines = f.readlines()

    if logger:
        logger.info(f"Scoring {len(lines):,} traces from {trace_file}...")

    for line in lines:
        rec = json.loads(line)
        p1 = profiles.get(rec["drug1_id"], {})
        p2 = profiles.get(rec["drug2_id"], {})
        try:
            gf = score_trace(rec["teacher_cot"], p1, p2,
                             precision_weight=precision_weight)
        except Exception as e:
            gf = {"grounded_score": 0.0, "entity_precision": 0.0,
                  "entity_recall": 0.0, "error": str(e)}
        scored.append({**rec, **gf})

    return scored


# ── Comparison report ──────────────────────────────────────────────────────────

def build_report(
    tanimoto_scored: list[dict],
    pathway_scored: list[dict],
    logger,
) -> dict:
    """Build comparison report with per-tier breakdown."""

    def aggregate(records):
        tiers = defaultdict(list)
        all_scores = []
        all_precision = []
        for r in records:
            gs = r.get("grounded_score", 0.0)
            ep = r.get("entity_precision", 0.0)
            tier = r.get("tier", "mid")
            tiers[tier].append(gs)
            all_scores.append(gs)
            all_precision.append(ep)
        return {
            "overall_mean":    round(float(np.mean(all_scores)), 4),
            "overall_std":     round(float(np.std(all_scores)), 4),
            "precision_mean":  round(float(np.mean(all_precision)), 4),
            "n":               len(all_scores),
            "by_tier": {
                tier: {
                    "mean": round(float(np.mean(scores)), 4),
                    "std":  round(float(np.std(scores)), 4),
                    "n":    len(scores),
                }
                for tier, scores in tiers.items()
            },
        }

    tan_stats  = aggregate(tanimoto_scored)
    path_stats = aggregate(pathway_scored)

    # Align by orig_idx for paired comparison
    tan_by_idx  = {r["orig_idx"]: r.get("grounded_score", 0.0) for r in tanimoto_scored}
    path_by_idx = {r["orig_idx"]: r.get("grounded_score", 0.0) for r in pathway_scored}
    common = set(tan_by_idx.keys()) & set(path_by_idx.keys())

    deltas = [path_by_idx[i] - tan_by_idx[i] for i in common]
    pathway_better = sum(1 for d in deltas if d > 0.01)
    tanimoto_better = sum(1 for d in deltas if d < -0.01)
    tied = len(deltas) - pathway_better - tanimoto_better

    mean_delta = float(np.mean(deltas)) if deltas else 0.0

    report = {
        "n_pairs_compared": len(common),
        "tanimoto": tan_stats,
        "pathway":  path_stats,
        "delta_pathway_minus_tanimoto": {
            "mean":   round(mean_delta, 4),
            "median": round(float(np.median(deltas)), 4) if deltas else 0.0,
            "pathway_better_pct":  round(100 * pathway_better  / len(common), 1),
            "tanimoto_better_pct": round(100 * tanimoto_better / len(common), 1),
            "tied_pct":            round(100 * tied            / len(common), 1),
        },
        "verdict": (
            "PATHWAY BETTER"   if mean_delta >  0.01 else
            "TANIMOTO BETTER"  if mean_delta < -0.01 else
            "NO SIGNIFICANT DIFFERENCE"
        ),
    }

    logger.info("\n" + "=" * 60)
    logger.info("PILOT COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Pairs compared: {len(common):,}")
    logger.info(f"Tanimoto  — grounded_score mean: {tan_stats['overall_mean']:.4f}")
    logger.info(f"Pathway   — grounded_score mean: {path_stats['overall_mean']:.4f}")
    logger.info(f"Delta (pathway - tanimoto): {mean_delta:+.4f}")
    logger.info(f"Pathway better:  {pathway_better:,} ({100*pathway_better/len(common):.1f}%)")
    logger.info(f"Tanimoto better: {tanimoto_better:,} ({100*tanimoto_better/len(common):.1f}%)")
    logger.info(f"\nBy tier:")
    for tier in ("head", "mid", "tail"):
        t = tan_stats["by_tier"].get(tier, {})
        p = path_stats["by_tier"].get(tier, {})
        if t and p:
            delta = p["mean"] - t["mean"]
            logger.info(
                f"  {tier:4s}: Tan={t['mean']:.4f}  Path={p['mean']:.4f}  "
                f"Δ={delta:+.4f}  (n={t['n']})"
            )
    logger.info(f"\nVERDICT: {report['verdict']}")
    logger.info("=" * 60)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Subset pilot: Tanimoto vs Pathway RAG")
    parser.add_argument("--config", default="configs/config_pilot.yaml")
    parser.add_argument("--n-pairs", type=int, default=4000)
    parser.add_argument("--score-only", action="store_true",
                        help="Skip generation, just score existing traces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = setup_logging("subset_pilot",
                           log_dir=os.path.join(cfg["project"]["output_dir"], "logs"))
    set_seed(args.seed)

    proc = cfg["data"]["processed_dir"]

    logger.info("=" * 60)
    logger.info("Subset Pilot: Tanimoto vs Pathway RAG")
    logger.info(f"N pairs: {args.n_pairs:,}")
    logger.info("=" * 60)

    # Load shared data
    logger.info("Loading data...")
    train_df = pd.read_json(os.path.join(proc, "train.jsonl"), lines=True)
    with open(os.path.join(proc, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(proc, "drug_profiles.json")) as f:
        profiles = json.load(f)
    with open(os.path.join(proc, "tier_map.json")) as f:
        tier_map = {int(k): v for k, v in json.load(f).items()}

    logger.info(f"  Train: {len(train_df):,} pairs | Profiles: {len(profiles):,}")

    # Sample pairs
    logger.info(f"\nSampling {args.n_pairs:,} pairs (stratified by tier)...")
    sampled_df = sample_pairs_stratified(train_df, tier_map,
                                         n_total=args.n_pairs, seed=args.seed)

    # Add tier label to sampled pairs for reporting
    sampled_df["frequency_tier"] = sampled_df["label"].map(tier_map)

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the sample once so both conditions use identical pairs
    sample_path = out_dir / "sampled_pairs.jsonl"
    if not sample_path.exists():
        sampled_df.to_json(sample_path, orient="records", lines=True)
        logger.info(f"  Sample saved: {sample_path}")
    else:
        # Reload the same sample for reproducibility
        sampled_df = pd.read_json(sample_path, lines=True)
        logger.info(f"  Loaded existing sample: {sample_path}")

    if not args.score_only:
        # ── Condition 1: Tanimoto retrieval ───────────────────────────────────
        logger.info("\n--- Condition 1: Tanimoto Retrieval ---")
        tan_retrievals = get_tanimoto_retrievals(sampled_df, train_df, cfg, logger)
        tan_trace_file = generate_traces_for_condition(
            condition_name="tanimoto",
            sampled_df=sampled_df,
            train_df=train_df,
            retrievals=tan_retrievals,
            label_map=label_map,
            profiles=profiles,
            cfg=cfg,
            logger=logger,
            use_rj_prompts=False,
        )

        # ── Condition 2: Pathway retrieval ────────────────────────────────────
        logger.info("\n--- Condition 2: Pathway Retrieval ---")
        path_retrievals = get_pathway_retrievals(
            sampled_df, train_df, profiles, cfg, logger
        )
        path_trace_file = generate_traces_for_condition(
            condition_name="pathway",
            sampled_df=sampled_df,
            train_df=train_df,
            retrievals=path_retrievals,
            label_map=label_map,
            profiles=profiles,
            cfg=cfg,
            logger=logger,
            use_rj_prompts=True,   # RJ prompts for pathway condition
        )
    else:
        tan_trace_file  = str(out_dir / "teacher_traces" / "tanimoto_traces.jsonl")
        path_trace_file = str(out_dir / "teacher_traces" / "pathway_traces.jsonl")
        logger.info("Score-only mode — skipping generation")

    # ── Score both conditions ─────────────────────────────────────────────────
    precision_weight = cfg.get("grounded_eval", {}).get("precision_weight", 0.7)

    logger.info("\n--- Scoring traces with grounded factuality ---")
    tanimoto_scored = score_traces(tan_trace_file,  profiles, precision_weight, logger)
    pathway_scored  = score_traces(path_trace_file, profiles, precision_weight, logger)

    # ── Build report ──────────────────────────────────────────────────────────
    report = build_report(tanimoto_scored, pathway_scored, logger)

    report_json = out_dir / "comparison_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved: {report_json}")

    # Human-readable summary
    summary = [
        "=" * 60,
        "SUBSET PILOT — TANIMOTO vs PATHWAY RAG",
        f"Teacher: {cfg['teacher']['model_name']}",
        f"N pairs: {report['n_pairs_compared']:,}",
        "=" * 60,
        "",
        f"Tanimoto grounded score (mean): {report['tanimoto']['overall_mean']:.4f}",
        f"Pathway  grounded score (mean): {report['pathway']['overall_mean']:.4f}",
        f"Delta (pathway − tanimoto):     {report['delta_pathway_minus_tanimoto']['mean']:+.4f}",
        "",
        "By tier:",
    ]
    for tier in ("head", "mid", "tail"):
        t = report["tanimoto"]["by_tier"].get(tier, {})
        p = report["pathway"]["by_tier"].get(tier, {})
        if t and p:
            summary.append(
                f"  {tier:4s}  Tan={t['mean']:.4f}  Path={p['mean']:.4f}  "
                f"Δ={p['mean']-t['mean']:+.4f}  (n={t['n']})"
            )
    summary += [
        "",
        f"Pathway better on:  {report['delta_pathway_minus_tanimoto']['pathway_better_pct']:.1f}% of pairs",
        f"Tanimoto better on: {report['delta_pathway_minus_tanimoto']['tanimoto_better_pct']:.1f}% of pairs",
        "",
        f"VERDICT: {report['verdict']}",
        "=" * 60,
    ]
    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    report_txt = out_dir / "comparison_report.txt"
    with open(report_txt, "w") as f:
        f.write(summary_text)
    logger.info(f"Summary saved: {report_txt}")


if __name__ == "__main__":
    main()
