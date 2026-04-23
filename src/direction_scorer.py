"""
direction_scorer.py

Direction-aware evaluation of teacher traces for drug-drug interaction CoT distillation.

The existing grounded_factuality scorer checks whether pharmacological entity names
(CYP enzymes, transporters, targets) appear in the trace. It cannot check whether
the trace correctly states the DIRECTION of the interaction effect.

This scorer checks:
  1. Does the trace correctly state whether Drug X levels INCREASE or DECREASE?
  2. For prodrug pairs specifically: does the trace reason about ACTIVATION
     (not elimination) and get the direction right?
  3. Is the trace conclusive (commits to a direction) or ambiguous/unclear?

APPROACH:
  - Extract ground truth direction from the interaction label text
    (always "can be increased" or "can be decreased" in DrugBank)
  - Extract the ## Summary section from the trace
    (shorter, more decisive than full reasoning)
  - Check whether the subject (serum concentration / metabolism / excretion /
    absorption) moves in the direction stated in the label
  - Use subject + direction context window matching, not bare regex
  - No LLM calls — purely rule-based

RETURNS per trace:
  direction_result:  correct / wrong / ambiguous / not_stated / unknown
  label_direction:   increase / decrease / unknown
  trace_direction:   increase / decrease / ambiguous / not_stated / unknown
  subject:           serum_concentration / metabolism / excretion / absorption / other
  is_prodrug_pair:   True / False
  prodrug_mention:   True / False (did trace mention prodrug/activation concepts?)

USAGE:
  from src.direction_scorer import score_direction, score_trace_file

  # Score a single trace
  result = score_direction(label_text, teacher_cot, drug1_id, drug2_id, prodrug_ids)

  # Score all traces in a JSONL file
  results = score_trace_file("outputs/teacher_traces/tanimoto_traces.jsonl", prodrug_ids)
"""

import re
import json
from pathlib import Path
from collections import Counter


# ── Subject extraction ─────────────────────────────────────────────────────────

SUBJECT_PATTERNS = [
    ("serum_concentration", [
        "serum concentration",
        "plasma concentration",
        "plasma level",
        "drug level",
        "blood level",
        "drug concentration",
        "serum level",
    ]),
    ("metabolism", [
        "metabolism",
        "metabolic rate",
        "metabolized",
        "metabolised",
        "breakdown",
        "metabolic clearance",
    ]),
    ("excretion", [
        "excretion",
        "excretion rate",
        "excreted",
        "renal clearance",
        "renal elimination",
        "urinary excretion",
    ]),
    ("absorption", [
        "absorption",
        "absorbed",
        "bioavailability",
        "gastrointestinal absorption",
        "oral absorption",
    ]),
    ("activity", [
        "activity",
        "pharmacological activity",
        "therapeutic activity",
        "enzyme activity",
    ]),
]

INCREASE_WORDS = [
    "increas", "elevat", "higher", "accumulate", "greater", "rise",
    "augment", "amplif", "enhance", "potentiat", "build up",
]

DECREASE_WORDS = [
    "decreas", "lower", "reduc", "diminish", "less", "fall",
    "diminut", "attenu", "impair", "deplet", "drop",
]

PRODRUG_WORDS = [
    "prodrug", "activat", "bioactivat", "active metabolite",
    "active form", "converted to", "conversion to", "inactive until",
]


# ── Label parsing ──────────────────────────────────────────────────────────────

def extract_label_direction(label_text: str) -> tuple[str, str]:
    """
    Extract the subject and direction from a DrugBank interaction label.

    Returns (subject, direction) where:
      subject:   serum_concentration / metabolism / excretion / absorption /
                 activity / other
      direction: increase / decrease / unknown

    Examples:
      "The serum concentration of Drug X can be increased" → (serum_concentration, increase)
      "The metabolism of Drug X can be decreased"          → (metabolism, decrease)
      "The excretion rate of Drug X can be increased"      → (excretion, increase)
    """
    label_lower = label_text.lower()

    # Extract subject
    subject = "other"
    for subj_name, patterns in SUBJECT_PATTERNS:
        if any(p in label_lower for p in patterns):
            subject = subj_name
            break

    # Extract direction
    direction = "unknown"
    if "can be increased" in label_lower or "may be increased" in label_lower:
        direction = "increase"
    elif "can be decreased" in label_lower or "may be decreased" in label_lower:
        direction = "decrease"
    elif "can increase" in label_lower or "may increase" in label_lower:
        direction = "increase"
    elif "can decrease" in label_lower or "may decrease" in label_lower:
        direction = "decrease"

    return subject, direction


# ── Summary extraction ─────────────────────────────────────────────────────────

def extract_summary_section(trace: str) -> str:
    """
    Extract the ## Summary section from a structured teacher trace.

    The summary is typically 2-3 sentences and states the conclusion
    more clearly than the full reasoning section. It's the best place
    to check the stated direction of effect.
    """
    match = re.search(
        r"##\s*Summary\s*(.*?)(?:##|\Z)",
        trace,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: use last 400 chars of trace
    return trace[-400:].strip()


# ── Direction detection in text ────────────────────────────────────────────────

def detect_direction_near_subject(
    text: str,
    subject: str,
    window: int = 200,
) -> str:
    """
    Detect whether the subject moves up or down in the given text.

    Looks for direction words within `window` characters of each
    mention of the subject (or its proxy terms).

    Returns: increase / decrease / ambiguous / not_stated
    """
    text_lower = text.lower()

    # Collect all subject mention positions
    subject_positions = []
    for subj_name, patterns in SUBJECT_PATTERNS:
        if subj_name == subject or subject == "other":
            for pattern in patterns:
                for m in re.finditer(re.escape(pattern), text_lower):
                    subject_positions.append(m.start())

    if not subject_positions:
        # No subject mention — check entire text for direction words
        # as a last resort (less reliable)
        has_increase = any(w in text_lower for w in INCREASE_WORDS)
        has_decrease = any(w in text_lower for w in DECREASE_WORDS)
        if has_increase and not has_decrease:
            return "increase"
        elif has_decrease and not has_increase:
            return "decrease"
        return "not_stated"

    # For each subject mention, check nearby direction words
    found_increase = False
    found_decrease = False

    for pos in subject_positions:
        context = text_lower[max(0, pos - window):pos + window]
        if any(w in context for w in INCREASE_WORDS):
            found_increase = True
        if any(w in context for w in DECREASE_WORDS):
            found_decrease = True

    if found_increase and not found_decrease:
        return "increase"
    elif found_decrease and not found_increase:
        return "decrease"
    elif found_increase and found_decrease:
        return "ambiguous"
    else:
        return "not_stated"


# ── Prodrug detection ──────────────────────────────────────────────────────────

def check_prodrug_mention(trace: str) -> bool:
    """
    Check whether the trace mentions prodrug/activation concepts.
    Used to assess whether the teacher is aware of the prodrug context.
    """
    trace_lower = trace.lower()
    return any(w in trace_lower for w in PRODRUG_WORDS)


# ── Main scoring function ──────────────────────────────────────────────────────

def score_direction(
    label_text: str,
    teacher_cot: str,
    drug1_id: str,
    drug2_id: str,
    prodrug_ids: set,
) -> dict:
    """
    Score a single teacher trace for directional correctness.

    Args:
        label_text:   The DrugBank interaction label (ground truth)
        teacher_cot:  The full teacher trace text
        drug1_id:     DrugBank ID of drug 1
        drug2_id:     DrugBank ID of drug 2
        prodrug_ids:  Set of DrugBank IDs known to be prodrugs

    Returns dict with keys:
        direction_result:  correct / wrong / ambiguous / not_stated / unknown
        label_direction:   increase / decrease / unknown
        trace_direction:   increase / decrease / ambiguous / not_stated
        subject:           serum_concentration / metabolism / etc.
        is_prodrug_pair:   bool
        prodrug_mention:   bool
    """
    # Extract ground truth from label
    subject, label_direction = extract_label_direction(label_text)

    # Can't evaluate if label direction is unknown
    if label_direction == "unknown":
        return {
            "direction_result": "unknown",
            "label_direction": "unknown",
            "trace_direction": "unknown",
            "subject": subject,
            "is_prodrug_pair": drug1_id in prodrug_ids or drug2_id in prodrug_ids,
            "prodrug_mention": check_prodrug_mention(teacher_cot),
        }

    # Extract summary section for direction checking
    summary = extract_summary_section(teacher_cot)

    # Detect direction in summary
    trace_direction = detect_direction_near_subject(summary, subject)

    # If summary is unclear, try the full trace
    if trace_direction in ("ambiguous", "not_stated"):
        full_direction = detect_direction_near_subject(teacher_cot, subject)
        if full_direction in ("increase", "decrease"):
            trace_direction = full_direction

    # Compare
    if trace_direction == label_direction:
        result = "correct"
    elif trace_direction == "ambiguous":
        result = "ambiguous"
    elif trace_direction == "not_stated":
        result = "not_stated"
    else:
        result = "wrong"

    return {
        "direction_result": result,
        "label_direction": label_direction,
        "trace_direction": trace_direction,
        "subject": subject,
        "is_prodrug_pair": drug1_id in prodrug_ids or drug2_id in prodrug_ids,
        "prodrug_mention": check_prodrug_mention(teacher_cot),
    }


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_trace_file(
    trace_file: str,
    prodrug_ids: set,
    output_file: str = None,
) -> list[dict]:
    """
    Score all traces in a JSONL file for directional correctness.

    Args:
        trace_file:   Path to JSONL trace file
        prodrug_ids:  Set of DrugBank IDs known to be prodrugs
        output_file:  Optional path to save scored results as JSONL

    Returns list of scored record dicts (original record + direction scores).
    """
    scored = []

    with open(trace_file) as f:
        lines = f.readlines()

    for line in lines:
        rec = json.loads(line)
        direction_scores = score_direction(
            label_text=rec.get("label_text", ""),
            teacher_cot=rec.get("teacher_cot", ""),
            drug1_id=rec.get("drug1_id", ""),
            drug2_id=rec.get("drug2_id", ""),
            prodrug_ids=prodrug_ids,
        )
        scored.append({**rec, **direction_scores})

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for rec in scored:
                f.write(json.dumps(rec) + "\n")

    return scored


# ── Summary reporting ──────────────────────────────────────────────────────────

def print_direction_report(
    tanimoto_scored: list[dict],
    pathway_scored: list[dict],
    prodrug_ids: set,
):
    """Print a comparison report of direction accuracy between two conditions."""

    def aggregate(records, label):
        total = len(records)
        results = Counter(r["direction_result"] for r in records)
        prodrug = [r for r in records if r["is_prodrug_pair"]]
        prodrug_results = Counter(r["direction_result"] for r in prodrug)

        print(f"\n{label} ({total:,} traces):")
        print(f"  Overall:")
        print(f"    Correct:    {results['correct']:,} ({100*results['correct']/total:.1f}%)")
        print(f"    Wrong:      {results['wrong']:,} ({100*results['wrong']/total:.1f}%)")
        print(f"    Ambiguous:  {results['ambiguous']:,} ({100*results['ambiguous']/total:.1f}%)")
        print(f"    Not stated: {results['not_stated']:,} ({100*results['not_stated']/total:.1f}%)")
        print(f"    Unknown:    {results['unknown']:,} ({100*results['unknown']/total:.1f}%)")

        if prodrug:
            n = len(prodrug)
            print(f"\n  Prodrug pairs ({n:,} pairs, {100*n/total:.1f}% of total):")
            print(f"    Correct:    {prodrug_results['correct']:,} ({100*prodrug_results['correct']/n:.1f}%)")
            print(f"    Wrong:      {prodrug_results['wrong']:,} ({100*prodrug_results['wrong']/n:.1f}%)")
            print(f"    Ambiguous:  {prodrug_results['ambiguous']:,} ({100*prodrug_results['ambiguous']/n:.1f}%)")
            print(f"    Not stated: {prodrug_results['not_stated']:,} ({100*prodrug_results['not_stated']/n:.1f}%)")
            prod_mention = sum(1 for r in prodrug if r["prodrug_mention"])
            print(f"    Mentions prodrug/activation: {prod_mention:,}/{n:,} ({100*prod_mention/n:.1f}%)")

        # By subject
        print(f"\n  By interaction subject:")
        subject_counts = Counter(r["subject"] for r in records)
        for subj, count in subject_counts.most_common():
            subj_records = [r for r in records if r["subject"] == subj]
            correct = sum(1 for r in subj_records if r["direction_result"] == "correct")
            wrong = sum(1 for r in subj_records if r["direction_result"] == "wrong")
            print(f"    {subj:<25} n={count:,}  correct={correct:,} ({100*correct/count:.0f}%)  wrong={wrong:,} ({100*wrong/count:.0f}%)")

    print("=" * 70)
    print("DIRECTION-AWARE EVALUATION REPORT")
    print("=" * 70)

    aggregate(tanimoto_scored, "TANIMOTO + original prompts")
    aggregate(pathway_scored, "PATHWAY + RJ prompts")

    # Head-to-head comparison on matched pairs
    tan_by_idx = {r["orig_idx"]: r for r in tanimoto_scored}
    path_by_idx = {r["orig_idx"]: r for r in pathway_scored}
    common = set(tan_by_idx.keys()) & set(path_by_idx.keys())

    both_correct = sum(
        1 for idx in common
        if tan_by_idx[idx]["direction_result"] == "correct"
        and path_by_idx[idx]["direction_result"] == "correct"
    )
    tan_only = sum(
        1 for idx in common
        if tan_by_idx[idx]["direction_result"] == "correct"
        and path_by_idx[idx]["direction_result"] != "correct"
    )
    path_only = sum(
        1 for idx in common
        if path_by_idx[idx]["direction_result"] == "correct"
        and tan_by_idx[idx]["direction_result"] != "correct"
    )
    neither = len(common) - both_correct - tan_only - path_only

    print(f"\n{'='*70}")
    print(f"HEAD-TO-HEAD ON {len(common):,} MATCHED PAIRS:")
    print(f"  Both correct:       {both_correct:,} ({100*both_correct/len(common):.1f}%)")
    print(f"  Tanimoto only:      {tan_only:,} ({100*tan_only/len(common):.1f}%)")
    print(f"  Pathway only:       {path_only:,} ({100*path_only/len(common):.1f}%)")
    print(f"  Neither correct:    {neither:,} ({100*neither/len(common):.1f}%)")
    print("=" * 70)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Direction-aware trace scorer")
    parser.add_argument("--tanimoto", required=True, help="Path to tanimoto traces JSONL")
    parser.add_argument("--pathway", required=True, help="Path to pathway traces JSONL")
    parser.add_argument("--prodrug-ids", default="data/processed/dataset_A/prodrug_ids.json")
    parser.add_argument("--output-dir", default="outputs/rj_subset_exp1")
    args = parser.parse_args()

    with open(args.prodrug_ids) as f:
        prodrug_ids = set(json.load(f).keys())

    print("Scoring Tanimoto traces...")
    tan_scored = score_trace_file(
        args.tanimoto, prodrug_ids,
        output_file=f"{args.output_dir}/tanimoto_traces_direction_scored.jsonl"
    )

    print("Scoring Pathway traces...")
    path_scored = score_trace_file(
        args.pathway, prodrug_ids,
        output_file=f"{args.output_dir}/pathway_traces_direction_scored.jsonl"
    )

    print_direction_report(tan_scored, path_scored, prodrug_ids)

    # Save summary report
    report_path = f"{args.output_dir}/direction_report.txt"
    import sys
    with open(report_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print_direction_report(tan_scored, path_scored, prodrug_ids)
        sys.stdout = old_stdout
    print(f"\nReport saved to {report_path}")
