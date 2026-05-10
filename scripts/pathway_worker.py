import json, sys, time, os
import pandas as pd
import numpy as np

TASK_ID     = int(os.environ["SLURM_ARRAY_TASK_ID"])
N_TASKS     = 20
OUT_DIR     = "/scratch/rjafri/rj_subset_exp1/data/processed/dataset_A/pathway_chunks"
CHUNK_FILE  = f"{OUT_DIR}/chunk_{TASK_ID:02d}.json"
TOP_K       = 5
MIN_DIVERSE = 2
PATH_THRESH = 0.1
TAN_THRESH  = 0.3

os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(CHUNK_FILE):
    with open(CHUNK_FILE) as f:
        existing = json.load(f)
    print(f"Chunk {TASK_ID} already has {len(existing)} results — skipping", flush=True)
    sys.exit(0)

print(f"[{time.strftime('%H:%M:%S')}] Task {TASK_ID}: loading data...", flush=True)
sys.path.insert(0, '/scratch/rjafri/rj_subset_exp1')

PROC = "/scratch/rjafri/rj_subset_exp1/data/processed/dataset_A"

train_df   = pd.read_json(f"{PROC}/train.jsonl", lines=True)
with open(f"{PROC}/drug_profiles.json") as f:
    profiles = json.load(f)
with open(f"{PROC}/drug_id_order.json") as f:
    drug_id_order = json.load(f)
sim_matrix  = np.load(f"{PROC}/drug_similarity_matrix.npz")["matrix"]
drug_to_idx = {did: i for i, did in enumerate(drug_id_order)}
print(f"[{time.strftime('%H:%M:%S')}] Loaded", flush=True)

from src.pathway_retrieval import build_pathway_index
idx   = build_pathway_index(profiles)
empty = {"enzymes": {}, "transporters": {}, "targets": {}}
print(f"[{time.strftime('%H:%M:%S')}] Pathway index built", flush=True)

# Pre-extract candidate data
cand_d1     = [idx.get(d, empty) for d in train_df["drug1_id"].tolist()]
cand_d2     = [idx.get(d, empty) for d in train_df["drug2_id"].tolist()]
cand_labels = train_df["label"].tolist()
cand_idx    = list(train_df.index)
cand_d1_ids = train_df["drug1_id"].tolist()
cand_d2_ids = train_df["drug2_id"].tolist()
n_cand      = len(train_df)

# Pre-build matrix index arrays for vectorised Tanimoto
cand_mi1 = np.array([drug_to_idx.get(d, -1) for d in cand_d1_ids], dtype=np.int32)
cand_mi2 = np.array([drug_to_idx.get(d, -1) for d in cand_d2_ids], dtype=np.int32)
print(f"[{time.strftime('%H:%M:%S')}] Candidates pre-extracted", flush=True)


def get_tanimoto_vector(q_d1_id, q_d2_id):
    """Vectorised: returns shape (n_cand,) max Tanimoto scores."""
    tan = np.zeros(n_cand, dtype=np.float32)
    for qd in [q_d1_id, q_d2_id]:
        qmi = drug_to_idx.get(qd, -1)
        if qmi == -1:
            continue
        row = sim_matrix[qmi]
        valid1 = cand_mi1 >= 0
        tan[valid1] = np.maximum(tan[valid1], row[cand_mi1[valid1]])
        valid2 = cand_mi2 >= 0
        tan[valid2] = np.maximum(tan[valid2], row[cand_mi2[valid2]])
    return tan


def retrieve_hybrid(q_d1_id, q_d2_id, q_label, q_orig_idx):
    """
    Tier 1: pathway AND Tanimoto both above threshold
    Tier 2: pathway only
    Tier 3: Tanimoto only
    Fill TOP_K from Tier 1 first, then 2, then 3.
    """
    q1 = idx.get(q_d1_id, empty)
    q2 = idx.get(q_d2_id, empty)

    # Vectorised Tanimoto for all candidates at once
    tan_vec = get_tanimoto_vector(q_d1_id, q_d2_id)

    tier1, tier2, tier3 = [], [], []

    for ci, (cn1, cn2, cl) in enumerate(zip(cand_d1, cand_d2, cand_labels)):
        if cand_idx[ci] == q_orig_idx:
            continue

        # Pathway score
        p = 0.0
        for field, w in [("enzymes", 3), ("transporters", 2), ("targets", 1)]:
            qk = set(q1.get(field, {})) | set(q2.get(field, {}))
            ck = set(cn1.get(field, {})) | set(cn2.get(field, {}))
            ov = len(qk & ck)
            if ov > 0:
                p += w * ov / max(len(qk | ck), 1)

        t = float(tan_vec[ci])
        has_p = p >= PATH_THRESH
        has_t = t >= TAN_THRESH

        if has_p and has_t:
            tier1.append((p + t, ci, cl))
        elif has_p:
            tier2.append((p, ci, cl))
        elif has_t:
            tier3.append((t, ci, cl))

    for tier in [tier1, tier2, tier3]:
        tier.sort(key=lambda x: -x[0])

    selected, seen = [], set()
    for tier in [tier1, tier2, tier3]:
        for _, ci, cl in tier:
            if len(selected) >= TOP_K:
                break
            if cl not in seen or len(seen) >= MIN_DIVERSE:
                selected.append(int(cand_idx[ci]))
                seen.add(cl)
        if len(selected) >= TOP_K:
            break

    # Pad if short
    if len(selected) < TOP_K:
        for tier in [tier1, tier2, tier3]:
            for _, ci, _ in tier:
                if int(cand_idx[ci]) not in selected:
                    selected.append(int(cand_idx[ci]))
                if len(selected) >= TOP_K:
                    break
            if len(selected) >= TOP_K:
                break

    return selected[:TOP_K]


# Determine this task's slice
all_indices = list(train_df.index)
chunk_size  = len(all_indices) // N_TASKS
start       = TASK_ID * chunk_size
end         = start + chunk_size if TASK_ID < N_TASKS - 1 else len(all_indices)
my_indices  = all_indices[start:end]
my_df       = train_df.loc[my_indices]
print(f"[{time.strftime('%H:%M:%S')}] Task {TASK_ID}: {len(my_indices):,} pairs ({start}-{end})", flush=True)

results = {}
t0 = time.time()

for qi, (orig_idx, row) in enumerate(my_df.iterrows()):
    results[str(orig_idx)] = retrieve_hybrid(
        row["drug1_id"], row["drug2_id"],
        row["label"], orig_idx
    )

    if (qi + 1) % 500 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (qi + 1) * (len(my_indices) - qi - 1)
        print(f"  {qi+1:,}/{len(my_indices):,} "
              f"[{elapsed/60:.0f}min elapsed, {eta/60:.0f}min remaining]",
              flush=True)

with open(CHUNK_FILE, "w") as f:
    json.dump(results, f)

print(f"[{time.strftime('%H:%M:%S')}] Task {TASK_ID} done: "
      f"{len(results):,} pairs -> {CHUNK_FILE} "
      f"({(time.time()-t0)/60:.1f}min)", flush=True)
