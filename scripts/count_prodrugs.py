"""
count_prodrugs.py

Scans DrugBank XML and drug_profiles.json to answer:
  1. How many drugs in DrugBank are flagged as prodrugs?
  2. How many of those make it into our filtered datasets?
  3. Which specific prodrugs appear in Dataset A and Dataset B?
  4. How many interaction pairs involve at least one prodrug?

DrugBank flags prodrugs in two ways:
  a. The drug's <groups> element contains "prodrug"
  b. The drug's description or mechanism text contains "prodrug"

We check both.

Run after prepare_experiment_datasets.py:
    python scripts/count_prodrugs.py

No GPU needed. Runs in ~5 minutes on login node.
"""

import json
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from collections import defaultdict

XML_PATH  = "data/raw/drugbank_full.xml"
PROC_DIR  = Path("data/processed")
NS = {"db": "http://www.drugbank.ca"}


def find_prodrugs_in_xml() -> dict:
    """
    Scan DrugBank XML and return {drugbank_id -> prodrug_info} for all prodrugs.

    Checks two signals:
      1. <groups><group>prodrug</group></groups>  <- explicit DrugBank flag
      2. Description or mechanism text contains "prodrug" <- text evidence
    """
    print("Scanning DrugBank XML for prodrugs...")
    print("(This takes ~3-4 minutes)")

    prodrugs = {}
    drug_count = 0

    for event, elem in ET.iterparse(XML_PATH, events=("end",)):
        if elem.tag != f'{{{NS["db"]}}}drug':
            elem.clear()
            continue

        pk = elem.find('db:drugbank-id[@primary="true"]', NS)
        if pk is None:
            elem.clear()
            continue

        dbid = pk.text
        name_el = elem.find("db:name", NS)
        name = name_el.text.strip() if name_el is not None and name_el.text else ""

        drug_count += 1
        if drug_count % 5000 == 0:
            print(f"  Scanned {drug_count:,} drugs, found {len(prodrugs):,} prodrugs so far...")

        # Signal 1: explicit <groups><group>prodrug</group></groups> flag
        group_flag = False
        for grp in elem.findall("db:groups/db:group", NS):
            if grp.text and grp.text.strip().lower() == "prodrug":
                group_flag = True
                break

        # Signal 2: text evidence in description or mechanism
        desc_el = elem.find("db:description", NS)
        mech_el = elem.find("db:mechanism-of-action", NS)
        desc_text = (desc_el.text or "").lower() if desc_el is not None else ""
        mech_text = (mech_el.text or "").lower() if mech_el is not None else ""
        text_flag = "prodrug" in desc_text or "prodrug" in mech_text

        if group_flag or text_flag:
            # Get the metabolising enzyme if we can find it
            # (the enzyme that converts the prodrug to its active form)
            activating_enzymes = []
            for enz in elem.findall("db:enzymes/db:enzyme", NS):
                enz_name_el = enz.find("db:name", NS)
                actions = [a.text.strip().lower()
                          for a in enz.findall("db:actions/db:action", NS)
                          if a.text]
                # For a prodrug, the activating enzyme has action "substrate"
                # (the drug is a substrate of the enzyme that activates it)
                if "substrate" in actions and enz_name_el is not None:
                    pp = enz.find("db:polypeptide", NS)
                    gene = ""
                    if pp is not None:
                        gene_el = pp.find("db:gene-name", NS)
                        if gene_el is not None and gene_el.text:
                            gene = gene_el.text.strip()
                    activating_enzymes.append(
                        gene if gene else enz_name_el.text.strip()
                    )

            prodrugs[dbid] = {
                "name": name,
                "group_flag": group_flag,
                "text_flag": text_flag,
                "activating_enzymes": activating_enzymes[:3],
            }

        elem.clear()

    print(f"\nScan complete: {drug_count:,} drugs scanned, {len(prodrugs):,} prodrugs found")
    return prodrugs


def analyze_prodrug_presence(prodrugs: dict):
    """
    Check how many prodrugs appear in our filtered datasets and
    how many interaction pairs involve at least one prodrug.
    """
    print("\n" + "="*60)
    print("PRODRUG ANALYSIS")
    print("="*60)

    # Load drug profiles to see which drugs made it into the dataset
    profiles_path = PROC_DIR / "drug_profiles.json"
    if not profiles_path.exists():
        print("drug_profiles.json not found — run extract_dataset_from_xml.py first")
        return

    with open(profiles_path) as f:
        profiles = json.load(f)

    # Which prodrugs are in the profiled drugs (i.e. have DDI interactions)?
    prodrugs_in_profiles = {
        dbid: info for dbid, info in prodrugs.items()
        if dbid in profiles
    }

    print(f"\nTotal prodrugs in DrugBank:          {len(prodrugs):,}")
    print(f"Prodrugs with DDI interactions:      {len(prodrugs_in_profiles):,}")
    print(f"  (these are in drug_profiles.json)")

    # Which prodrugs make it into Dataset A and B?
    for ds_name, ds_dir in [("Dataset A (>=130)", "dataset_A"),
                             ("Dataset B (>=20)",  "dataset_B")]:
        train_path = PROC_DIR / ds_dir / "train.jsonl"
        if not train_path.exists():
            print(f"\n{ds_name}: train.jsonl not found, skipping")
            continue

        train_df = pd.read_json(train_path, lines=True)

        # Unique drugs in this dataset
        dataset_drugs = set(train_df["drug1_id"]) | set(train_df["drug2_id"])
        prodrugs_in_dataset = {
            dbid: info for dbid, info in prodrugs_in_profiles.items()
            if dbid in dataset_drugs
        }

        # Pairs where at least one drug is a prodrug
        pairs_with_prodrug = sum(
            1 for _, row in train_df.iterrows()
            if row["drug1_id"] in prodrugs or row["drug2_id"] in prodrugs
        )

        # Pairs where the prodrug is the substrate (most pharmacologically relevant)
        # These are the pairs most likely to have direction errors in teacher traces
        pairs_prodrug_as_substrate = 0
        for _, row in train_df.iterrows():
            for drug_id in [row["drug1_id"], row["drug2_id"]]:
                if drug_id in prodrugs_in_dataset:
                    pairs_prodrug_as_substrate += 1
                    break

        print(f"\n{ds_name}:")
        print(f"  Unique drugs in dataset:           {len(dataset_drugs):,}")
        print(f"  Prodrugs in dataset:               {len(prodrugs_in_dataset):,}")
        print(f"  Pairs with >=1 prodrug:            {pairs_with_prodrug:,} "
              f"({100*pairs_with_prodrug/len(train_df):.1f}%)")

        # List the specific prodrugs found
        print(f"\n  Prodrugs found (name, activating enzymes):")
        for dbid, info in sorted(prodrugs_in_dataset.items(),
                                  key=lambda x: x[1]["name"]):
            flag_str = ""
            if info["group_flag"]:
                flag_str += "[explicit] "
            if info["text_flag"]:
                flag_str += "[text] "
            enzymes = ", ".join(info["activating_enzymes"]) if info["activating_enzymes"] else "unknown"
            print(f"    {info['name']:<30} {flag_str}enzymes: {enzymes}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
  For each prodrug in the dataset, the teacher model needs to know
  that enzyme inhibition DECREASES active drug levels (not increases).
  Without this flag, traces for prodrug pairs may have the wrong
  direction of effect — saying 'inhibition increases drug levels'
  when it should say 'inhibition decreases active drug levels'.

  These are the pairs most at risk for directional errors in traces.
  Adding is_prodrug=True to drug profiles allows:
    1. Hard rejection rule to flag traces with wrong direction
    2. Teacher prompt to explicitly warn about prodrug reversal
    3. Grounded factuality to check direction of effect
""")


def main():
    if not Path(XML_PATH).exists():
        print(f"ERROR: {XML_PATH} not found")
        print("Make sure drugbank_full.xml is in data/raw/")
        return

    prodrugs = find_prodrugs_in_xml()
    analyze_prodrug_presence(prodrugs)

    # Save prodrug list for use by other scripts
    out_path = PROC_DIR / "prodrug_ids.json"
    out_data = {
        dbid: {
            "name": info["name"],
            "activating_enzymes": info["activating_enzymes"],
            "group_flag": info["group_flag"],
        }
        for dbid, info in prodrugs.items()
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Prodrug list saved to {out_path}")
    print(f"Use this in extract_dataset_from_xml.py to add is_prodrug flag to profiles")


if __name__ == "__main__":
    main()
