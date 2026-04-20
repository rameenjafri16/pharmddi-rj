"""
severity_classifier.py

Rule-based severity classifier for drug-drug interactions.

83.8% of training pairs in the PharmCoT dataset have no DDInter severity label.
The remaining 16.2% are either teacher-generated (unreliable) or DDInter-sourced
(reliable but sparse). This means the student model is trained to predict severity
from mostly missing or hallucinated labels.

This classifier provides evidence-based severity labels using three signal sources:

  1. NTI (Narrow Therapeutic Index) drug involvement
     Drugs where small changes in plasma concentration cause toxicity or failure.
     Any interaction involving an NTI drug is at least Moderate, often Major.

  2. DrugBank drug category overlap
     DrugBank annotates drugs with pharmacological categories (e.g. "Agents causing
     hyperkalemia", "QTc Prolonging Agents", "Central Nervous System Depressants").
     Overlapping high-risk categories between two drugs strongly predicts severity.

  3. Interaction template pattern
     The label text contains direct severity signals: "risk or severity of bleeding",
     "risk or severity of QTc prolongation", "risk or severity of seizure" etc.

SEVERITY DEFINITIONS (following DDInter/clinical pharmacology conventions):
  Major:    Life-threatening or requiring major intervention. Avoid combination
            or use only with intensive monitoring.
  Moderate: Significant clinical effect. May require dose adjustment or close
            monitoring. Use with caution.
  Minor:    Mild effect, limited clinical significance. Interaction is real but
            rarely requires intervention beyond awareness.

USAGE:
  from src.severity_classifier import classify_severity, build_severity_map

  # Classify a single pair
  result = classify_severity(drug1_id, drug2_id, label_text, profiles)
  # result = {"severity": "Major", "reason": "NTI drug involved: Warfarin",
  #           "confidence": "high", "source": "rule_based"}

  # Build severity map for entire dataset
  severity_map = build_severity_map(train_df, profiles)
"""

import re
import json
from pathlib import Path


# ── Narrow Therapeutic Index drugs ────────────────────────────────────────────
# These drugs have a small margin between therapeutic and toxic doses.
# Any DDI involving an NTI drug warrants at least Moderate severity.
# Sources: FDA NTI drug list, clinical pharmacology literature.

NTI_DRUG_IDS = {
    # Anticoagulants
    "DB00682",  # Warfarin
    "DB01418",  # Acenocoumarol
    "DB00946",  # Phenprocoumon

    # Antiepileptics
    "DB00252",  # Phenytoin
    "DB01320",  # Fosphenytoin (prodrug of phenytoin)
    "DB00564",  # Carbamazepine
    "DB00313",  # Valproic acid
    "DB00532",  # Mephenytoin

    # Cardiac glycosides
    "DB00390",  # Digoxin
    "DB01404",  # Digitoxin

    # Immunosuppressants
    "DB00091",  # Cyclosporine
    "DB00864",  # Tacrolimus
    "DB00877",  # Sirolimus (Rapamycin)

    # Mood stabilisers
    "DB00697",  # Lithium (main salt)
    "DB01356",  # Lithium cation

    # Bronchodilators
    "DB00277",  # Theophylline
    "DB01588",  # Aminophylline

    # Antimetabolites
    "DB00563",  # Methotrexate

    # Aminoglycosides
    "DB00839",  # Tobramycin
    "DB00798",  # Gentamicin
    "DB01082",  # Streptomycin
    "DB00681",  # Amphotericin B

    # Glycopeptides
    "DB00512",  # Vancomycin

    # Antiarrhythmics
    "DB01256",  # Amiodarone (narrow therapeutic window)
    "DB01069",  # Procainamide (narrow therapeutic window)
    "DB00519",  # Quinidine

    # Thyroid
    "DB00451",  # Levothyroxine (NTI in some regions)
}

NTI_DRUG_NAMES = {
    "warfarin", "acenocoumarol", "phenprocoumon",
    "phenytoin", "fosphenytoin", "carbamazepine", "valproic acid",
    "digoxin", "digitoxin",
    "cyclosporine", "tacrolimus", "sirolimus", "rapamycin",
    "lithium",
    "theophylline", "aminophylline",
    "methotrexate",
    "tobramycin", "gentamicin", "streptomycin", "vancomycin",
    "amiodarone", "procainamide", "quinidine",
    "amphotericin",
}

# High-risk QTc drugs — QTc interaction involving any of these is Major
# (not just Moderate). Source: CredibleMeds highest-risk QTc list.
HIGH_RISK_QTC_NAMES = {
    "vandetanib", "sotalol", "dofetilide", "ibutilide",
    "disopyramide", "methadone", "haloperidol", "droperidol",
    "arsenic trioxide", "thioridazine", "pimozide", "cisapride",
    "moxifloxacin", "sparfloxacin", "domperidone", "eribulin",
    "ribociclib", "osimertinib", "lapatinib",
}

# Direct anticoagulants — when combined with thrombolytics → Major
DIRECT_ANTICOAGULANT_NAMES = {
    "warfarin", "heparin", "rivaroxaban", "apixaban", "dabigatran",
    "edoxaban", "fondaparinux", "enoxaparin", "tinzaparin",
    "dalteparin", "bivalirudin", "argatroban", "ibrutinib",
    "vorapaxar",
}

THROMBOLYTIC_NAMES = {
    "alteplase", "streptokinase", "urokinase", "tenecteplase",
    "reteplase", "anistreplase",
}

# Antineoplastic drugs — excretion/metabolism changes are Moderate not Minor
ANTINEOPLASTIC_NAMES = {
    "clofarabine", "methotrexate", "fluorouracil", "capecitabine",
    "gemcitabine", "cytarabine", "paclitaxel", "docetaxel",
    "doxorubicin", "vincristine", "vinblastine", "etoposide",
    "irinotecan", "topotecan", "oxaliplatin", "cisplatin",
    "carboplatin", "cyclophosphamide", "ifosfamide", "busulfan",
    "lenalidomide", "thalidomide", "bortezomib", "dasatinib",
    "imatinib", "erlotinib", "gefitinib", "sorafenib", "sunitinib",
}

# Psychiatric drugs — metabolism decrease is Major (narrow therapeutic window)
PSYCHIATRIC_DRUG_NAMES = {
    "brexpiprazole", "aripiprazole", "olanzapine", "quetiapine",
    "risperidone", "haloperidol", "clozapine", "ziprasidone",
    "lurasidone", "paliperidone", "amisulpride", "perphenazine",
    "fluphenazine", "thioridazine", "chlorpromazine",
    "amitriptyline", "nortriptyline", "imipramine", "desipramine",
    "clomipramine", "doxepin", "fluoxetine", "paroxetine",
    "sertraline", "venlafaxine", "duloxetine", "lithium",
}

# Potent CYP inducers — serum concentration decrease of any drug is Major
POTENT_CYP_INDUCERS = {
    "enzalutamide", "apalutamide", "lumacaftor", "dabrafenib",
    "mitotane", "teriflunomide", "osimertinib", "rifampicin",
    "rifampin", "rifabutin", "carbamazepine", "phenytoin",
    "phenobarbital", "primidone", "st john", "hypericum",
    "efavirenz", "nevirapine", "etravirine", "bosentan",
    "modafinil", "brigatinib",
}

# Potent CYP inhibitors — serum concentration increase of any drug is Major
POTENT_CYP_INHIBITORS = {
    "ketoconazole", "itraconazole", "voriconazole", "posaconazole",
    "fluconazole", "clarithromycin", "erythromycin", "telithromycin",
    "ritonavir", "cobicistat", "indinavir", "nelfinavir", "saquinavir",
    "lopinavir", "atazanavir", "tipranavir", "nefazodone",
    "mibefradil", "isometheptene",
}

# Biologic immunosuppressants — not always caught by category matching
BIOLOGIC_IMMUNOSUPPRESSANTS = {
    "golimumab", "adalimumab", "infliximab", "certolizumab",
    "etanercept", "rituximab", "natalizumab", "vedolizumab",
    "ustekinumab", "secukinumab", "ixekizumab", "guselkumab",
    "belimumab", "abatacept", "tocilizumab", "sarilumab",
    "anakinra", "canakinumab", "basiliximab", "daclizumab",
    "methylprednisolone", "prednisone", "prednisolone",
}


# ── High-risk drug category pairs ─────────────────────────────────────────────
# When both drugs belong to these categories, the interaction is likely Major.
# Category names from DrugBank drug profiles.

MAJOR_CATEGORY_COMBOS = [
    # QTc prolongation — additive hERG channel blocking
    ({"qtc prolonging", "qt prolonging", "torsade"},
     {"qtc prolonging", "qt prolonging", "torsade"}),

    # Serotonin syndrome
    ({"serotonergic", "serotonin"},
     {"serotonergic", "serotonin"}),

    # CNS/respiratory depression
    ({"central nervous system depressant", "cns depressant", "opioid",
      "benzodiazepine", "sedative"},
     {"central nervous system depressant", "cns depressant", "opioid",
      "benzodiazepine", "sedative", "respiratory depressant"}),

    # Bleeding risk — anticoagulant + antiplatelet
    ({"anticoagulant", "antithrombotic", "coagulation"},
     {"antiplatelet", "platelet aggregation inhibitor", "fibrinolytic"}),

    # Live vaccine + immunosuppressant
    ({"live vaccine", "live attenuated vaccine", "vaccine"},
     {"immunosuppressive", "immunosuppressant", "antineoplastic",
      "cytotoxic"}),

    # Myelosuppression — two cytotoxic agents
    ({"antineoplastic", "cytotoxic", "myelosuppressive"},
     {"antineoplastic", "cytotoxic", "myelosuppressive"}),
]

MODERATE_CATEGORY_COMBOS = [
    # Hyperkalemia risk
    ({"agents causing hyperkalemia", "potassium-sparing",
      "angiotensin", "ace inhibitor", "arb"},
     {"agents causing hyperkalemia", "potassium-sparing",
      "angiotensin", "ace inhibitor", "arb", "potassium supplement"}),

    # Hypotension risk
    ({"antihypertensive", "agents that produce hypotension",
      "vasodilator", "alpha blocker"},
     {"antihypertensive", "agents that produce hypotension",
      "vasodilator", "alpha blocker", "phosphodiesterase inhibitor"}),

    # Hypoglycemia risk
    ({"antidiabetic", "hypoglycemic", "insulin", "sulfonylurea"},
     {"antidiabetic", "hypoglycemic", "insulin", "sulfonylurea",
      "beta blocker"}),

    # Nephrotoxicity
    ({"nephrotoxic", "renal toxicity"},
     {"nephrotoxic", "renal toxicity", "nsaid", "aminoglycoside",
      "contrast media"}),
]


# ── Label text pattern rules ───────────────────────────────────────────────────
# Direct severity signals from the interaction label text.

MAJOR_LABEL_PATTERNS = [
    # Life-threatening adverse effects
    r"risk.*seizure",
    r"risk.*status epilepticus",
    r"risk.*neuroleptic malignant",
    r"risk.*torsade",
    r"risk.*cardiac arrest",
    r"risk.*anaphylax",
    r"risk.*agranulocytosis",
    r"risk.*aplastic anemia",
    r"risk.*rhabdomyolysis.*severe",
    r"risk.*fatal",
    r"risk.*life.threatening",
    r"risk.*severe bleeding",
    r"risk.*hemorrhagic stroke",
    # Dangerous activity combinations
    r"(increase|potentiate).*anticoagulant.*activities",
    r"(increase|potentiate).*antithrombotic.*activities",
    r"(increase|potentiate).*antineoplastic.*activities",
    r"(increase|potentiate).*myelosuppressive.*activities",
    r"decrease.*anticoagulant.*activities",
]

MODERATE_LABEL_PATTERNS = [
    # PK interactions — serum concentration changes
    r"serum concentration.*can be increased",
    r"serum concentration.*can be decreased",
    r"metabolism.*can be (increased|decreased)",
    # Absorption decrease (antacids, chelation, food interactions)
    r"cause a decrease in the absorption",
    r"absorption.*can be decreased",
    r"absorption.*resulting in a reduced serum",
    r"bioavailability.*can be (increased|decreased)",
    # Risk-based moderate outcomes
    r"risk.*bleeding",
    r"risk.*hyperkalemia",
    r"risk.*hypokalemia",
    r"risk.*hypotension",
    r"risk.*orthostatic hypotension",
    r"risk.*rhabdomyolysis",
    r"risk.*nephrotox",
    r"risk.*hepatotox",
    r"risk.*myelosuppression",
    r"risk.*neutropenia",
    r"risk.*thrombocytopenia",
    r"risk.*hyponatremia",
    r"risk.*hypoglycemia",
    r"risk.*electrolyte",
    r"risk.*qt",
    r"risk.*cardiac",
    r"risk.*hypersensitivity",
    r"risk.*photosensitivity",
    r"risk.*ototox",
    r"risk.*serotonin",
    r"risk.*infection",
    r"risk.*respiratory depression",
    # Activity patterns — moderate risk
    r"(increase|decrease).*hypotensive.*activities",
    r"(increase|decrease).*hypoglycemic.*activities",
    r"(increase|decrease).*bradycardic.*activities",
    r"(increase|decrease).*vasodilatory.*activities",
    r"(increase|decrease).*antihypertensive.*activities",
    r"(increase|decrease).*sedative.*activities",
    r"(increase|decrease).*neuromuscular.*activities",
    r"(increase|decrease).*hyperkalemic.*activities",
    r"(increase|decrease).*hypokalemic.*activities",
    r"(increase|decrease).*diuretic.*activities",
    r"(increase|decrease).*antiplatelet.*activities",
    r"(increase|decrease).*anticholinergic.*activities",
    r"(increase|decrease).*qtc.prolonging.*activities",
    r"(increase|decrease).*neuroexcitatory.*activities",
    r"(increase|decrease).*central nervous system depressant.*activities",
    r"(increase|decrease).*hypertensive.*activities",
    r"(increase|decrease).*vasoconstrictive.*activities",
    r"(increase|decrease).*immunosuppressive.*activities",
    r"(increase|decrease).*respiratory depressant.*activities",
    # Effectiveness decrease (diagnostic agents, vaccines)
    r"may decrease effectiveness",
    r"therapeutic efficacy.*can be decreased",
]


MINOR_LABEL_PATTERNS = [
    # Only genuinely minor PK effects
    # therapeutic efficacy and bioavailability moved to MODERATE_LABEL_PATTERNS
    r"excretion.*can be (increased|decreased)",
]


# ── Main classifier ────────────────────────────────────────────────────────────

def _get_drug_categories(drug_id: str, profiles: dict) -> set:
    """Get lowercased drug categories from DrugBank profile."""
    profile = profiles.get(drug_id, {})
    cats = profile.get("categories", [])
    return {c.lower() for c in cats}


def _get_drug_name(drug_id: str, profiles: dict) -> str:
    """Get drug name from profile."""
    return profiles.get(drug_id, {}).get("name", "").lower()


def _check_category_combo(cats1: set, cats2: set,
                           combo_rules: list) -> str | None:
    """
    Check if the two drug category sets match any combo rule.
    Returns the matching rule description or None.
    """
    for rule_cats1, rule_cats2 in combo_rules:
        match1 = cats1 & rule_cats1
        match2 = cats2 & rule_cats2
        if match1 and match2:
            return f"{list(match1)[0]} + {list(match2)[0]}"
        # Also check reversed
        match1r = cats1 & rule_cats2
        match2r = cats2 & rule_cats1
        if match1r and match2r:
            return f"{list(match1r)[0]} + {list(match2r)[0]}"
    return None


def classify_severity(
    drug1_id: str,
    drug2_id: str,
    label_text: str,
    profiles: dict,
    drug1_name: str = "",
    drug2_name: str = "",
) -> dict:
    """
    Classify the severity of a drug-drug interaction.

    Checks in priority order:
      1. NTI drug involvement → Major or Moderate
      2. High-risk category combinations → Major or Moderate
      3. Label text pattern matching → Major, Moderate, or Minor
      4. Default → Unknown

    Returns dict with:
      severity:    Major / Moderate / Minor / Unknown
      confidence:  high / medium / low
      reason:      human-readable explanation
      source:      rule_based
    """
    label_lower = label_text.lower()

    # Get drug info
    d1_name = drug1_name.lower() or _get_drug_name(drug1_id, profiles)
    d2_name = drug2_name.lower() or _get_drug_name(drug2_id, profiles)
    d1_cats = _get_drug_categories(drug1_id, profiles)
    d2_cats = _get_drug_categories(drug2_id, profiles)

    # ── Rule 1: NTI drug involvement ──────────────────────────────────────────
    d1_is_nti = (drug1_id in NTI_DRUG_IDS or
                 any(n in d1_name for n in NTI_DRUG_NAMES))
    d2_is_nti = (drug2_id in NTI_DRUG_IDS or
                 any(n in d2_name for n in NTI_DRUG_NAMES))

    if d1_is_nti or d2_is_nti:
        nti_drug = drug1_name if d1_is_nti else drug2_name
        # NTI + serum concentration change → Major
        if ("serum concentration" in label_lower or
                "metabolism" in label_lower):
            return {
                "severity": "Major",
                "confidence": "high",
                "reason": f"NTI drug {nti_drug} with PK interaction — "
                          f"small concentration changes cause toxicity/failure",
                "source": "rule_based",
            }
        # NTI + any interaction → at least Moderate
        return {
            "severity": "Moderate",
            "confidence": "high",
            "reason": f"NTI drug involved: {nti_drug}",
            "source": "rule_based",
        }

    # ── Rule 2: Major label text patterns (with drug-aware overrides) ────────
    for pattern in MAJOR_LABEL_PATTERNS:
        if re.search(pattern, label_lower):

            # QTc — only Major if one drug is a known high-risk QTc agent
            if "qt" in pattern:
                d1_hqtc = any(n in d1_name for n in HIGH_RISK_QTC_NAMES)
                d2_hqtc = any(n in d2_name for n in HIGH_RISK_QTC_NAMES)
                if not d1_hqtc and not d2_hqtc:
                    return {
                        "severity": "Moderate",
                        "confidence": "medium",
                        "reason": "QTc risk but neither drug is a high-risk QTc agent",
                        "source": "rule_based",
                    }

            # Anticoagulant activities — only Major for direct anticoag + thrombolytic
            if "anticoagulant.*activities" in pattern:
                d1_direct = any(n in d1_name for n in DIRECT_ANTICOAGULANT_NAMES)
                d2_direct = any(n in d2_name for n in DIRECT_ANTICOAGULANT_NAMES)
                d1_thrombo = any(n in d1_name for n in THROMBOLYTIC_NAMES)
                d2_thrombo = any(n in d2_name for n in THROMBOLYTIC_NAMES)
                if not ((d1_direct or d2_direct) and (d1_thrombo or d2_thrombo)):
                    return {
                        "severity": "Moderate",
                        "confidence": "medium",
                        "reason": "Anticoagulant activity increase but not "
                                  "direct anticoagulant + thrombolytic",
                        "source": "rule_based",
                    }

            return {
                "severity": "Major",
                "confidence": "high",
                "reason": f"Label indicates high-severity outcome: '{pattern}'",
                "source": "rule_based",
            }

    # ── Rule 3: Major category combinations ──────────────────────────────────
    major_match = _check_category_combo(d1_cats, d2_cats,
                                         MAJOR_CATEGORY_COMBOS)
    if major_match:
        return {
            "severity": "Major",
            "confidence": "medium",
            "reason": f"High-risk category combination: {major_match}",
            "source": "rule_based",
        }

    # ── Rule 3b: Biologic immunosuppressant involvement ───────────────────────
    # These are often missed by category matching but always Major when combined
    # with live vaccines or other immunosuppressants
    d1_biologic = any(n in d1_name for n in BIOLOGIC_IMMUNOSUPPRESSANTS)
    d2_biologic = any(n in d2_name for n in BIOLOGIC_IMMUNOSUPPRESSANTS)
    if d1_biologic or d2_biologic:
        if "infection" in label_lower or "immunosuppressive" in label_lower \
                or "vaccine" in label_lower or "myelosuppressive" in label_lower:
            bio_drug = drug1_name if d1_biologic else drug2_name
            return {
                "severity": "Major",
                "confidence": "medium",
                "reason": f"Biologic immunosuppressant {bio_drug} with "
                          f"infection/immunosuppression risk",
                "source": "rule_based",
            }

    # ── Rule 4: Moderate label text patterns (with drug-aware upgrades) ─────
    for pattern in MODERATE_LABEL_PATTERNS:
        if re.search(pattern, label_lower):

            # Infection risk + live vaccine → always Major
            # Live vaccines are definitively Major when combined with immunosuppressants
            if "infection" in pattern or "serotonin" in pattern:
                live_vaccines = {
                    "smallpox", "vaccinia", "yellow fever", "bcg",
                    "bacillus calmette", "rubella", "mumps", "measles",
                    "varicella", "rotavirus", "oral polio", "typhoid oral",
                    "adenovirus", "live attenuated",
                }
                d1_vaccine = any(v in d1_name for v in live_vaccines)
                d2_vaccine = any(v in d2_name for v in live_vaccines)
                if (d1_vaccine or d2_vaccine) and "infection" in pattern:
                    vax = drug1_name if d1_vaccine else drug2_name
                    return {
                        "severity": "Major",
                        "confidence": "high",
                        "reason": f"Live vaccine {vax} infection risk — "
                                  f"always Major with immunosuppression",
                        "source": "rule_based",
                    }
                # Serotonin syndrome + known serotonergic drug → Major
                serotonergic = {
                    "tramadol", "fentanyl", "meperidine", "pethidine",
                    "linezolid", "methylene blue", "lithium", "triptans",
                    "sumatriptan", "zolmitriptan", "rizatriptan",
                    "ssri", "snri", "maoi", "phenelzine", "tranylcypromine",
                    "selegiline", "moclobemide", "St john", "dextromethorphan",
                    "nefazodone", "trazodone", "mirtazapine",
                }
                if "serotonin" in pattern:
                    d1_sero = any(s in d1_name for s in serotonergic)
                    d2_sero = any(s in d2_name for s in serotonergic)
                    if d1_sero or d2_sero:
                        return {
                            "severity": "Major",
                            "confidence": "high",
                            "reason": "Serotonin syndrome risk with known "
                                      "serotonergic agent",
                            "source": "rule_based",
                        }

            # Immunosuppressive activities + both drugs are immunosuppressants → Major
            if "immunosuppressive" in pattern:
                all_immunosuppressants = (
                    set(BIOLOGIC_IMMUNOSUPPRESSANTS) |
                    {"cyclosporine", "tacrolimus", "sirolimus", "mycophenolate",
                     "azathioprine", "methotrexate", "leflunomide", "fingolimod",
                     "natalizumab", "dimethyl fumarate", "teriflunomide",
                     "hydroxychloroquine", "chloroquine", "sulfasalazine"}
                )
                d1_immuno = any(n in d1_name for n in all_immunosuppressants)
                d2_immuno = any(n in d2_name for n in all_immunosuppressants)
                if d1_immuno and d2_immuno:
                    return {
                        "severity": "Major",
                        "confidence": "high",
                        "reason": "Double immunosuppression — additive "
                                  "immunosuppression risk",
                        "source": "rule_based",
                    }

            # Absorption decrease + antineoplastic → Major
            if "absorption" in pattern or "cause a decrease in the absorption" in pattern:
                d1_chemo = any(n in d1_name for n in ANTINEOPLASTIC_NAMES)
                d2_chemo = any(n in d2_name for n in ANTINEOPLASTIC_NAMES)
                if d1_chemo or d2_chemo:
                    chemo_drug = drug1_name if d1_chemo else drug2_name
                    return {
                        "severity": "Major",
                        "confidence": "medium",
                        "reason": f"Absorption decrease of antineoplastic "
                                  f"{chemo_drug} — treatment failure risk",
                        "source": "rule_based",
                    }

            # QTc + high-risk QTc drug → upgrade to Major
            if "qt" in pattern:
                d1_hqtc = any(n in d1_name for n in HIGH_RISK_QTC_NAMES)
                d2_hqtc = any(n in d2_name for n in HIGH_RISK_QTC_NAMES)
                if d1_hqtc or d2_hqtc:
                    hqtc_drug = drug1_name if d1_hqtc else drug2_name
                    return {
                        "severity": "Major",
                        "confidence": "high",
                        "reason": f"QTc risk involving high-risk QTc agent "
                                  f"{hqtc_drug}",
                        "source": "rule_based",
                    }

            # Metabolism decrease + psychiatric drug → upgrade to Major
            if "metabolism" in pattern:
                d1_psych = any(n in d1_name for n in PSYCHIATRIC_DRUG_NAMES)
                d2_psych = any(n in d2_name for n in PSYCHIATRIC_DRUG_NAMES)
                if d1_psych or d2_psych:
                    psych_drug = drug1_name if d1_psych else drug2_name
                    return {
                        "severity": "Major",
                        "confidence": "medium",
                        "reason": f"Metabolism change of psychiatric drug "
                                  f"{psych_drug} — narrow therapeutic window",
                        "source": "rule_based",
                    }

            # Serum concentration decrease + potent CYP inducer → Major
            if "serum concentration.*can be decreased" in pattern or \
               "metabolism.*can be increased" in pattern:
                d1_inducer = any(n in d1_name for n in POTENT_CYP_INDUCERS)
                d2_inducer = any(n in d2_name for n in POTENT_CYP_INDUCERS)
                if d1_inducer or d2_inducer:
                    ind_drug = drug1_name if d1_inducer else drug2_name
                    return {
                        "severity": "Major",
                        "confidence": "high",
                        "reason": f"Potent CYP inducer {ind_drug} — "
                                  f"significant drug level reduction",
                        "source": "rule_based",
                    }

            # Serum concentration increase + potent CYP inhibitor → Major
            if "serum concentration.*can be increased" in pattern or \
               "metabolism.*can be decreased" in pattern:
                d1_inhibitor = any(n in d1_name for n in POTENT_CYP_INHIBITORS)
                d2_inhibitor = any(n in d2_name for n in POTENT_CYP_INHIBITORS)
                if d1_inhibitor or d2_inhibitor:
                    inh_drug = drug1_name if d1_inhibitor else drug2_name
                    return {
                        "severity": "Major",
                        "confidence": "high",
                        "reason": f"Potent CYP inhibitor {inh_drug} — "
                                  f"significant drug level increase",
                        "source": "rule_based",
                    }

            # Bleeding + direct anticoagulant OR antiplatelet → upgrade to Major
            if "bleeding" in pattern:
                d1_anticoag = any(n in d1_name for n in DIRECT_ANTICOAGULANT_NAMES)
                d2_anticoag = any(n in d2_name for n in DIRECT_ANTICOAGULANT_NAMES)
                antiplatelet = {"eptifibatide", "tirofiban", "abciximab",
                                "clopidogrel", "prasugrel", "ticagrelor",
                                "cangrelor", "vorapaxar", "dipyridamole"}
                d1_ap = any(n in d1_name for n in antiplatelet)
                d2_ap = any(n in d2_name for n in antiplatelet)
                if d1_anticoag or d2_anticoag or d1_ap or d2_ap:
                    return {
                        "severity": "Major",
                        "confidence": "high",
                        "reason": "Bleeding risk involving anticoagulant/antiplatelet",
                        "source": "rule_based",
                    }

            return {
                "severity": "Moderate",
                "confidence": "medium",
                "reason": f"Label indicates moderate-severity outcome: '{pattern}'",
                "source": "rule_based",
            }

    # ── Rule 5: Moderate category combinations ───────────────────────────────
    moderate_match = _check_category_combo(d1_cats, d2_cats,
                                            MODERATE_CATEGORY_COMBOS)
    if moderate_match:
        return {
            "severity": "Moderate",
            "confidence": "medium",
            "reason": f"Moderate-risk category combination: {moderate_match}",
            "source": "rule_based",
        }

    # ── Rule 6: Minor label text patterns (with drug-aware upgrades) ─────────
    for pattern in MINOR_LABEL_PATTERNS:
        if re.search(pattern, label_lower):

            # Excretion change + antineoplastic → upgrade to Moderate
            if "excretion.*can be" in pattern:
                d1_chemo = any(n in d1_name for n in ANTINEOPLASTIC_NAMES)
                d2_chemo = any(n in d2_name for n in ANTINEOPLASTIC_NAMES)
                if d1_chemo or d2_chemo:
                    chemo_drug = drug1_name if d1_chemo else drug2_name
                    return {
                        "severity": "Moderate",
                        "confidence": "medium",
                        "reason": f"Excretion change of antineoplastic drug "
                                  f"{chemo_drug} — toxicity risk",
                        "source": "rule_based",
                    }

            # Therapeutic efficacy decreased + immunosuppressant → upgrade to Major
            if "therapeutic efficacy.*can be decreased" in pattern:
                d1_immuno = any(c in d1_cats for c in
                                {"immunosuppressive", "antineoplastic agents",
                                 "cytotoxic"})
                d2_immuno = any(c in d2_cats for c in
                                {"immunosuppressive", "antineoplastic agents",
                                 "cytotoxic"})
                if d1_immuno or d2_immuno:
                    return {
                        "severity": "Major",
                        "confidence": "medium",
                        "reason": "Vaccine/drug efficacy decreased by "
                                  "immunosuppressant — infection/treatment risk",
                        "source": "rule_based",
                    }

            return {
                "severity": "Minor",
                "confidence": "low",
                "reason": f"Label suggests minor clinical significance: '{pattern}'",
                "source": "rule_based",
            }

    # ── Default ───────────────────────────────────────────────────────────────
    return {
        "severity": "Unknown",
        "confidence": "low",
        "reason": "No matching severity rule",
        "source": "rule_based",
    }


# ── Batch classification ───────────────────────────────────────────────────────

def build_severity_map(
    train_df,
    profiles: dict,
    existing_severity_col: str = "severity",
    existing_source_col: str = "severity_source",
    override_unknown_only: bool = True,
) -> dict:
    """
    Build a severity map for all pairs in train_df.

    If override_unknown_only=True (default), only fills in severity for
    pairs where existing severity is Unknown/missing. Preserves DDInter
    labels which are more reliable than rule-based classification.

    Returns dict mapping row index → severity classification result.
    """
    results = {}
    for idx, row in train_df.iterrows():
        existing = str(row.get(existing_severity_col, "Unknown"))
        source = str(row.get(existing_source_col, "none"))

        # Keep DDInter labels — they're from a validated clinical database
        if override_unknown_only and source == "ddinter":
            results[idx] = {
                "severity": existing,
                "confidence": "high",
                "reason": "DDInter validated label",
                "source": "ddinter",
            }
            continue

        # Override teacher-generated labels — they are 92.4% Major (hallucinated)
        # Real DDInter distribution is 70% Moderate, 26.5% Major, 3.5% Minor
        # Teacher severity is unreliable and should always be replaced
        if override_unknown_only and source == "teacher":
            pass  # fall through to rule-based classification below

        # Classify everything else
        result = classify_severity(
            drug1_id=row["drug1_id"],
            drug2_id=row["drug2_id"],
            label_text=str(row.get("label_text", "")),
            profiles=profiles,
            drug1_name=str(row.get("drug1_name", "")),
            drug2_name=str(row.get("drug2_name", "")),
        )
        results[idx] = result

    return results


# ── Evaluation against DDInter ground truth ───────────────────────────────────

def evaluate_against_ddinter(
    train_df,
    profiles: dict,
) -> dict:
    """
    Evaluate classifier accuracy against DDInter-labelled pairs.
    These are the only pairs with reliable ground truth severity.

    Returns accuracy metrics and confusion matrix.
    """
    from collections import Counter

    ddinter_rows = train_df[
        train_df.get("severity_source", train_df.get("severity_source", "none")) == "ddinter"
    ] if "severity_source" in train_df.columns else train_df[
        train_df["severity"] != "Unknown"
    ]

    correct = 0
    total = 0
    confusion = Counter()
    wrong_examples = []

    for idx, row in ddinter_rows.iterrows():
        true_sev = str(row.get("severity", "Unknown"))
        if true_sev == "Unknown":
            continue

        result = classify_severity(
            drug1_id=row["drug1_id"],
            drug2_id=row["drug2_id"],
            label_text=str(row.get("label_text", "")),
            profiles=profiles,
            drug1_name=str(row.get("drug1_name", "")),
            drug2_name=str(row.get("drug2_name", "")),
        )
        pred_sev = result["severity"]

        confusion[(true_sev, pred_sev)] += 1
        total += 1
        if pred_sev == true_sev:
            correct += 1
        elif len(wrong_examples) < 10:
            wrong_examples.append({
                "pair": f"{row.get('drug1_name')} + {row.get('drug2_name')}",
                "label": str(row.get("label_text", ""))[:80],
                "true": true_sev,
                "pred": pred_sev,
                "reason": result["reason"],
            })

    # Coverage — how many Unknown pairs get classified
    unknown_rows = train_df[
        train_df.get("severity", "Unknown").astype(str) == "Unknown"
    ] if "severity" in train_df.columns else train_df

    classified = 0
    for _, row in unknown_rows.head(1000).iterrows():
        result = classify_severity(
            drug1_id=row["drug1_id"],
            drug2_id=row["drug2_id"],
            label_text=str(row.get("label_text", "")),
            profiles=profiles,
        )
        if result["severity"] != "Unknown":
            classified += 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "confusion": dict(confusion),
        "coverage_on_unknown": classified / min(1000, len(unknown_rows)),
        "wrong_examples": wrong_examples,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("Loading data...")
    train_df = pd.read_json(
        "data/processed/dataset_A/train.jsonl", lines=True
    )
    with open("data/processed/drug_profiles.json") as f:
        profiles = json.load(f)

    print(f"Evaluating classifier against DDInter ground truth...")
    metrics = evaluate_against_ddinter(train_df, profiles)

    print(f"\n=== Severity Classifier Evaluation ===")
    print(f"Accuracy on DDInter pairs: {metrics['accuracy']:.1%} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"Coverage on Unknown pairs: {metrics['coverage_on_unknown']:.1%}")

    print(f"\nConfusion matrix (true → predicted):")
    for (true, pred), count in sorted(metrics["confusion"].items()):
        print(f"  {true:10} → {pred:10}: {count:,}")

    print(f"\nWrong classification examples:")
    for ex in metrics["wrong_examples"][:5]:
        print(f"  {ex['pair']}")
        print(f"  Label: {ex['label']}")
        print(f"  True: {ex['true']}  |  Predicted: {ex['pred']}")
        print(f"  Reason: {ex['reason']}")
        print()
