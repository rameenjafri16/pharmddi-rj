# PharmCoT-RJ: Pathway-Aware Retrieval for Drug-Drug Interaction CoT Distillation

**Rameen Jafri** | University of Guelph | DATA 6400  
**April 2026**

## Background 
PharmCoT is a knowledge distillation pipeline for drug-drug interaction (DDI) classification. Given a pair of drugs, the goal is to predict which of 129 fine-grained interaction types applies — for example, "the serum concentration of Drug A can be increased when combined with Drug B" — along with a severity label (Major/Moderate/Minor) and a mechanistic explanation of why the interaction occurs.

The pipeline works in three stages:
Stage 1 — Teacher trace generation. A large language model (Llama-3.3-70B-Instruct) is prompted with a drug pair, its pharmacological profiles (enzymes, transporters, targets from DrugBank), five retrieved example interactions, and the ground-truth interaction label. It produces a structured chain-of-thought trace: a step-by-step mechanistic explanation, a concise summary, and a severity classification.
Stage 2 — Quality filtering. Generated traces are filtered by a three-model judge ensemble (OpenBioLLM-70B, TxGemma-27B, Qwen2.5-72B) that scores each trace on factual accuracy, mechanistic depth, and clinical relevance. Only high-quality traces are kept for training.
Stage 3 — Student training. A smaller model (Qwen3-8B) is fine-tuned via LoRA on the filtered traces. The student learns to produce the same structured mechanistic reasoning at inference time without needing the teacher's scale.
The trained student classifies drug pairs into 129 interaction types while simultaneously generating a clinically interpretable explanation. 

---

## Contribution 1: Pathway-Aware RAG Retrieval

### Background & Problem

In Stage 1 of the pipeline, before the teacher model explains a drug interaction, it is shown five example interactions to provide context. The quality of these examples directly affects the quality of the reasoning trace. In theory, if the examples are mechanistically relevant, the teacher will produce better explanations.

The original pipeline selects these five examples using Tanimoto structural similarity: it computes Morgan fingerprints for each drug, calculates pairwise Tanimoto coefficients across all training pairs, and retrieves the five pairs whose drugs are most structurally similar to the query pair.

Morgan fingerprints encode a drug's molecular structure as a binary vector where each bit represents the presence or absence of a particular substructural pattern within a given radius of each atom. Two drugs with similar fingerprints share similar local chemical environments around their atoms.

<img width="850" height="445" alt="image" src="https://github.com/user-attachments/assets/8bb1a6f3-0dd6-46dd-9a51-fa8d15115c9e" />

This approach has a fundamental pharmacological flaw. Tanimoto similarity measures how alike two molecules look (whether they share the same substructural fragments). But drug-drug interactions are not determined by what drugs look like. They are determined by what drugs do biologically: which enzymes metabolise them, which transporters move them across membranes, which receptors they bind. Two drugs can be structurally identical yet interact through completely different pathways, and two structurally unrelated drugs can interact through exactly the same mechanism.

The practical consequence is that Tanimoto-retrieved examples are often mechanistically irrelevant to the query pair. The teacher model is being shown "here are five interactions involving structurally similar drugs" when what it needs is "here are five interactions involving drugs that share the same biological pathways."

### Methods

To quantify this problem, I developed a metric called **Mechanistic Overlap Rate (MOR)**.

MOR measures the fraction of retrieved examples that share at least one biological pathway node with the query pair. A pathway node is any enzyme, transporter, or receptor that both drugs in a pair have in common — for example, if the query pair involves two CYP3A4 substrates, any retrieved example where at least one drug is also a CYP3A4 substrate counts as a mechanistic hit. MOR ranges from 0 to 1, where 1 means every retrieved example is mechanistically relevant to the query.

To make this concrete: suppose the query pair is warfarin + fluconazole, which interact because fluconazole inhibits CYP2C9, the enzyme that metabolises warfarin. A mechanistically relevant retrieved example would be something like warfarin + amiodarone (also a CYP2C9 inhibitor) or phenytoin + fluconazole (another CYP2C9 substrate affected by the same inhibitor). These examples give the teacher model the right reasoning template. A mechanistically irrelevant example — say, ibuprofen + naproxen retrieved because both are small NSAIDs with similar ring structures — tells the teacher nothing useful about CYP2C9 inhibition and may actively mislead it.

Under the Tanimoto strategy, roughly 1 in 5 retrieved examples is mechanistically irrelevant in exactly this way. Under pathway retrieval, essentially all 5 are on-target.

Pathway-aware retrieval selects examples based on pathway node overlap rather than structural similarity. For each query pair, I extract the full set of enzyme, transporter, and target annotations from DrugBank 5.1.17 for both drugs and compute a weighted overlap score against all training pairs. The scoring function upweights enzyme sharing (3×) over transporter sharing (2×) over target sharing (1×), reflecting the relative clinical importance of each annotation type — CYP-mediated metabolic interactions are more pharmacologically predictable and better annotated than transporter or target interactions. The top-5 scoring training pairs are returned as retrieved examples.

To evaluate both strategies systematically, I constructed two filtered datasets from DrugBank 5.1.17:

Dataset A (≥130 pairs per class): 129 interaction classes, 236K training pairs, 256K test pairs — matching the original pipeline's filtering threshold exactly, enabling direct comparison.
Dataset B (≥20 pairs per class): 194 interaction classes, 239K training pairs, 257K test pairs — includes 65 rare interaction classes excluded by the original pipeline.

Within each dataset, interaction classes were split into frequency tiers based on the number of training pairs per class:

Head (15 classes): the most frequent interaction types
Mid (65 classes): intermediate frequency
Tail (49 classes in A / 114 in B): rare interactions

For each tier, I sampled query pairs from each class, ran both retrieval strategies, and computed MOR for each retrieved set. I also measured coverage (the percentage of query pairs for which at least one retrieved example exists) to test whether pathway retrieval sacrifices recall for precision.

### Results

![MOR Comparison by Tier](figures/fig5_mor_comparison_by_tier.png)

Pathway retrieval achieves near-perfect mechanistic relevance across all tiers and both datasets. Tanimoto retrieval performs consistently around 80% — meaning roughly 1 in 5 examples shown to the teacher is mechanistically irrelevant to the query pair regardless of how common or rare the interaction class is.

**Dataset A results:**

| Tier | Tanimoto MOR | Pathway MOR | Δ | Tanimoto Coverage | Pathway Coverage |
|------|-------------|-------------|---|-------------------|-----------------|
| Head | 79.9% | 99.3% | +19.5pp | 84.0% | 99.3% |
| Mid | 79.0% | 99.4% | +20.4pp | 81.2% | 99.4% |
| Tail | 84.2% | 99.6% | +15.4pp | 84.9% | 99.6% |

**Dataset B results:**

| Tier | Tanimoto MOR | Pathway MOR | Δ | Tanimoto Coverage | Pathway Coverage |
|------|-------------|-------------|---|-------------------|-----------------|
| Head | 79.1% | 98.0% | +18.9pp | 83.3% | 98.0% |
| Mid | 78.6% | 99.5% | +20.9pp | 81.2% | 99.5% |
| Tail | 82.1% | 99.9% | +17.8pp | 83.4% | 99.9% |

![Coverage and MOR Double Win](figures/fig7_coverage_and_mor.png)

Five findings stand out beyond the headline numbers.

**Finding 1**: There is no coverage tradeoff. The natural concern with pathway retrieval is that it might cover fewer pairs — not every drug in DrugBank has complete enzyme and transporter annotations. In practice the opposite is true. Pathway retrieval covers 98–100% of pairs compared to Tanimoto's 81–85%. The reason is that DrugBank target annotations — which include receptor binding and protein interaction data — are present for 72% of drugs, providing near-complete coverage even when enzyme-specific data is sparse. Switching to pathway retrieval improves both the quality and the breadth of retrieved examples simultaneously.

**Finding 2**: The improvement is uniform, not concentrated in rare classes. Before running the experiment I expected pathway retrieval to help most for rare tail classes, where structural similarity is presumably weakest and mechanistic reasoning matters most. The data shows something more interesting: the MOR improvement is consistent across head, mid, and tail classes with no clear trend. This is a stronger result than the original hypothesis — it means the flaw in Tanimoto retrieval is systematic across the entire interaction space, not a niche problem affecting only rare classes.

![Delta vs Class Frequency](figures/fig6_delta_vs_frequency.png)

**Finding 3**: Tanimoto similarity is essentially a random signal for this task. Tanimoto similarity between interacting drug pairs averages ~0.10 across all tiers — barely above zero. This is not because rare interaction classes have unusually dissimilar drugs; the distribution is flat across head, mid, and tail. Structurally similar drugs are simply not more likely to share interaction mechanisms than structurally dissimilar ones. This confirms that the flaw in the original retrieval strategy is not a matter of degree — it is a fundamental category error.

![Tanimoto Similarity Distribution](figures/fig3_tanimoto_similarity_by_tier.png)

**Finding 4**: Target annotations drive coverage, not enzyme annotations. 98% of pathway coverage comes from target annotations rather than enzyme or transporter data. This was unexpected. CYP-mediated metabolic interactions are the most clinically common type of DDI, so I assumed enzyme annotations would dominate. In practice, the broader target annotation layer — which includes receptors, ion channels, and other protein binding data — is what makes pathway retrieval work at scale. Enzyme data alone would have given much lower coverage.

![Annotation Type Breakdown](figures/fig_annotation_type_breakdown.png)

**Finding 5**: Pathway retrieval unlocks 23 interaction classes that are invisible to Tanimoto. In Dataset B, 23 tail-class interaction types have zero Tanimoto coverage — no training pairs involving these interactions have drugs with sufficient fingerprint overlap to retrieve any examples. These are the classes the original pipeline excluded entirely with the ≥130 pairs threshold. Pathway retrieval achieves near-complete coverage on all 23. This raises the possibility of expanding the pipeline's classification scope from 129 to 194 interaction types — not by collecting more data, but by fixing the retrieval strategy.

![Coverage Divergence](figures/fig_coverage_divergence.png)

---

### What This Means

The retrieval quality experiment proves that pathway retrieval is pharmacologically superior to Tanimoto retrieval — the examples it selects are almost always mechanistically relevant, while Tanimoto examples are wrong about 1 in 5 times. This is a clean, quantitative result that does not require running the full pipeline to validate.

What it does not yet prove is whether this improvement in example quality translates into better teacher traces, and in turn a better student model. That question requires running teacher generation under both conditions and comparing the resulting traces — which is exactly what the pilot experiment (described in Contribution 3) is designed to test.

The hypothesis is straightforward: a teacher shown mechanistically relevant examples should produce traces that reason more accurately about the underlying pharmacological mechanism. If that hypothesis holds, we would expect to see higher grounded factuality scores on pathway traces, and ultimately higher classification F1 on the student model trained on those traces. The effect should be largest on the 8.3% of training pairs involving prodrug interactions, where the direction of effect is reversed and mechanistically correct examples are most critical — these pairs are the subject of Contribution 2.

Whether the improvement is large enough to be clinically meaningful is an open question. A ~20 percentage point improvement in MOR is a substantial pharmacological difference. Whether it moves the needle on student F1 depends on how sensitive the distillation pipeline is to example quality — something only the full experiment can answer.


## Contribution 2: Pharmacologically-Correct Teacher Prompts

### Background

Beyond the retrieval strategy, the teacher model's reasoning quality depends on the information it receives in its prompt. In PharmCoT, each teacher prompt contains: the query drug pair, their pharmacological profiles from DrugBank (enzymes, transporters, targets, mechanism of action), the five retrieved examples, and the ground-truth interaction label. The teacher reads all of this and produces a step-by-step mechanistic explanation.

The prompt is therefore the teacher's entire pharmacological context. Anything missing from the prompt is information the teacher has to infer on its own — and large language models, even at 70B parameters, make systematic errors when asked to reason about pharmacology without explicit guidance. Three such errors exist in the original prompt design.

---

### Fix 1: PK/PD Interaction Type Flag

#### The Problem

Drug-drug interactions fall into two fundamentally different categories that require completely different reasoning frameworks.

A **pharmacokinetic (PK) interaction** occurs when one drug changes how much of the other drug gets into the body — by blocking or accelerating the enzymes that metabolise it, the transporters that move it across membranes, or the proteins that bind it in the bloodstream. The reasoning here is about ADME: absorption, distribution, metabolism, excretion. The teacher needs to think about CYP enzymes, P-glycoprotein, drug half-life, plasma concentrations.

A **pharmacodynamic (PD) interaction** occurs when two drugs act on the same receptor or physiological system at the same time, producing a combined effect that is stronger, weaker, or qualitatively different from either drug alone. The reasoning here is about receptor occupancy, downstream signalling, and physiological consequences. The teacher needs to think about mechanisms of action, target overlap, and additive or antagonistic effects.

The original PharmCoT prompt contains no signal about which type applies. The teacher has to infer the interaction type from the label text alone — for example, "The serum concentration of Drug A can be increased when combined with Drug B" implies PK reasoning, while "Drug A may increase the CNS depressant activities of Drug B" implies PD reasoning. This inference is usually possible from the label, but it is an unnecessary cognitive load that introduces inconsistency. A teacher that misclassifies interaction type will apply the wrong reasoning framework entirely — generating a trace about receptor binding for a metabolic interaction, or about enzyme inhibition for a pharmacodynamic one.

#### My Approach

I built a keyword-based classifier, `classify_pk_pd()`, that maps any DrugBank interaction template to either PK or PD with 100% coverage across all 129 interaction classes in Dataset A — verified empirically with zero ambiguous cases.

The classifier checks for PK-specific language first (serum concentration, metabolism, excretion, absorption, half-life, clearance, bioavailability, protein binding) and returns PK if any match. Otherwise it returns PD. PK takes priority over PD when both keyword types appear in the same template, because many templates describe a PK mechanism that leads to a PD outcome — for example, "excretion rate decreased resulting in lower serum concentration" is a PK interaction even though serum concentration has clinical consequences.

The teacher prompt now includes a single line for every trace:

> *Interaction type: PK (pharmacokinetic — reason about ADME mechanisms, 
> enzyme/transporter roles, and drug level changes)*

or

> *Interaction type: PD (pharmacodynamic — reason about receptor/system effects 
> and combined pharmacological actions)*

Of the 129 classes in Dataset A, 15 are PK and 114 are PD — a distribution that reflects the clinical reality that most documented DDIs are pharmacodynamic in nature, while the most dangerous and well-characterised ones tend to be pharmacokinetic.


### Fix 2: Prodrug Warning

#### The Problem

A prodrug is a pharmacologically inactive compound that must be converted into its active form by an enzyme in the body before it can exert any therapeutic effect. This conversion typically happens in the liver and is carried out by CYP enzymes or esterases. The drug you swallow is not the drug that works — it is a chemical precursor that your body activates.

This creates a critical asymmetry in how enzyme inhibition affects drug levels. For a normal drug, the standard reasoning is:

> Enzyme inhibitor + substrate → enzyme is blocked → drug is not broken down → 
> **more drug accumulates in blood** → stronger effect, potential toxicity

For a prodrug, the reasoning is exactly reversed:

> Enzyme inhibitor + prodrug substrate → enzyme is blocked → prodrug is not 
> activated → **less active drug in blood** → weaker effect, potential treatment 
> failure

The teacher model has no way to know which direction applies unless it is explicitly told the drug is a prodrug. Without this information, it will default to the standard substrate+inhibitor reasoning and generate a trace with the wrong direction of effect.

The clinical stakes are high. The most well-known example is **clopidogrel + omeprazole**. Clopidogrel is a prodrug — it requires CYP2C19 to convert it into its active thiol metabolite, which is what actually inhibits platelet aggregation and prevents blood clots. Omeprazole, a common heartburn drug, is a potent CYP2C19 inhibitor. Co-administration blocks clopidogrel's activation, reducing its antiplatelet effect by up to 40%. For a heart attack patient taking clopidogrel to prevent a second event, this interaction can be fatal. A teacher model reasoning about this pair without a prodrug flag would likely write:

> *"Omeprazole inhibits CYP2C19, blocking clopidogrel's metabolism, causing 
> clopidogrel levels to increase and enhancing its antiplatelet effect."*

This is pharmacologically backwards. The correct trace says:

> *"Omeprazole inhibits CYP2C19, blocking clopidogrel's bioactivation to its active 
> thiol metabolite. This reduces active metabolite levels, decreasing antiplatelet 
> efficacy and increasing thrombotic risk."*

Both traces mention CYP2C19. Both pass standard quality checks. But one is dangerously wrong.

#### Scale of the Problem

To quantify how often this occurs in the dataset, I scanned DrugBank 5.1.17 for prodrug annotations using two signals: explicit prodrug group flags and prodrug references in drug description or mechanism-of-action text. I identified **183 prodrugs** with DDI interactions in DrugBank, of which **125 appear in Dataset A**. These 125 prodrugs are involved in **19,639 training pairs — 8.3% of all training data**.

Notable prodrugs in the dataset include clopidogrel, prasugrel, and ticlopidine (antiplatelet agents), simvastatin and lovastatin (statins), nearly all ACE inhibitors (enalapril, ramipril, lisinopril, perindopril — the "-pril" suffix typically indicates 
a prodrug), fosphenytoin (epilepsy), capecitabine (cancer), levodopa (Parkinson's), and valganciclovir (antiviral).

#### My Approach

I added a prodrug warning to the teacher prompt for any pair involving a prodrug. 
The warning appears as:

> *⚠️ PRODRUG WARNING: Clopidogrel is a prodrug — it is pharmacologically inactive 
> until converted to its active form by an enzyme. If an enzyme involved in its 
> activation is inhibited, the result is DECREASED active drug levels (not increased). 
> Reason about activation, not elimination.*

This single sentence gives the teacher the pharmacological context it needs to reason in the correct direction. It fires for 8.3% of training pairs — the subset where direction-of-effect errors are most likely and most clinically consequential.

The prodrug list is saved to `data/processed/prodrug_ids.json` and loaded once at the start of teacher generation. The lookup adds negligible overhead — a set membership check per pair.

### Fix 3: Raised Profile Truncation Caps

#### The Problem

The teacher model's pharmacological context for each drug comes from its DrugBank profile — a structured summary of its enzymes, transporters, targets, mechanism of action, and other annotations. This profile is formatted and inserted into the prompt 
by `_format_drug_profile()` in `data_preparation.py`.

The original implementation truncates these profiles at prompt construction time:

- Enzymes: capped at **5**
- Transporters: capped at **3**
- Targets: capped at **3**

For most drugs these caps are not binding — the average drug in DrugBank has fewer than 5 enzyme annotations. But for heavily metabolised drugs, the caps quietly drop pharmacologically important information from the prompt with no indication that 
truncation occurred. The teacher receives an incomplete profile and has no way to know it is missing data.

This matters most for drugs involved in complex polypharmacy interactions — exactly the cases where complete enzyme information is most critical. Nicotine, for example, is metabolised by CYP2A6, CYP2B6, CYP2C9, CYP2D6, CYP2E1, FMO3, and several UGT enzymes. Under the original caps, the teacher sees only the first five. If the interaction being explained involves an enzyme that appears sixth or later in DrugBank's listing, the teacher's prompt contains no mention of it. The resulting trace may reason about the wrong enzyme entirely.

A diagnostic check across all 4,629 drug profiles in Dataset A found that 94 drugs (2.0%) hit the enzyme cap, 207 drugs (4.5%) hit the target cap, and 239 drugs (5.2%) hit the transporter cap. The drugs hitting the enzyme cap include Nicotine, Troglitazone, and Dapsone — all compounds with broad CYP involvement where complete enzyme context is clinically meaningful.

#### My Approach

I raised the truncation caps in `_format_drug_profile()`:

- Enzymes: 5 → **8**
- Transporters: 3 → **5**  
- Targets: 3 → **5**

This is a three-number change with no computational overhead. Prompt length increases only for the small fraction of drugs that were previously truncated. The caps were not removed entirely because very long profiles can push complex prompts toward the model's context window limit. The raised values represent a balance between completeness and prompt length informed by the annotation count distribution across the dataset.
