# Bioassay Active Learning Pipeline

End-to-end pipeline for curating raw ChEMBL bioassay data into a model-ready dataset, training a predictive model, and applying active learning for experimental prioritization.

Built as a practical implementation of the data → model → experiment → insight loop used in computational receptor biology and small molecule discovery.

## Overview

**Target:** Adenosine A2A receptor (CHEMBL251) — a GPCR relevant to Parkinson's disease and cancer immunotherapy.

**Task:** Binary activity prediction (active = Ki <= 100 nM) and experimental prioritization via active learning.

**Dataset:** 5,458 curated compounds from ChEMBL, 42.5% active.

## Pipeline

```
data-collection/
    pull_chembl.py          — fetch raw assay data from ChEMBL API

eda/
    eda_chembl.py           — full dataset EDA (missing data, distributions, duplicates)
    plot_raw_values.py      — Ki distribution plots, linear vs log scale
    plot_censored_comparison.py — censoring artifact visualization

data-processing/
    structural_curation.py  — SMILES validation, salt stripping, stereochemistry, canonicalization
    value_curation.py       — QC flags, aggregation, binarization, censored inactives
    splitting.py            — scaffold and temporal dataset splits
    featurization.py        — Morgan fingerprints + physicochemical descriptors

modeling/
    random_forest.py        — calibrated Random Forest baseline with uncertainty estimation
    active_learning.py      — active learning simulation: 4 acquisition strategies
```

## Data Source

ChEMBL (https://www.ebi.ac.uk/chembl/) — manually curated bioactive molecule database aggregating binding and functional assay data from published literature. Queried at the individual activity record level to preserve cross-assay heterogeneity.

## Curation Decisions

### Measurement type: Ki only
Ki is corrected for reference ligand concentration via the Cheng-Prusoff equation, making it more comparable across assays than IC50. IC50 is assay-condition dependent in a way Ki is not. Mixing measurement types without harmonization introduces systematic bias into training labels.

### Assay type: Binding only
Binding assays (assay_type=B) measure physical receptor occupancy. Functional assays (assay_type=F) measure downstream cellular response. These are fundamentally different quantities and should not be naively combined.

### Censored data handling
Censored records (Ki >, >=) represent compounds where the lab hit their max test concentration without observing sufficient inhibition. They carry no real Ki value but are informative as inactives. Records with censored cutoff >= 10,000 nM (pChEMBL <= 5) are incorporated as hard inactive labels after verifying no overlap with the active set.

### Activity threshold: pChEMBL >= 7 (Ki <= 100 nM)
Initial analysis at pChEMBL >= 6 gave 76.3% active — reflecting ChEMBL's publication bias toward active compounds. Raising to pChEMBL >= 7 gives a more balanced 42.5% active rate and a more stringent, biologically meaningful definition of activity.

### Duplicate aggregation
Same compound measured across multiple assays: geometric mean of pChEMBL values (equivalent to mean in log space). Compounds with coefficient of variation > 0.2 flagged as high variance. 4,437 / 4,990 compounds came from a single assay.

### Stereochemistry
Compounds with undefined stereocenters flagged but not dropped (309 compounds, 5.7%). Enantiomers can have dramatically different binding affinities; silently merging stereoisomers could corrupt activity labels.

## Dataset Splits

| Split | Train | Test | Train hit% | Test hit% | Scaffold overlap |
|-------|-------|------|------------|-----------|-----------------|
| Random | 4,366 | 1,092 | 42.7% | 41.8% | 363 scaffolds |
| Scaffold | 4,363 | 1,095 | 43.1% | 40.2% | 0 |
| Temporal | 4,460 | 998 | 37.5% | 64.8% | N/A |

**Scaffold split** — Murcko scaffolds kept intact across train/test boundary. Tests generalization to novel chemical series. 0 scaffold overlap confirmed.

**Temporal split** — compounds ordered by molecule ChEMBL ID (proxy for publication order). Tests generalization to newer compounds. The 37.5% → 64.8% hit rate shift reflects medicinal chemistry optimization trends over time and represents a realistic distribution shift scenario for production deployment.

## Featurization

- **Morgan fingerprints (ECFP4)** — radius=2, 2048 bits. Encodes local chemical environment. Industry standard for QSAR modeling.
- **Physicochemical descriptors** — MW, LogP, TPSA, HBD, HBA, RotBonds, ArRings, HeavyAtoms, Complexity, FractionCSP3. Interpretable global molecular properties.
- **Total features:** 2,058

Active vs inactive descriptor analysis: active compounds show higher TPSA, HBA, and molecular complexity, consistent with A2A's hydrogen-bonding-rich binding pocket. Lower FractionCSP3 in actives reflects the planar aromatic cores common in A2A antagonists.

## Modeling Results

**Random Forest (calibrated, 200 trees, class_weight=balanced)**

| Split | AUC-ROC |
|-------|---------|
| Scaffold | 0.890 |
| Temporal | 0.831 |

AUC gap between scaffold and temporal splits reflects distribution shift — the model trained on older compounds faces a more active population at test time. Probability calibration applied via isotonic regression to correct RF overconfidence. Uncertainty estimated as prediction variance across trees.

## Active Learning Results

Simulated 10 rounds × 20 compounds = 200 experiments from the scaffold test pool (1,095 compounds, 40.2% active).

| Strategy | Actives found | Enrichment |
|----------|--------------|------------|
| Random | 89 | 1.11x |
| Exploitation | 180 | 2.25x |
| Exploration | 120 | 1.50x |
| Balanced | 159 | 1.99x |

**Key findings:**
- Exploitation (highest predicted probability) finds the most actives per experiment but degrades model AUC over rounds due to selection bias — the training set becomes depleted of inactives, making the decision boundary harder to learn
- Exploration (highest uncertainty) shows volatile per-round hit rates but improves model AUC most consistently by sampling diverse chemical space
- Balanced strategy (0.5 × probability + 0.5 × uncertainty, normalized) achieves strong enrichment while maintaining better model quality than pure exploitation
- Random barely beats expected baseline (1.11x), confirming the model adds real value

**Practical implications:**
- Use exploitation when experimental budget is limited and the goal is finding actives fast (lead finding)
- Use exploration when building a platform model for reuse across campaigns
- Use balanced when both goals matter — finding actives now while improving the model for future rounds

## Requirements

```
pip install chembl-webresource-client rdkit scikit-learn pandas numpy matplotlib
```

## Usage

```bash
# 1. Pull data
python data-collection/pull_chembl.py

# 2. EDA
python eda/eda_chembl.py
python eda/plot_raw_values.py
python eda/plot_censored_comparison.py

# 3. Curation & splitting
python data-processing/structural_curation.py
python data-processing/value_curation.py
python data-processing/splitting.py
python data-processing/featurization.py

# 4. Modeling
python modeling/random_forest.py
python modeling/active_learning.py
```