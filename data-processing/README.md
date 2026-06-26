# Data Processing Pipeline

Pipeline for curating raw ChEMBL bioassay data into a model-ready dataset for the A2A adenosine receptor (CHEMBL251).

## Data Source

ChEMBL (https://www.ebi.ac.uk/chembl/) — a manually curated database of bioactive molecules with drug-like properties, aggregating binding and functional assay data from published literature.

## Scope

We restrict to **Ki measurements from binding assays** (assay_type = B, standard_type = Ki, standard_relation = =) as Ki is the most thermodynamically grounded and cross-assay comparable measurement type. IC50 and EC50 are excluded as they are assay-condition dependent in a way Ki is not.

## Pipeline Steps

### 1. Data Collection (`data-collection/pull_chembl.py`)
Pulls raw activity records from the ChEMBL REST API for the A2A receptor target (CHEMBL251). Queries at the individual assay level rather than the compound level, preserving heterogeneity across experiments.

Output: `data/chembl_a2a_raw.csv` — 12,644 records across all measurement types and assay formats.

### 2. EDA (`eda/`)
Exploratory analysis of the raw data prior to curation.

- `eda_chembl.py` — full dataset overview: missing data audit, measurement type breakdown, censored data analysis, duplicate compound analysis, QC flag audit, per-assay statistics
- `plot_raw_values.py` — side-by-side comparison of distributions with and without censoring annotation, illustrating threshold artifacts from assay cutoff concentrations

Key findings from EDA:
- 1,070 censored Ki records (Ki >, >=) clustering at round-number cutoffs (10,000 nM, 1,000 nM), confirming these are assay boundaries not real measurements
- 347 compounds (5.7%) with undefined stereocenters
- 2,165 compounds measured in more than one assay, with high inter-assay variability (median CV > 1.0 for Ki)
- ChEMBL QC flags present on 196 records (outside typical range, potential transcription error)

### 3. Structural Curation (`data-processing/structural_curation.py`)
Cleans and standardizes chemical structures.

Steps:
1. **Filter** to Ki, binding assays, exact measurements only (6,090 records)
2. **Validate SMILES** — drop records RDKit cannot parse
3. **Strip salts** — remove counterions using RDKit SaltRemover, fall back to largest fragment heuristic
4. **Flag undefined stereocenters** — compounds where chiral centers exist but 3D arrangement is unspecified in the SMILES; flagged not dropped
5. **Canonicalize** — convert all SMILES to RDKit canonical form for reliable deduplication

Output: `chembl_a2a_ki_structural.csv` — 6,090 records, 5,000 unique compounds, 347 stereo-flagged.

## Key Design Decisions

**Why Ki only?**
Ki is corrected for reference ligand concentration via the Cheng-Prusoff equation, making it more comparable across assays than IC50. Mixing measurement types without harmonization would introduce systematic bias into training labels.

**Why keep censored records as flags rather than dropping?**
Censored values (Ki > X nM) are informative — they indicate inactives. Dropping them entirely would artificially enrich the dataset for actives. Instead they are excluded from continuous value modeling but can be used as hard inactive labels in binarization.

**Why flag rather than drop undefined stereocenters?**
Enantiomers can have dramatically different binding affinities. Silently merging stereosiomers could corrupt activity labels. Flagging preserves the information and allows downstream analysis of whether stereochemical ambiguity affects model performance.