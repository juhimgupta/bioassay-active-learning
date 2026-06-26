"""
Pull raw bioassay data from ChEMBL for the Adenosine A2A receptor (CHEMBL251).

Install first:
    pip install chembl-webresource-client pandas

Run:
    python3 data-collection/pull_chembl.py
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_ID = 'CHEMBL251'  # Adenosine A2A receptor (a GPCR)

# ── Pull raw activity records ────────────────────────────────────────────────
# We query at the individual ASSAY level — not the pre-aggregated compound level.
# This is what gives us the messy, multi-assay heterogeneity to work with.

print(f"Pulling activity data for {TARGET_ID}...")

activity = new_client.activity
results = activity.filter(
    target_chembl_id=TARGET_ID,
    standard_type__in=['IC50', 'Ki', 'EC50', 'Kd'],
).only([
    'molecule_chembl_id',
    'canonical_smiles',
    'standard_type',        # IC50 / Ki / EC50 / Kd
    'standard_value',       # numerical value
    'standard_units',       # nM, uM, M ... often inconsistent
    'standard_relation',    # '=', '>', '<', '>=' (inequalities = censored data)
    'assay_chembl_id',
    'assay_description',
    'assay_type',           # B=binding, F=functional
    'document_chembl_id',
    'activity_comment',     # e.g. "Not Active", "inconclusive"
    'data_validity_comment',# e.g. "Outside typical range"
    'pchembl_value',        # -log10(value in M), pre-computed by ChEMBL
])

df = pd.DataFrame(list(results))

print(f"\nRaw pull: {len(df)} records, {df['molecule_chembl_id'].nunique()} unique compounds")
print(f"Columns: {df.columns.tolist()}")
print()
print("── First 5 rows (raw) ──")
print(df.head().to_string())

# Save raw
df.to_csv('data/chembl_a2a_raw.csv', index=False)
print("\nSaved: data/chembl_a2a_raw.csv")

# ── Quick look at the messiness ──────────────────────────────────────────────
print("\n── What makes this data messy ──")

print(f"\n1. Standard units (should all be the same, but aren't):")
print(df['standard_units'].value_counts().head(10))

print(f"\n2. Standard relation (inequalities = censored measurements):")
print(df['standard_relation'].value_counts())

print(f"\n3. Activity comments (flags from the data submitter):")
print(df['activity_comment'].value_counts().head(10))

print(f"\n4. Data validity comments (ChEMBL's own QC flags):")
print(df['data_validity_comment'].value_counts())

print(f"\n5. Assay types present:")
print(df['assay_type'].value_counts())

print(f"\n6. Measurement types:")
print(df['standard_type'].value_counts())

print(f"\n7. Duplicate compounds (same molecule, multiple assays):")
dup_counts = df['molecule_chembl_id'].value_counts()
print(f"   Compounds with >1 measurement: {(dup_counts > 1).sum()}")
print(f"   Max measurements for one compound: {dup_counts.max()}")
print(f"   Median measurements per compound: {dup_counts.median()}")