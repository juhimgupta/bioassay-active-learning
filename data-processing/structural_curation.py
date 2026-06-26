"""
Section 1: Structural Curation
================================
Run from project root:
    python curation/01_structural_curation.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/chembl_a2a_raw.csv')

# ── Filter to Ki, binding assays, exact measurements only ────────────────────
ki = df[(df['standard_type'] == 'Ki') &
        (df['assay_type'] == 'B') &
        (df['standard_relation'] == '=')].copy()

print(f"Starting records: {len(ki):,}")

# ── Step 1: Validate SMILES ───────────────────────────────────────────────────
def validate_smiles(smiles):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(smiles)

ki['mol'] = ki['canonical_smiles'].apply(validate_smiles)
n_invalid = ki['mol'].isna().sum()
print(f"\nStep 1 — SMILES validation:")
print(f"  Invalid SMILES dropped: {n_invalid}")
ki = ki[ki['mol'].notna()].copy()
print(f"  Remaining: {len(ki):,}")

# ── Step 2: Strip salts ───────────────────────────────────────────────────────
remover = SaltRemover()

def strip_salts(mol):
    if mol is None:
        return None
    mol_stripped = remover.StripMol(mol)
    # fallback: if fragments remain, keep largest by heavy atom count
    if '.' in Chem.MolToSmiles(mol_stripped):
        fragments = Chem.GetMolFrags(mol_stripped, asMols=True)
        mol_stripped = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    return mol_stripped

ki['mol'] = ki['mol'].apply(strip_salts)
n_lost = ki['mol'].isna().sum()
print(f"\nStep 2 — Salt stripping:")
print(f"  Records lost: {n_lost}")
ki = ki[ki['mol'].notna()].copy()
print(f"  Remaining: {len(ki):,}")

# ── Step 3: Flag stereochemistry ──────────────────────────────────────────────
def flag_stereochemistry(mol):
    if mol is None:
        return None, False
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    unassigned = [c for c in chiral_centers if c[1] == '?']
    return mol, len(unassigned) > 0

results = ki['mol'].apply(flag_stereochemistry)
ki['mol']              = results.apply(lambda x: x[0])
ki['undefined_stereo'] = results.apply(lambda x: x[1])

print(f"\nStep 3 — Stereochemistry flagging:")
print(f"  Compounds with undefined stereocenters: {ki['undefined_stereo'].sum():,} "
      f"({ki['undefined_stereo'].mean()*100:.1f}%) — flagged, not dropped")

# ── Step 4: Canonicalize ──────────────────────────────────────────────────────
def canonicalize(mol):
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

ki['canonical_smiles_clean'] = ki['mol'].apply(canonicalize)

# check how many SMILES changed after cleaning
changed = (ki['canonical_smiles_clean'] != ki['canonical_smiles']).sum()
print(f"\nStep 4 — Canonicalization:")
print(f"  SMILES changed after cleaning: {changed:,} ({changed/len(ki)*100:.1f}%)")

# ── Summary ───────────────────────────────────────────────────────────────────
ki = ki.drop(columns=['mol'])

print(f"\n── Structural curation summary ──────────────────────────────")
print(f"  Input records:              {len(df):,}")
print(f"  After type/assay filter:    {len(ki):,}")
print(f"  Unique compounds:           {ki['canonical_smiles_clean'].nunique():,}")
print(f"  Flagged undefined stereo:   {ki['undefined_stereo'].sum():,}")

ki.to_csv('data/processed/chembl_a2a_ki_structural.csv', index=False)
print(f"\nSaved: data/processed/chembl_a2a_ki_structural.csv")