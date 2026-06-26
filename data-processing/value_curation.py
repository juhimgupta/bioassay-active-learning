"""
Section 2: Value Curation & Aggregation (v2)
=============================================
Changes from v1:
  - Raised activity threshold to pChEMBL >= 7 (Ki <= 100 nM)
  - Incorporated censored records as hard inactives

Run from project root after structural_curation.py:
    python data-processing/value_curation.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Load structurally curated data ────────────────────────────────────────────
df = pd.read_csv('data/processed/chembl_a2a_ki_structural.csv')
df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')

print("=" * 65)
print("Section 2: Value Curation & Aggregation (v2)")
print("=" * 65)
print(f"\nStarting records: {len(df):,}")
print(f"Unique compounds: {df['canonical_smiles_clean'].nunique():,}")

# ── Step 1: Drop QC-flagged records ──────────────────────────────────────────
bad_qc = ['Outside typical range', 'Potential transcription error']
n_before = len(df)
df = df[~df['data_validity_comment'].isin(bad_qc)].copy()
print(f"\nStep 1 — QC flag removal:")
print(f"  Dropped: {n_before - len(df)}")
print(f"  Remaining: {len(df):,}")

# ── Step 2: Drop inhibition < 50% rows ───────────────────────────────────────
n_before = len(df)
inh_mask = df['activity_comment'].str.contains('Inhibition', na=False)
df = df[~inh_mask].copy()
print(f"\nStep 2 — Inhibition < 50% removal:")
print(f"  Dropped: {n_before - len(df)}")
print(f"  Remaining: {len(df):,}")

# ── Step 3: Drop records missing pChEMBL ─────────────────────────────────────
n_before = len(df)
df = df[df['pchembl_value'].notna()].copy()
print(f"\nStep 3 — Drop missing pChEMBL:")
print(f"  Dropped: {n_before - len(df)}")
print(f"  Remaining: {len(df):,}")

# ── Step 4: Aggregate duplicates ─────────────────────────────────────────────
agg = (df.groupby('canonical_smiles_clean')
         .agg(
             pchembl_mean=('pchembl_value', 'mean'),
             pchembl_std=('pchembl_value', 'std'),
             pchembl_min=('pchembl_value', 'min'),
             pchembl_max=('pchembl_value', 'max'),
             n_assays=('pchembl_value', 'count'),
             undefined_stereo=('undefined_stereo', 'any'),
             molecule_chembl_id=('molecule_chembl_id', 'first'),
             assay_chembl_id=('assay_chembl_id', lambda x: '|'.join(x.unique())),
         )
         .reset_index())

print(f"\nStep 4 — Duplicate aggregation:")
print(f"  Unique compounds: {len(agg):,}")
print(f"  Compounds from 1 assay:    {(agg['n_assays'] == 1).sum():,}")
print(f"  Compounds from 2-5 assays: {((agg['n_assays'] > 1) & (agg['n_assays'] <= 5)).sum():,}")
print(f"  Compounds from >5 assays:  {(agg['n_assays'] > 5).sum():,}")

# ── Step 5: Flag high variance compounds ──────────────────────────────────────
agg['cv'] = agg['pchembl_std'] / agg['pchembl_mean']
agg['high_variance'] = agg['cv'] > 0.2
print(f"\nStep 5 — High variance flagging (CV > 0.2):")
print(f"  High variance compounds: {agg['high_variance'].sum():,}")
print(f"  Single-assay compounds (no CV): {agg['cv'].isna().sum():,}")

# ── Step 6: Binarize with raised threshold ────────────────────────────────────
THRESHOLD = 7.0  # pChEMBL >= 7 = Ki <= 100 nM
agg['active'] = (agg['pchembl_mean'] >= THRESHOLD).astype(int)
agg['data_source'] = 'exact'

print(f"\nStep 6 — Binarization (pChEMBL >= {THRESHOLD}, Ki <= 100 nM):")
print(f"  Active:   {agg['active'].sum():,} ({agg['active'].mean()*100:.1f}%)")
print(f"  Inactive: {(agg['active']==0).sum():,} ({(1-agg['active'].mean())*100:.1f}%)")

# ── Step 7: Incorporate censored records as hard inactives ────────────────────
print(f"\nStep 7 — Incorporating censored records as hard inactives:")

raw = pd.read_csv('data/chembl_a2a_raw.csv')
raw['standard_value'] = pd.to_numeric(raw['standard_value'], errors='coerce')

# Pull censored Ki binding records
censored = raw[
    (raw['standard_type'] == 'Ki') &
    (raw['assay_type'] == 'B') &
    (raw['standard_relation'].isin(['>', '>='])) &
    (raw['standard_value'].notna())
].copy()
print(f"  Raw censored Ki binding records: {len(censored):,}")

# Only keep records where cutoff >= 10,000 nM (pChEMBL <= 5)
# i.e. we're confident these are genuinely inactive
CENSORED_CUTOFF_NM = 10000
censored = censored[censored['standard_value'] >= CENSORED_CUTOFF_NM].copy()
print(f"  After cutoff filter (>= {CENSORED_CUTOFF_NM} nM): {len(censored):,}")

# Apply same structural curation steps
remover = SaltRemover()

def curate_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = remover.StripMol(mol)
    if '.' in Chem.MolToSmiles(mol):
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)

censored['canonical_smiles_clean'] = censored['canonical_smiles'].apply(curate_smiles)
censored = censored[censored['canonical_smiles_clean'].notna()].copy()
print(f"  After structural curation: {len(censored):,}")

# Deduplicate censored records
censored_dedup = (censored.groupby('canonical_smiles_clean')
                           .agg(molecule_chembl_id=('molecule_chembl_id', 'first'))
                           .reset_index())
print(f"  Unique censored compounds: {len(censored_dedup):,}")

# Remove any overlap with our existing curated compounds
existing_smiles = set(agg['canonical_smiles_clean'])
censored_dedup = censored_dedup[
    ~censored_dedup['canonical_smiles_clean'].isin(existing_smiles)
].copy()
print(f"  After removing overlap with curated set: {len(censored_dedup):,}")

# Also remove any overlap with active compounds specifically
active_smiles = set(agg[agg['active'] == 1]['canonical_smiles_clean'])
n_active_overlap = censored_dedup['canonical_smiles_clean'].isin(active_smiles).sum()
print(f"  Compounds overlapping with actives (already removed): {n_active_overlap}")

# Label as inactive and add metadata
censored_dedup['active']          = 0
censored_dedup['pchembl_mean']    = np.nan
censored_dedup['pchembl_std']     = np.nan
censored_dedup['pchembl_min']     = np.nan
censored_dedup['pchembl_max']     = np.nan
censored_dedup['n_assays']        = np.nan
censored_dedup['cv']              = np.nan
censored_dedup['high_variance']   = False
censored_dedup['undefined_stereo']= False
censored_dedup['assay_chembl_id'] = np.nan
censored_dedup['data_source']     = 'censored_inactive'

# ── Step 8: Combine and final summary ────────────────────────────────────────
final = pd.concat([agg, censored_dedup], ignore_index=True)

print(f"\nStep 8 — Final combined dataset:")
print(f"  Total compounds:  {len(final):,}")
print(f"  Active:           {final['active'].sum():,} ({final['active'].mean()*100:.1f}%)")
print(f"  Inactive:         {(final['active']==0).sum():,} ({(1-final['active'].mean())*100:.1f}%)")
print(f"    From exact measurements: {(agg['active']==0).sum():,}")
print(f"    From censored records:   {len(censored_dedup):,}")

print(f"\n── Final summary ────────────────────────────────────────────")
print(f"  Unique compounds:         {len(final):,}")
print(f"  Active (pChEMBL >= 7):    {final['active'].sum():,}")
print(f"  Inactive:                 {(final['active']==0).sum():,}")
print(f"  High variance flagged:    {final['high_variance'].sum():,}")
print(f"  Undefined stereo flagged: {final['undefined_stereo'].sum():,}")

# ── Plots ─────────────────────────────────────────────────────────────────────
DARK  = '#0f1117'
PANEL = '#1a1d27'
BLUE  = '#4c9be8'
GREEN = '#4ecb71'
AMBER = '#f0a500'
RED   = '#e85555'
LIGHT = '#c8ccd8'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=LIGHT, fontsize=9, fontweight='bold', pad=8)
    ax.tick_params(colors=LIGHT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3248')
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)

fig = plt.figure(figsize=(16, 5))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: pChEMBL distribution, active vs inactive (exact only)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(agg[agg['active']==0]['pchembl_mean'].dropna(), bins=40,
         color=BLUE, alpha=0.7, label='Inactive (exact)', density=True)
ax1.hist(agg[agg['active']==1]['pchembl_mean'].dropna(), bins=40,
         color=GREEN, alpha=0.7, label='Active', density=True)
ax1.axvline(x=THRESHOLD, color=AMBER, linestyle='--', linewidth=1.5)
style_ax(ax1, 'pChEMBL distribution\nactive vs inactive (exact)')
ax1.set_xlabel('pChEMBL mean')
ax1.set_ylabel('Density')
ax1.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 2: Final class balance including censored inactives
ax2 = fig.add_subplot(gs[0, 1])
counts = [final['active'].sum(), (final['active']==0).sum()]
labels = [f'Active\n(n={counts[0]:,})', f'Inactive\n(n={counts[1]:,})']
ax2.bar(labels, counts, color=[GREEN, BLUE], edgecolor='none', alpha=0.85, width=0.5)
style_ax(ax2, 'Final class balance\n(including censored inactives)')
ax2.set_ylabel('Number of compounds')

# Plot 3: CV distribution
ax3 = fig.add_subplot(gs[0, 2])
cv_vals = agg['cv'].dropna()
ax3.hist(cv_vals, bins=40, color=BLUE, edgecolor='none', alpha=0.85)
ax3.axvline(x=0.2, color=RED, linestyle='--', linewidth=1.5)
style_ax(ax3, 'Coefficient of variation\n(multi-assay compounds only)')
ax3.set_xlabel('CV (pChEMBL std / mean)')
ax3.set_ylabel('Count')

fig.suptitle('ChEMBL A2A — Value Curation & Aggregation (v2)',
             color='white', fontsize=11, fontweight='bold', y=1.02)

plt.savefig('eda/value_curation_plots.png', dpi=150,
            bbox_inches='tight', facecolor=DARK)
print("\nSaved: eda/value_curation_plots.png")

final.to_csv('data/processed/chembl_a2a_ki_curated.csv', index=False)
print("Saved: data/processed/chembl_a2a_ki_curated.csv")