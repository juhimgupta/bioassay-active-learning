"""
Exploratory Data Analysis — ChEMBL A2A Receptor Raw Bioassay Data
=================================================================
Run after pull_chembl.py has generated chembl_a2a_raw.csv

    python eda/eda_chembl.py

Sections:
    1. Basic dataset overview
    2. Missing data audit
    3. Measurement type & assay type breakdown
    4. Activity value distributions
    5. Censored data analysis
    6. Duplicate compound analysis
    7. Activity comment & QC flag audit
    8. pChEMBL value analysis
    9. SMILES / structure audit
    10. Per-assay statistics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/chembl_a2a_raw.csv')
print("=" * 65)
print("EDA: ChEMBL A2A Receptor (CHEMBL251) Raw Bioassay Data")
print("=" * 65)


# =============================================================================
# SECTION 1: Basic overview
# =============================================================================
print("\n── SECTION 1: Basic Overview ──────────────────────────────────")
print(f"  Total records:              {len(df):,}")
print(f"  Unique compounds:           {df['molecule_chembl_id'].nunique():,}")
print(f"  Unique assays:              {df['assay_chembl_id'].nunique():,}")
print(f"  Unique documents/papers:    {df['document_chembl_id'].nunique():,}")
print(f"  Columns:                    {df.shape[1]}")
print(f"\n  Dtypes:\n{df.dtypes.to_string()}")


# =============================================================================
# SECTION 2: Missing data audit
# =============================================================================
print("\n── SECTION 2: Missing Data Audit ──────────────────────────────")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_%': missing_pct})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_%', ascending=False)
print(missing_df.to_string())
print(f"\n  Note: pchembl_value missing for {df['pchembl_value'].isna().sum():,} rows")
print(f"  → These are rows where ChEMBL couldn't compute -log10(value)")
print(f"    (usually censored values or non-standard units)")


# =============================================================================
# SECTION 3: Measurement & assay type breakdown
# =============================================================================
print("\n── SECTION 3: Measurement & Assay Type Breakdown ──────────────")

print("\n  Standard type (what was measured):")
st = df['standard_type'].value_counts()
for k, v in st.items():
    print(f"    {k:<10} {v:>6,}  ({v/len(df)*100:.1f}%)")

print("\n  Assay type (how it was measured):")
at = df['assay_type'].value_counts()
labels = {'B': 'Binding', 'F': 'Functional', 'A': 'ADMET'}
for k, v in at.items():
    print(f"    {k} ({labels.get(k,'?')})   {v:>6,}  ({v/len(df)*100:.1f}%)")

print("\n  Cross-tab: standard_type × assay_type:")
print(pd.crosstab(df['standard_type'], df['assay_type']).to_string())

print("\n  Units present (should be uniform — are they?):")
print(df['standard_units'].value_counts().to_string())


# =============================================================================
# SECTION 4: Activity value distributions
# =============================================================================
print("\n── SECTION 4: Activity Value Distributions ─────────────────────")

df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')

print("\n  Raw standard_value statistics (all types combined):")
print(df['standard_value'].describe().round(2).to_string())

print("\n  Per measurement type:")
for stype in df['standard_type'].unique():
    sub = df[df['standard_type'] == stype]['standard_value'].dropna()
    if len(sub) == 0:
        continue
    print(f"\n  {stype} (n={len(sub):,}):")
    print(f"    min={sub.min():.2f}  median={sub.median():.2f}  "
          f"mean={sub.mean():.2f}  max={sub.max():.2f}")
    # Flag suspicious ranges
    if stype in ['IC50', 'Ki', 'Kd'] and sub.max() > 1e7:
        n_extreme = (sub > 1e7).sum()
        print(f"    ⚠ {n_extreme} values > 10,000,000 nM — likely errors or cutoff values")
    if stype in ['IC50', 'Ki', 'Kd'] and sub.min() < 0.001:
        n_tiny = (sub < 0.001).sum()
        print(f"    ⚠ {n_tiny} values < 0.001 nM — suspiciously potent, check these")

print("\n  pChEMBL value statistics (rows where available):")
pchembl = df['pchembl_value'].dropna().astype(float)
print(f"    n={len(pchembl):,}  min={pchembl.min():.2f}  "
      f"median={pchembl.median():.2f}  mean={pchembl.mean():.2f}  max={pchembl.max():.2f}")
print(f"    pChEMBL >= 6 (<=1uM, considered 'active'): "
      f"{(pchembl >= 6).sum():,} ({(pchembl >= 6).mean()*100:.1f}%)")
print(f"    pChEMBL >= 7 (<=100nM, potent):            "
      f"{(pchembl >= 7).sum():,} ({(pchembl >= 7).mean()*100:.1f}%)")


# =============================================================================
# SECTION 5: Censored data analysis
# =============================================================================
print("\n── SECTION 5: Censored Data Analysis ──────────────────────────")

rel = df['standard_relation'].value_counts()
print("\n  Standard relation breakdown:")
for k, v in rel.items():
    label = {
        '=':  'exact measurement',
        '>':  'CENSORED — only a lower bound (inactive cutoff)',
        '<':  'CENSORED — only an upper bound',
        '>=': 'CENSORED — lower bound inclusive',
        '<=': 'CENSORED — upper bound inclusive',
    }.get(k, '')
    print(f"    '{k}'  {v:>5,}  ({v/len(df)*100:.1f}%)  {label}")

n_censored = df['standard_relation'].isin(['>', '<', '>=', '<=']).sum()
print(f"\n  Total censored: {n_censored:,} ({n_censored/len(df)*100:.1f}%)")
print(f"  → These CANNOT be used as continuous values.")
print(f"    Options: drop them, or treat '> X nM' as inactive label")

# What values do the censored rows cluster around?
censored_vals = df[df['standard_relation'].isin(['>', '<'])]['standard_value']
print(f"\n  Most common censored cutoff values (nM):")
print(censored_vals.value_counts().head(10).to_string())


# =============================================================================
# SECTION 6: Duplicate compound analysis
# =============================================================================
print("\n── SECTION 6: Duplicate Compound Analysis ──────────────────────")

counts = df['molecule_chembl_id'].value_counts()
print(f"\n  Compounds with exactly 1 measurement:  {(counts == 1).sum():,}")
print(f"  Compounds with 2-5 measurements:       {((counts >= 2) & (counts <= 5)).sum():,}")
print(f"  Compounds with 6-10 measurements:      {((counts >= 6) & (counts <= 10)).sum():,}")
print(f"  Compounds with >10 measurements:       {(counts > 10).sum():,}")
print(f"  Max measurements for one compound:     {counts.max()}")

# Look at the most-measured compound
top_compound = counts.idxmax()
top_rows = df[df['molecule_chembl_id'] == top_compound]
print(f"\n  Most-measured compound: {top_compound} ({counts.max()} records)")
print(f"  SMILES: {top_rows['canonical_smiles'].iloc[0]}")
print(f"  Assay types: {top_rows['assay_type'].value_counts().to_dict()}")
print(f"  Measurement types: {top_rows['standard_type'].value_counts().to_dict()}")
vals = top_rows['standard_value'].dropna().astype(float)
print(f"  Value range: {vals.min():.2f} – {vals.max():.2f} nM  (median: {vals.median():.2f})")
print(f"  → Spread of {vals.max()/vals.min():.0f}x across assays — this is the 'activity cliff' problem")

# For duplicates: how much do values vary across assays for the same compound?
print(f"\n  Value variability for compounds measured multiple times:")
dup_compounds = counts[counts > 1].index
dup_df = df[df['molecule_chembl_id'].isin(dup_compounds) &
            (df['standard_type'] == 'Ki') &
            (df['standard_relation'] == '=')].copy()
dup_df['standard_value'] = pd.to_numeric(dup_df['standard_value'], errors='coerce')

if len(dup_df) > 0:
    cv_by_compound = (dup_df.groupby('molecule_chembl_id')['standard_value']
                             .agg(lambda x: x.std() / x.mean() if len(x) > 1 else np.nan)
                             .dropna())
    print(f"    Coefficient of variation (Ki, exact measurements only):")
    print(f"    median CV = {cv_by_compound.median():.2f}  "
          f"(1.0 = std equals mean — very noisy!)")
    print(f"    Compounds with CV > 1.0: {(cv_by_compound > 1.0).sum():,}")


# =============================================================================
# SECTION 7: Activity comment & QC flag audit
# =============================================================================
print("\n── SECTION 7: Activity Comments & QC Flags ────────────────────")

print("\n  Activity comments (submitter-provided flags):")
ac = df['activity_comment'].value_counts(dropna=False)
for k, v in ac.head(15).items():
    print(f"    {str(k):<65} {v:>5,}")

# The "Inhibition < 50%" rows — these are a special case
inh_mask = df['activity_comment'].str.contains('Inhibition', na=False)
print(f"\n  'Inhibition < 50%' rows: {inh_mask.sum():,}")
print(f"  → These compounds never reached 50% inhibition at max tested conc.")
print(f"    They have no meaningful IC50/Ki — should be labeled INACTIVE")
print(f"    (currently they still have a standard_value in the data — misleading!)")

print("\n  ChEMBL data validity comments (QC flags):")
dvc = df['data_validity_comment'].value_counts(dropna=False)
for k, v in dvc.head(10).items():
    print(f"    {str(k):<40} {v:>5,}")

outside_range = df['data_validity_comment'] == 'Outside typical range'
transcription = df['data_validity_comment'] == 'Potential transcription error'
print(f"\n  'Outside typical range': {outside_range.sum()} rows — likely drop")
print(f"  'Potential transcription error': {transcription.sum()} rows — definitely drop")

# What do the outside-range values look like?
if outside_range.sum() > 0:
    otr = df[outside_range]['standard_value'].astype(float)
    print(f"\n  Outside-range value distribution:")
    print(f"    min={otr.min():.2e}  max={otr.max():.2e}  median={otr.median():.2e}")


# =============================================================================
# SECTION 8: pChEMBL value analysis
# =============================================================================
print("\n── SECTION 8: pChEMBL Value Analysis ──────────────────────────")
print("""
  pChEMBL = -log10(standard_value in Molar)
  So Ki = 100 nM → pChEMBL = -log10(100e-9) = 7.0
  Standard thresholds:
    pChEMBL >= 5  →  <= 10 µM   (weak activity)
    pChEMBL >= 6  →  <= 1 µM    (moderate, common 'active' cutoff)
    pChEMBL >= 7  →  <= 100 nM  (potent)
    pChEMBL >= 8  →  <= 10 nM   (very potent)
""")

pchembl_df = df[df['pchembl_value'].notna()].copy()
pchembl_df['pchembl_value'] = pchembl_df['pchembl_value'].astype(float)

print(f"  Rows with pChEMBL available: {len(pchembl_df):,} / {len(df):,}")
print(f"  Rows WITHOUT pChEMBL:        {df['pchembl_value'].isna().sum():,}")
print(f"  → Missing pChEMBL = ChEMBL couldn't standardize the value")
print(f"    (censored rows, non-nM units, or activity comments blocking conversion)")

print(f"\n  pChEMBL by measurement type:")
for stype in ['Ki', 'IC50', 'EC50', 'Kd']:
    sub = pchembl_df[pchembl_df['standard_type'] == stype]['pchembl_value']
    if len(sub) == 0:
        continue
    print(f"    {stype:<6} n={len(sub):>5,}  "
          f"median={sub.median():.2f}  mean={sub.mean():.2f}  "
          f"active(>=6)={( sub>=6).mean()*100:.1f}%")


# =============================================================================
# SECTION 9: SMILES / structure audit
# =============================================================================
print("\n── SECTION 9: SMILES / Structure Audit ────────────────────────")

missing_smiles = df['canonical_smiles'].isna().sum()
print(f"\n  Records with missing SMILES: {missing_smiles}")

smiles_counts = df.groupby('canonical_smiles')['molecule_chembl_id'].nunique()
conflicts = smiles_counts[smiles_counts > 1]
print(f"  SMILES mapping to >1 ChEMBL ID: {len(conflicts)}")
print(f"  → Same structure, different IDs (salts, stereoisomers, mixtures)")

# Duplicate SMILES across compounds
dup_smiles = df['canonical_smiles'].value_counts()
print(f"\n  Most common SMILES (same structure, multiple records):")
print(dup_smiles.head(5).to_string())


# =============================================================================
# SECTION 10: Per-assay statistics
# =============================================================================
print("\n── SECTION 10: Per-Assay Statistics ───────────────────────────")

assay_stats = (df.groupby('assay_chembl_id')
                 .agg(
                     n_compounds=('molecule_chembl_id', 'nunique'),
                     n_records=('molecule_chembl_id', 'count'),
                     measurement_types=('standard_type', lambda x: '/'.join(x.unique())),
                     assay_type=('assay_type', 'first'),
                     pchembl_mean=('pchembl_value', lambda x: pd.to_numeric(x, errors='coerce').mean()),
                 )
                 .sort_values('n_compounds', ascending=False)
                 .reset_index())

print(f"\n  Total unique assays: {len(assay_stats):,}")
print(f"\n  Largest assays (by compound count):")
print(assay_stats.head(10)[['assay_chembl_id', 'n_compounds', 'n_records',
                              'measurement_types', 'assay_type', 'pchembl_mean']].to_string(index=False))

print(f"\n  Assay size distribution:")
print(f"    Assays with 1 compound:      {(assay_stats['n_compounds'] == 1).sum():,}")
print(f"    Assays with 2-10 compounds:  {((assay_stats['n_compounds'] > 1) & (assay_stats['n_compounds'] <= 10)).sum():,}")
print(f"    Assays with 11-100 compounds:{((assay_stats['n_compounds'] > 10) & (assay_stats['n_compounds'] <= 100)).sum():,}")
print(f"    Assays with >100 compounds:  {(assay_stats['n_compounds'] > 100).sum():,}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n── Generating plots → eda_plots.png ───────────────────────────")

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

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. pChEMBL distribution by measurement type
ax1 = fig.add_subplot(gs[0, 0])
colors = [BLUE, GREEN, AMBER, RED]
for i, stype in enumerate(['Ki', 'IC50', 'EC50', 'Kd']):
    vals = pchembl_df[pchembl_df['standard_type'] == stype]['pchembl_value'].dropna()
    if len(vals):
        ax1.hist(vals, bins=30, alpha=0.6, color=colors[i], label=stype, density=True)
ax1.axvline(x=6, color='white', linestyle='--', alpha=0.5, linewidth=1, label='pChEMBL=6 (1µM)')
style_ax(ax1, 'pChEMBL Distribution by Measurement Type')
ax1.set_xlabel('pChEMBL value')
ax1.set_ylabel('Density')
ax1.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# 2. Records per assay (log scale)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(np.log10(assay_stats['n_compounds'] + 1), bins=30,
         color=BLUE, edgecolor='none', alpha=0.85)
style_ax(ax2, 'Assay Size Distribution\n(log10 compound count)')
ax2.set_xlabel('log10(compounds per assay)')
ax2.set_ylabel('Number of assays')

# 3. Censored vs exact by measurement type
ax3 = fig.add_subplot(gs[0, 2])
exact = df[df['standard_relation'] == '=']['standard_type'].value_counts()
censored = df[df['standard_relation'] != '=']['standard_type'].value_counts()
x = np.arange(len(exact))
ax3.bar(x - 0.2, exact.values, 0.4, label='Exact (=)', color=GREEN, alpha=0.85)
ax3.bar(x + 0.2, [censored.get(k, 0) for k in exact.index], 0.4,
        label='Censored (>, <)', color=RED, alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(exact.index)
style_ax(ax3, 'Exact vs Censored by Measurement Type')
ax3.set_ylabel('Record count')
ax3.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# 4. Measurements per compound distribution
ax4 = fig.add_subplot(gs[1, 0])
clip_counts = counts.clip(upper=20)
ax4.hist(clip_counts, bins=20, color=AMBER, edgecolor='none', alpha=0.85)
style_ax(ax4, 'Measurements per Compound\n(clipped at 20)')
ax4.set_xlabel('Number of measurements')
ax4.set_ylabel('Number of compounds')

# 5. pChEMBL by assay type
ax5 = fig.add_subplot(gs[1, 1])
for i, atype in enumerate(['B', 'F']):
    vals = pchembl_df[pchembl_df['assay_type'] == atype]['pchembl_value'].dropna()
    label = {'B': 'Binding', 'F': 'Functional'}.get(atype, atype)
    ax5.hist(vals, bins=30, alpha=0.65, color=colors[i], label=label, density=True)
ax5.axvline(x=6, color='white', linestyle='--', alpha=0.5, linewidth=1)
style_ax(ax5, 'pChEMBL: Binding vs Functional Assays')
ax5.set_xlabel('pChEMBL value')
ax5.set_ylabel('Density')
ax5.legend(fontsize=8, labelcolor=LIGHT, facecolor=PANEL)

# 6. Missing data heatmap
ax6 = fig.add_subplot(gs[1, 2])
key_cols = ['canonical_smiles', 'standard_value', 'standard_units',
            'standard_relation', 'pchembl_value', 'activity_comment',
            'data_validity_comment', 'assay_type']
missing_matrix = df[key_cols].isnull().astype(int)
im = ax6.imshow(missing_matrix.T, aspect='auto', cmap='RdYlGn_r',
                interpolation='none', vmin=0, vmax=1)
ax6.set_yticks(range(len(key_cols)))
ax6.set_yticklabels(key_cols, fontsize=6)
ax6.set_xlabel('Record index')
style_ax(ax6, 'Missing Data Map\n(red = missing)')

fig.suptitle('EDA: ChEMBL A2A Receptor (CHEMBL251) — Raw Bioassay Data',
             color='white', fontsize=12, fontweight='bold', y=0.99)

plt.savefig('eda/eda_plots.png', dpi=150, bbox_inches='tight', facecolor=DARK)
print("  Saved: eda/eda_plots.png")

print("\n" + "=" * 65)
print("EDA COMPLETE — Key decisions for curation script:")
print("=" * 65)
print("""
  Based on this EDA, the curation script will need to:

  1. DROP data_validity_comment = 'Outside typical range' or
     'Potential transcription error'

  2. SEPARATE binding (B) and functional (F) assays — don't mix

  3. PICK ONE measurement type to model (recommend: Ki from binding assays)
     OR create a unified pChEMBL column and use that

  4. HANDLE censored relations (>, <):
     - Option A: drop them entirely
     - Option B: treat '> X nM' as inactive if X >= activity threshold

  5. HANDLE 'Inhibition < 50%' activity comments:
     - Force label = INACTIVE regardless of standard_value

  6. AGGREGATE duplicates:
     - Geometric mean of pChEMBL values (standard practice)
     - Flag high-CV compounds as unreliable

  7. BINARIZE with a threshold:
     - Common choice: pChEMBL >= 6 (Ki/IC50 <= 1 µM) = active
     - Discuss why this threshold and what it means biologically
""")