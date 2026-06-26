"""
Section 3: Dataset Splitting
==============================
Implements scaffold-aware and temporal splits for the curated A2A dataset.

Why not random splits:
  Structurally similar molecules tend to have similar activity (similar property
  principle). Random splits leak information across train/test because close
  analogs of training compounds end up in the test set, making the model look
  better than it will perform on genuinely novel chemistry.

Splits implemented:
  1. Scaffold split  — Murcko scaffolds kept intact across train/test boundary
  2. Temporal split  — train on older papers, test on newer ones (uses
                       document_chembl_id as a proxy for publication order)

Run from project root:
    python data-processing/splitting.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/chembl_a2a_ki_curated.csv')

print("=" * 65)
print("Section 3: Dataset Splitting")
print("=" * 65)
print(f"\nTotal compounds: {len(df):,}")
print(f"Active:          {df['active'].sum():,} ({df['active'].mean()*100:.1f}%)")
print(f"Inactive:        {(df['active']==0).sum():,} ({(1-df['active'].mean())*100:.1f}%)")

TEST_FRAC = 0.2
SEED      = 42

# =============================================================================
# SPLIT 1: Scaffold-aware split
# =============================================================================
print("\n── Split 1: Scaffold-aware ─────────────────────────────────────")

def get_murcko_scaffold(smiles):
    """
    Strip all side chains and return the core ring system.
    Molecules sharing a scaffold belong to the same chemical series
    and should not be split across train/test.
    """
    if pd.isna(smiles):
        return 'NO_SCAFFOLD'
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'NO_SCAFFOLD'
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return 'NO_SCAFFOLD'

# Compute scaffolds — only for exact measurement compounds (censored have no pchembl)
print("  Computing Murcko scaffolds...")
df['murcko_scaffold'] = df['canonical_smiles_clean'].apply(get_murcko_scaffold)

n_unique_scaffolds = df['murcko_scaffold'].nunique()
print(f"  Unique scaffolds: {n_unique_scaffolds:,}")

# Compounds with no ring system get their own 'NO_SCAFFOLD' group
n_no_scaffold = (df['murcko_scaffold'] == 'NO_SCAFFOLD').sum()
print(f"  Acyclic compounds (no scaffold): {n_no_scaffold:,}")

def scaffold_split(df, test_frac=0.2, seed=42):
    """
    Group compounds by Murcko scaffold, then assign whole scaffold groups
    to train or test — never splitting a group across both.

    Scaffolds are shuffled randomly then assigned greedily to test
    until the target fraction is reached.
    """
    np.random.seed(seed)

    # Group indices by scaffold
    scaffold_to_indices = df.groupby('murcko_scaffold').apply(
        lambda x: x.index.tolist()
    ).to_dict()

    # Shuffle scaffold order
    scaffolds = list(scaffold_to_indices.items())
    np.random.shuffle(scaffolds)

    target_test_size = int(len(df) * test_frac)
    test_indices, train_indices = [], []
    test_count = 0

    for scaffold, indices in scaffolds:
        if test_count < target_test_size:
            test_indices.extend(indices)
            test_count += len(indices)
        else:
            train_indices.extend(indices)

    return pd.Index(train_indices), pd.Index(test_indices)

train_scaffold_idx, test_scaffold_idx = scaffold_split(df, TEST_FRAC, SEED)

train_scaffold = df.loc[train_scaffold_idx].copy()
test_scaffold  = df.loc[test_scaffold_idx].copy()

print(f"\n  Train: {len(train_scaffold):,} compounds  "
      f"({len(train_scaffold)/len(df)*100:.1f}%)  "
      f"hit rate: {train_scaffold['active'].mean()*100:.1f}%")
print(f"  Test:  {len(test_scaffold):,} compounds  "
      f"({len(test_scaffold)/len(df)*100:.1f}%)  "
      f"hit rate: {test_scaffold['active'].mean()*100:.1f}%")

# Verify no scaffold leakage
train_scaffolds = set(train_scaffold['murcko_scaffold'])
test_scaffolds  = set(test_scaffold['murcko_scaffold'])
overlap = train_scaffolds & test_scaffolds
print(f"\n  Scaffold overlap between train and test: {len(overlap)}")
print(f"  → {'✓ No leakage' if len(overlap) == 0 else '✗ Leakage detected'}")

# Save scaffold split
train_scaffold['split'] = 'train'
test_scaffold['split']  = 'test'
scaffold_split_df = pd.concat([train_scaffold, test_scaffold])
scaffold_split_df.to_csv('data/processed/split_scaffold.csv', index=False)
print(f"\n  Saved: data/processed/split_scaffold.csv")

# =============================================================================
# SPLIT 2: Temporal split
# =============================================================================
print("\n── Split 2: Temporal ───────────────────────────────────────────")
print("""
  Note: document_chembl_id is used as a proxy for publication order.
  ChEMBL assigns document IDs sequentially so higher ID ≈ more recent paper.
  For exact publication years, enrich via the ChEMBL document API endpoint.
""")

# Censored inactives don't have assay_chembl_id — handle separately
has_assay = df['assay_chembl_id'].notna()
df_exact    = df[has_assay].copy()
df_censored = df[~has_assay].copy()

# Extract earliest document ID per compound (some have multiple assays)
# document_chembl_id format: CHEMBL1234567 — extract numeric part for ordering
def extract_doc_id(assay_ids):
    """Extract minimum document numeric ID across all assays for a compound."""
    if pd.isna(assay_ids):
        return np.nan
    # assay_chembl_id stores pipe-separated assay IDs, not doc IDs
    # use molecule_chembl_id numeric part as proxy instead
    return assay_ids

# Use molecule_chembl_id numeric portion as temporal proxy
df_exact['mol_id_num'] = (df_exact['molecule_chembl_id']
                           .str.replace('CHEMBL', '', regex=False)
                           .astype(float))

# Sort by molecule ID (proxy for when compound entered ChEMBL)
df_exact_sorted = df_exact.sort_values('mol_id_num').reset_index(drop=True)

# Cutoff: oldest (1 - test_frac) fraction goes to train
cutoff_idx  = int(len(df_exact_sorted) * (1 - TEST_FRAC))
train_temporal = df_exact_sorted.iloc[:cutoff_idx].copy()
test_temporal  = df_exact_sorted.iloc[cutoff_idx:].copy()

# Add censored inactives to train only
# (we don't know their temporal position so conservatively put in train)
train_temporal = pd.concat([train_temporal, df_censored], ignore_index=True)

print(f"  Train: {len(train_temporal):,} compounds  "
      f"hit rate: {train_temporal['active'].mean()*100:.1f}%")
print(f"  Test:  {len(test_temporal):,} compounds  "
      f"hit rate: {test_temporal['active'].mean()*100:.1f}%")
print(f"\n  Molecule ID range — train: up to {train_temporal['mol_id_num'].max():,.0f}")
print(f"  Molecule ID range — test:  {test_temporal['mol_id_num'].min():,.0f} "
      f"to {test_temporal['mol_id_num'].max():,.0f}")

# Save temporal split
train_temporal['split'] = 'train'
test_temporal['split']  = 'test'
temporal_split_df = pd.concat([train_temporal, test_temporal])
temporal_split_df.to_csv('data/processed/split_temporal.csv', index=False)
print(f"\n  Saved: data/processed/split_temporal.csv")

# =============================================================================
# COMPARISON: Random split (for reference)
# =============================================================================
print("\n── Random split (reference baseline) ───────────────────────────")
np.random.seed(SEED)
test_random_idx  = df.sample(frac=TEST_FRAC, random_state=SEED).index
train_random_idx = df.index.difference(test_random_idx)
train_random = df.loc[train_random_idx]
test_random  = df.loc[test_random_idx]

# Check scaffold overlap for random split
train_random_scaffolds = set(train_random['murcko_scaffold'])
test_random_scaffolds  = set(test_random['murcko_scaffold'])
random_overlap = train_random_scaffolds & test_random_scaffolds
print(f"  Train: {len(train_random):,}  Test: {len(test_random):,}")
print(f"  Scaffold overlap: {len(random_overlap):,} scaffolds appear in both train and test")
# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n── Split comparison summary ─────────────────────────────────────")
print(f"  {'Split':<12} {'Train':>8} {'Test':>8} {'Train hit%':>12} {'Test hit%':>11} {'Scaffold overlap':>18}")
print(f"  {'-'*65}")
print(f"  {'Random':<12} {len(train_random):>8,} {len(test_random):>8,} "
      f"{train_random['active'].mean()*100:>11.1f}% "
      f"{test_random['active'].mean()*100:>10.1f}% "
      f"{len(random_overlap):>16,}")
print(f"  {'Scaffold':<12} {len(train_scaffold):>8,} {len(test_scaffold):>8,} "
      f"{train_scaffold['active'].mean()*100:>11.1f}% "
      f"{test_scaffold['active'].mean()*100:>10.1f}% "
      f"{len(overlap):>16,}")
print(f"  {'Temporal':<12} {len(train_temporal):>8,} {len(test_temporal):>8,} "
      f"{train_temporal['active'].mean()*100:>11.1f}% "
      f"{test_temporal['active'].mean()*100:>10.1f}% "
      f"{'N/A':>16}")

# =============================================================================
# PLOTS
# =============================================================================
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

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Plot 1: Scaffold size distribution
ax1 = fig.add_subplot(gs[0, 0])
scaffold_sizes = df.groupby('murcko_scaffold').size()
ax1.hist(scaffold_sizes.clip(upper=20), bins=20,
         color=BLUE, edgecolor='none', alpha=0.85)
style_ax(ax1, 'Compounds per scaffold\n(clipped at 20)')
ax1.set_xlabel('Compounds sharing scaffold')
ax1.set_ylabel('Number of scaffolds')

# Plot 2: pChEMBL distribution — scaffold train vs test
ax2 = fig.add_subplot(gs[0, 1])
train_pchembl = train_scaffold['pchembl_mean'].dropna()
test_pchembl  = test_scaffold['pchembl_mean'].dropna()
ax2.hist(train_pchembl, bins=35, color=BLUE, alpha=0.6,
         label='Train', density=True)
ax2.hist(test_pchembl,  bins=35, color=GREEN, alpha=0.6,
         label='Test', density=True)
style_ax(ax2, 'pChEMBL distribution\nscaffold split: train vs test')
ax2.set_xlabel('pChEMBL mean')
ax2.set_ylabel('Density')
ax2.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 3: Hit rate comparison across splits
ax3 = fig.add_subplot(gs[0, 2])
splits    = ['Random\ntrain', 'Random\ntest', 'Scaffold\ntrain',
             'Scaffold\ntest', 'Temporal\ntrain', 'Temporal\ntest']
hit_rates = [train_random['active'].mean(),  test_random['active'].mean(),
             train_scaffold['active'].mean(), test_scaffold['active'].mean(),
             train_temporal['active'].mean(), test_temporal['active'].mean()]
colors    = [BLUE, BLUE, GREEN, GREEN, AMBER, AMBER]
ax3.bar(splits, hit_rates, color=colors, edgecolor='none', alpha=0.85, width=0.6)
ax3.axhline(y=df['active'].mean(), color='white', linestyle='--',
            linewidth=1, alpha=0.5)
style_ax(ax3, 'Active hit rate by split\n(dashed = overall)')
ax3.set_ylabel('Fraction active')
ax3.set_ylim(0, 1)

# Plot 4: Scaffold overlap — random vs scaffold split
ax4 = fig.add_subplot(gs[1, 0])
categories = ['Random split\noverlap', 'Scaffold split\noverlap']
overlaps   = [len(random_overlap), len(overlap)]
ax4.bar(categories, overlaps, color=[RED, GREEN], edgecolor='none',
        alpha=0.85, width=0.5)
style_ax(ax4, 'Scaffold overlap\nrandom vs scaffold split')
ax4.set_ylabel('Scaffolds in both train & test')

# Plot 5: Temporal — molecule ID distribution train vs test
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(train_temporal['mol_id_num'].dropna(), bins=40,
         color=BLUE, alpha=0.6, label='Train', density=True)
ax5.hist(test_temporal['mol_id_num'].dropna(),  bins=40,
         color=AMBER, alpha=0.6, label='Test', density=True)
style_ax(ax5, 'Molecule ID distribution\ntemporal split')
ax5.set_xlabel('Molecule ChEMBL ID (numeric)')
ax5.set_ylabel('Density')
ax5.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 6: pChEMBL — temporal train vs test
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(train_temporal['pchembl_mean'].dropna(), bins=35,
         color=BLUE, alpha=0.6, label='Train', density=True)
ax6.hist(test_temporal['pchembl_mean'].dropna(),  bins=35,
         color=AMBER, alpha=0.6, label='Test', density=True)
style_ax(ax6, 'pChEMBL distribution\ntemporal split: train vs test')
ax6.set_xlabel('pChEMBL mean')
ax6.set_ylabel('Density')
ax6.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

fig.suptitle('ChEMBL A2A — Dataset Splitting: Scaffold vs Temporal vs Random',
             color='white', fontsize=11, fontweight='bold', y=0.99)

plt.savefig('eda/splitting_plots.png', dpi=150, bbox_inches='tight', facecolor=DARK)
print("\nSaved: eda/splitting_plots.png")