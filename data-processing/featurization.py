"""
Section 4: Featurization
=========================
Computes molecular features for all curated compounds.
Features are computed once and reused across scaffold and temporal splits.

Feature types:
  1. Morgan fingerprints (ECFP4) — radius=2, 2048 bits
     Encodes local chemical environment as a bit vector.
     Standard for QSAR modeling in drug discovery.

  2. Physicochemical descriptors
     Interpretable global molecular properties (MW, LogP, TPSA, etc.)
     Complement fingerprints by capturing global structure.

Run from project root:
    python data-processing/featurization.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/chembl_a2a_ki_curated.csv')

print("=" * 65)
print("Section 4: Featurization")
print("=" * 65)
print(f"\nCompounds to featurize: {len(df):,}")

# =============================================================================
# FEATURE 1: Morgan Fingerprints (ECFP4)
# =============================================================================
print("\n── Morgan Fingerprints (ECFP4, radius=2, 2048 bits) ────────────")

RADIUS = 2
N_BITS = 2048

def compute_morgan_fp(smiles):
    """
    Compute Morgan fingerprint as a bit vector.
    radius=2 = ECFP4 (diameter 4) — industry standard for QSAR.
    Returns zero vector if molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_BITS)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
    return np.array(fp)

fps = np.array([compute_morgan_fp(s) for s in df['canonical_smiles_clean']])

print(f"  Fingerprint matrix shape: {fps.shape}")
print(f"  Mean bit density: {fps.mean():.3f}  (fraction of bits set to 1)")
print(f"  Zero vectors (failed molecules): {(fps.sum(axis=1) == 0).sum()}")

# =============================================================================
# FEATURE 2: Physicochemical Descriptors
# =============================================================================
print("\n── Physicochemical Descriptors ─────────────────────────────────")

DESCRIPTOR_FUNCTIONS = {
    'MW':        Descriptors.MolWt,
    'LogP':      Descriptors.MolLogP,
    'TPSA':      Descriptors.TPSA,
    'HBD':       Descriptors.NumHDonors,
    'HBA':       Descriptors.NumHAcceptors,
    'RotBonds':  Descriptors.NumRotatableBonds,
    'ArRings':   Descriptors.NumAromaticRings,
    'HeavyAtoms':lambda m: m.GetNumHeavyAtoms(),
    'Complexity': Descriptors.BertzCT,
    'FractionCSP3': Descriptors.FractionCSP3,  # fraction of sp3 carbons
}

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in DESCRIPTOR_FUNCTIONS}
    return {name: fn(mol) for name, fn in DESCRIPTOR_FUNCTIONS.items()}

desc_list = [compute_descriptors(s) for s in df['canonical_smiles_clean']]
df_desc = pd.DataFrame(desc_list)

print(f"  Descriptors computed: {df_desc.shape[1]}")
print(f"  Missing values: {df_desc.isna().sum().sum()}")
print(f"\n  Descriptor statistics:")
print(df_desc.describe().round(2).to_string())

# Fill any missing descriptor values with column median
df_desc = df_desc.fillna(df_desc.median())

# =============================================================================
# COMBINE FEATURES
# =============================================================================
print("\n── Combined feature matrix ──────────────────────────────────────")

fp_cols   = [f'fp_{i}' for i in range(N_BITS)]
df_fps    = pd.DataFrame(fps, columns=fp_cols, index=df.index)
X_combined = pd.concat([df_fps, df_desc], axis=1)

print(f"  Morgan fingerprints:       {fps.shape[1]} features")
print(f"  Physicochemical descriptors: {df_desc.shape[1]} features")
print(f"  Total features:            {X_combined.shape[1]}")

# =============================================================================
# ACTIVE VS INACTIVE FEATURE COMPARISON
# =============================================================================
print("\n── Active vs Inactive descriptor comparison ────────────────────")
df_desc['active'] = df['active'].values
print(f"\n  {'Descriptor':<12} {'Inactive mean':>15} {'Active mean':>13} {'Difference':>12}")
print(f"  {'-'*55}")
for desc in DESCRIPTOR_FUNCTIONS.keys():
    inactive_mean = df_desc[df_desc['active']==0][desc].mean()
    active_mean   = df_desc[df_desc['active']==1][desc].mean()
    diff = active_mean - inactive_mean
    print(f"  {desc:<12} {inactive_mean:>15.2f} {active_mean:>13.2f} {diff:>+12.2f}")

# =============================================================================
# SAVE
# =============================================================================
np.save('data/processed/morgan_fps.npy', fps)
df_desc.drop(columns=['active']).to_csv('data/processed/descriptors.csv', index=False)
df[['canonical_smiles_clean', 'molecule_chembl_id', 'active',
    'pchembl_mean', 'data_source', 'undefined_stereo',
    'high_variance']].to_csv(
    'data/processed/compounds_meta.csv', index=False)

print(f"\n  Saved: data/processed/morgan_fps.npy      {fps.shape}")
print(f"  Saved: data/processed/descriptors.csv     {df_desc.shape}")
print(f"  Saved: data/processed/compounds_meta.csv  {df[['canonical_smiles_clean']].shape}")

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

active_mask   = df['active'].values == 1
inactive_mask = df['active'].values == 0

# Plot 1: Fingerprint bit density distribution
ax1 = fig.add_subplot(gs[0, 0])
bit_density = fps.mean(axis=0)
ax1.hist(bit_density, bins=50, color=BLUE, edgecolor='none', alpha=0.85)
style_ax(ax1, 'Morgan fingerprint bit density\n(fraction of compounds with bit set)')
ax1.set_xlabel('Bit frequency')
ax1.set_ylabel('Number of bits')

# Plot 2: LogP active vs inactive
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df_desc[inactive_mask]['LogP'], bins=40, color=BLUE,
         alpha=0.7, label='Inactive', density=True)
ax2.hist(df_desc[active_mask]['LogP'],   bins=40, color=GREEN,
         alpha=0.7, label='Active', density=True)
style_ax(ax2, 'LogP distribution\nactive vs inactive')
ax2.set_xlabel('LogP')
ax2.set_ylabel('Density')
ax2.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 3: MW active vs inactive
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df_desc[inactive_mask]['MW'], bins=40, color=BLUE,
         alpha=0.7, label='Inactive', density=True)
ax3.hist(df_desc[active_mask]['MW'],   bins=40, color=GREEN,
         alpha=0.7, label='Active', density=True)
style_ax(ax3, 'Molecular weight distribution\nactive vs inactive')
ax3.set_xlabel('MW (Da)')
ax3.set_ylabel('Density')
ax3.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 4: TPSA active vs inactive
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df_desc[inactive_mask]['TPSA'], bins=40, color=BLUE,
         alpha=0.7, label='Inactive', density=True)
ax4.hist(df_desc[active_mask]['TPSA'],   bins=40, color=GREEN,
         alpha=0.7, label='Active', density=True)
style_ax(ax4, 'TPSA distribution\nactive vs inactive')
ax4.set_xlabel('TPSA (Å²)')
ax4.set_ylabel('Density')
ax4.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 5: Descriptor correlation heatmap
ax5 = fig.add_subplot(gs[1, 1])
desc_cols = list(DESCRIPTOR_FUNCTIONS.keys())
corr = df_desc[desc_cols].corr()
im = ax5.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax5.set_xticks(range(len(desc_cols)))
ax5.set_yticks(range(len(desc_cols)))
ax5.set_xticklabels(desc_cols, rotation=45, ha='right', fontsize=6)
ax5.set_yticklabels(desc_cols, fontsize=6)
plt.colorbar(im, ax=ax5, fraction=0.046)
style_ax(ax5, 'Descriptor correlation matrix')

# Plot 6: HBD + HBA active vs inactive
ax6 = fig.add_subplot(gs[1, 2])
hbond_inactive = df_desc[inactive_mask]['HBD'] + df_desc[inactive_mask]['HBA']
hbond_active   = df_desc[active_mask]['HBD']   + df_desc[active_mask]['HBA']
ax6.hist(hbond_inactive, bins=20, color=BLUE,  alpha=0.7,
         label='Inactive', density=True)
ax6.hist(hbond_active,   bins=20, color=GREEN, alpha=0.7,
         label='Active', density=True)
style_ax(ax6, 'H-bond donors + acceptors\nactive vs inactive')
ax6.set_xlabel('Total H-bond count')
ax6.set_ylabel('Density')
ax6.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

fig.suptitle('ChEMBL A2A — Molecular Featurization',
             color='white', fontsize=11, fontweight='bold', y=0.99)

plt.savefig('eda/featurization_plots.png', dpi=150,
            bbox_inches='tight', facecolor=DARK)
print("  Saved: eda/featurization_plots.png")