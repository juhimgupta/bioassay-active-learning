"""
Ki distribution — with and without censoring annotation
========================================================
    python eda/plot_raw_values.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/chembl_a2a_raw.csv')
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')

ki = df[(df['standard_type'] == 'Ki') & (df['assay_type'] == 'B')].copy()
ki_exact    = ki[ki['standard_relation'] == '='].copy()
ki_censored = ki[ki['standard_relation'].isin(['>', '>='])].copy()
ki_all      = ki.copy()

log_all      = np.log10(ki_all['standard_value'].dropna() + 1e-6)
log_exact    = np.log10(ki_exact['standard_value'].dropna() + 1e-6)
log_censored = np.log10(ki_censored['standard_value'].dropna() + 1e-6)

pchembl_all   = ki_all[ki_all['pchembl_value'].notna()]['pchembl_value'].astype(float)
pchembl_exact = ki_exact[ki_exact['pchembl_value'].notna()]['pchembl_value'].astype(float)

DARK  = '#0f1117'
PANEL = '#1a1d27'
BLUE  = '#4c9be8'
RED   = '#e85555'
AMBER = '#f0a500'
LIGHT = '#c8ccd8'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=LIGHT, fontsize=9, fontweight='bold', pad=8)
    ax.tick_params(colors=LIGHT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3248')
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# ── Row 1: WITHOUT censoring annotation ─────────────────────────────────────

# Plot 1: linear Ki, all data
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(ki_all['standard_value'].dropna(), bins=100,
         color=BLUE, edgecolor='none', alpha=0.85)
style_ax(ax1, 'Ki (nM) — linear scale\nall records')
ax1.set_xlabel('Ki (nM)')
ax1.set_ylabel('Count')

# Plot 2: log Ki, all data, no split
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(log_all, bins=60, color=BLUE, alpha=0.85, edgecolor='none')
style_ax(ax2, 'Ki (nM) — log scale\nall records')
ax2.set_xlabel('log10(Ki in nM)')
ax2.set_ylabel('Count')

# Plot 3: scatter, all data, no split
ax3 = fig.add_subplot(gs[0, 2])
sample_all = ki_all.sample(min(800, len(ki_all)), random_state=42)
ax3.scatter(range(len(sample_all)),
            np.log10(sample_all['standard_value'].astype(float) + 1e-6),
            color=BLUE, alpha=0.3, s=8)
style_ax(ax3, 'Ki scatter (random sample)\nall records')
ax3.set_xlabel('Record index')
ax3.set_ylabel('log10(Ki in nM)')

# Plot 4: pChEMBL, all records
ax4 = fig.add_subplot(gs[0, 3])
ax4.hist(pchembl_all, bins=40, color=BLUE, edgecolor='none', alpha=0.85)
style_ax(ax4, 'pChEMBL distribution\nall records')
ax4.set_xlabel('pChEMBL value')
ax4.set_ylabel('Count')

# ── Row 2: WITH censoring annotation ────────────────────────────────────────

# Plot 5: linear Ki, exact vs censored
ax5 = fig.add_subplot(gs[1, 0])
ax5.hist(ki_exact['standard_value'].dropna(), bins=100,
         color=BLUE, alpha=0.7, edgecolor='none',
         label=f'Exact (n={len(ki_exact):,})')
ax5.hist(ki_censored['standard_value'].dropna(), bins=100,
         color=RED, alpha=0.7, edgecolor='none',
         label=f'Censored (n={len(ki_censored):,})')
style_ax(ax5, 'Ki (nM) — linear scale\nexact vs censored')
ax5.set_xlabel('Ki (nM)')
ax5.set_ylabel('Count')
ax5.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 6: log Ki, exact vs censored
ax6 = fig.add_subplot(gs[1, 1])
ax6.hist(log_exact,    bins=60, color=BLUE, alpha=0.7,
         label=f'Exact (n={len(ki_exact):,})', density=True)
ax6.hist(log_censored, bins=60, color=RED,  alpha=0.7,
         label=f'Censored (n={len(ki_censored):,})', density=True)
style_ax(ax6, 'Ki (nM) — log scale\nexact vs censored')
ax6.set_xlabel('log10(Ki in nM)')
ax6.set_ylabel('Density')
ax6.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 7: scatter, colored by relation
ax7 = fig.add_subplot(gs[1, 2])
sample_exact = ki_exact.sample(min(600, len(ki_exact)), random_state=42)
sample_cens  = ki_censored.sample(min(300, len(ki_censored)), random_state=42)
ax7.scatter(range(len(sample_exact)),
            np.log10(sample_exact['standard_value'].astype(float) + 1e-6),
            color=BLUE, alpha=0.3, s=8, label='Exact (=)')
ax7.scatter(range(len(sample_cens)),
            np.log10(sample_cens['standard_value'].astype(float) + 1e-6),
            color=RED, alpha=0.6, s=12, marker='_', linewidths=2,
            label='Censored (>)')
style_ax(ax7, 'Ki scatter (random sample)\nexact vs censored')
ax7.set_xlabel('Record index')
ax7.set_ylabel('log10(Ki in nM)')
ax7.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 8: pChEMBL, all vs exact only
ax8 = fig.add_subplot(gs[1, 3])
ax8.hist(pchembl_all,   bins=40, color=BLUE, alpha=0.5,
         label=f'All (n={len(pchembl_all):,})', density=True)
ax8.hist(pchembl_exact, bins=40, color=RED,  alpha=0.6,
         label=f'Exact only (n={len(pchembl_exact):,})', density=True)
style_ax(ax8, 'pChEMBL distribution\nall vs exact only')
ax8.set_xlabel('pChEMBL value')
ax8.set_ylabel('Density')
ax8.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

fig.suptitle('ChEMBL A2A — Ki distribution with and without censoring annotation\n'
             'Top row: unannotated    Bottom row: censored records highlighted',
             color='white', fontsize=11, fontweight='bold', y=0.99)

plt.savefig('eda/raw_values.png', dpi=150, bbox_inches='tight', facecolor=DARK)
print("Saved: eda/raw_values.png")