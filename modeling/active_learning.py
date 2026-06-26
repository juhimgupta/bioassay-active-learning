"""
Section 6: Active Learning — Experimental Prioritization
=========================================================
Simulates an active learning loop for compound screening prioritization.

The core question: given a trained model and a pool of untested compounds,
which ones should we send to the lab next to maximize actives found
per experiment run?

Acquisition strategies compared:
  1. Random          — baseline, no model (what you'd do without ML)
  2. Exploitation    — pick highest predicted probability
  3. Exploration     — pick highest uncertainty
  4. Balanced        — weighted combination of probability + uncertainty

The scaffold test set acts as our "virtual library" of untested compounds.
We simulate 10 rounds of active learning, picking 20 compounds per round.

Run from project root:
    python modeling/active_learning.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 65)
print("Section 6: Active Learning — Experimental Prioritization")
print("=" * 65)

fps      = np.load('data/processed/morgan_fps.npy')
desc     = pd.read_csv('data/processed/descriptors.csv')
meta     = pd.read_csv('data/processed/compounds_meta.csv')
X        = np.hstack([fps, desc.values])
y        = meta['active'].values

# Load scaffold split
split_df     = pd.read_csv('data/processed/split_scaffold.csv')
train_smiles = set(split_df[split_df['split']=='train']['canonical_smiles_clean'])
test_smiles  = set(split_df[split_df['split']=='test']['canonical_smiles_clean'])

train_idx = meta[meta['canonical_smiles_clean'].isin(train_smiles)].index
test_idx  = meta[meta['canonical_smiles_clean'].isin(test_smiles)].index

X_train_init = X[train_idx]
y_train_init = y[train_idx]
X_pool       = X[test_idx]   # our "virtual library" of untested compounds
y_pool       = y[test_idx]   # true labels (unknown in real life)

print(f"\nInitial training set: {len(X_train_init):,} compounds")
print(f"Virtual library pool: {len(X_pool):,} compounds "
      f"({y_pool.mean()*100:.1f}% active)")
print(f"\nActive learning setup:")

N_ROUNDS     = 10   # number of screening rounds
BATCH_SIZE   = 20   # compounds selected per round
SEED         = 42

print(f"  Rounds:     {N_ROUNDS}")
print(f"  Batch size: {BATCH_SIZE} compounds per round")
print(f"  Total budget: {N_ROUNDS * BATCH_SIZE} experiments")

# =============================================================================
# MODEL FITTING FUNCTION
# =============================================================================
def fit_model(X_train, y_train):
    """Fit a calibrated Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=100,   # fewer trees for speed in AL loop
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    )
    model = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    model.fit(X_train, y_train)
    return model

def get_predictions(model, X_pool):
    """Get predicted probabilities and uncertainty for pool compounds."""
    prob = model.predict_proba(X_pool)[:, 1]

    # Uncertainty via tree variance
    rf_base = model.calibrated_classifiers_[0].estimator
    tree_probs = np.array([
        tree.predict_proba(X_pool)[:, 1]
        for tree in rf_base.estimators_
    ])
    uncertainty = tree_probs.std(axis=0)
    return prob, uncertainty

# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================
def acquire_random(prob, uncertainty, pool_indices, batch_size, seed):
    """Random selection — no model used."""
    np.random.seed(seed)
    return np.random.choice(pool_indices, size=batch_size, replace=False)

def acquire_exploitation(prob, uncertainty, pool_indices, batch_size, seed):
    """Greedy — pick highest predicted probability."""
    scores = prob
    top_idx = np.argsort(scores)[::-1][:batch_size]
    return pool_indices[top_idx]

def acquire_exploration(prob, uncertainty, pool_indices, batch_size, seed):
    """Uncertainty sampling — pick most uncertain compounds."""
    scores = uncertainty
    top_idx = np.argsort(scores)[::-1][:batch_size]
    return pool_indices[top_idx]

def acquire_balanced(prob, uncertainty, pool_indices, batch_size, seed):
    """
    Balanced acquisition — weighted combination of probability and uncertainty.
    Normalizes both scores to [0,1] before combining so neither dominates.
    """
    prob_norm = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    unc_norm  = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
    scores    = 0.5 * prob_norm + 0.5 * unc_norm
    top_idx   = np.argsort(scores)[::-1][:batch_size]
    return pool_indices[top_idx]

STRATEGIES = {
    'Random':      acquire_random,
    'Exploitation': acquire_exploitation,
    'Exploration':  acquire_exploration,
    'Balanced':     acquire_balanced,
}

# =============================================================================
# ACTIVE LEARNING LOOP
# =============================================================================
def run_active_learning(strategy_name, acquire_fn):
    """
    Run one full active learning simulation.
    Returns cumulative actives found per round.
    """
    print(f"\n  Running: {strategy_name}")

    # Start with initial training set
    X_train = X_train_init.copy()
    y_train = y_train_init.copy()

    # Pool of remaining unlabeled compounds (indices into X_pool/y_pool)
    pool_indices = np.arange(len(X_pool))

    cumulative_actives = []
    cumulative_tested  = []
    round_aucs         = []
    total_actives      = 0

    for round_num in range(N_ROUNDS):
        # Fit model on current training set
        model = fit_model(X_train, y_train)

        # Get predictions for remaining pool
        X_remaining = X_pool[pool_indices]
        prob, uncertainty = get_predictions(model, X_remaining)

        # Select batch using acquisition function
        selected_local = acquire_fn(
            prob, uncertainty,
            np.arange(len(pool_indices)),  # local indices
            BATCH_SIZE, SEED + round_num
        )
        selected_pool = pool_indices[selected_local]

        # Reveal true labels (simulate running the experiment)
        new_labels  = y_pool[selected_pool]
        new_actives = new_labels.sum()
        total_actives += new_actives

        cumulative_actives.append(total_actives)
        cumulative_tested.append((round_num + 1) * BATCH_SIZE)

        # Compute AUC on full pool for tracking model improvement
        all_prob, _ = get_predictions(model, X_pool)
        try:
            auc = roc_auc_score(y_pool, all_prob)
        except:
            auc = 0.5
        round_aucs.append(auc)

        print(f"    Round {round_num+1:2d}: selected {BATCH_SIZE} compounds, "
              f"{new_actives} active ({new_actives/BATCH_SIZE*100:.0f}% hit rate) | "
              f"cumulative: {total_actives} actives | AUC: {auc:.3f}")

        # Add selected compounds to training set and remove from pool
        X_train      = np.vstack([X_train, X_pool[selected_pool]])
        y_train      = np.concatenate([y_train, new_labels])
        pool_indices = np.delete(pool_indices, selected_local)

    return {
        'cumulative_actives': cumulative_actives,
        'cumulative_tested':  cumulative_tested,
        'round_aucs':         round_aucs,
        'total_actives':      total_actives,
    }

print(f"\n── Running active learning simulations ─────────────────────────")
results = {}
for name, fn in STRATEGIES.items():
    results[name] = run_active_learning(name, fn)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n── Summary after {N_ROUNDS} rounds ({N_ROUNDS*BATCH_SIZE} experiments) ──────────────")
print(f"\n  {'Strategy':<15} {'Actives found':>15} {'Hit rate':>10} {'Final AUC':>12}")
print(f"  {'-'*55}")

# Baseline: how many actives would you find by random chance?
expected_random = int(y_pool.mean() * N_ROUNDS * BATCH_SIZE)
for name, res in results.items():
    hit_rate = res['total_actives'] / (N_ROUNDS * BATCH_SIZE)
    final_auc = res['round_aucs'][-1]
    print(f"  {name:<15} {res['total_actives']:>15} {hit_rate*100:>9.1f}% {final_auc:>12.3f}")

print(f"\n  Expected actives by random chance: ~{expected_random} "
      f"({y_pool.mean()*100:.1f}% hit rate)")
print(f"  Total actives in pool: {y_pool.sum()} ({y_pool.mean()*100:.1f}%)")

# =============================================================================
# PLOTS
# =============================================================================
DARK   = '#0f1117'
PANEL  = '#1a1d27'
BLUE   = '#4c9be8'
GREEN  = '#4ecb71'
AMBER  = '#f0a500'
RED    = '#e85555'
PURPLE = '#a855f7'
LIGHT  = '#c8ccd8'

STRATEGY_COLORS = {
    'Random':       BLUE,
    'Exploitation': GREEN,
    'Exploration':  AMBER,
    'Balanced':     RED,
}

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

# Plot 1: Cumulative actives found over rounds
ax1 = fig.add_subplot(gs[0, 0])
for name, res in results.items():
    ax1.plot(res['cumulative_tested'], res['cumulative_actives'],
             color=STRATEGY_COLORS[name], linewidth=2, label=name, marker='o',
             markersize=4)
# Random chance baseline
ax1.plot([BATCH_SIZE * (i+1) for i in range(N_ROUNDS)],
         [int(y_pool.mean() * BATCH_SIZE * (i+1)) for i in range(N_ROUNDS)],
         color='white', linewidth=1, linestyle='--', alpha=0.4,
         label='Expected random')
style_ax(ax1, 'Cumulative actives found\nvs experiments run')
ax1.set_xlabel('Experiments run')
ax1.set_ylabel('Cumulative actives found')
ax1.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 2: Hit rate per round
ax2 = fig.add_subplot(gs[0, 1])
for name, res in results.items():
    actives_per_round = [res['cumulative_actives'][0]] + [
        res['cumulative_actives'][i] - res['cumulative_actives'][i-1]
        for i in range(1, N_ROUNDS)
    ]
    hit_rates = [a / BATCH_SIZE for a in actives_per_round]
    ax2.plot(range(1, N_ROUNDS+1), hit_rates,
             color=STRATEGY_COLORS[name], linewidth=2,
             label=name, marker='o', markersize=4)
ax2.axhline(y=y_pool.mean(), color='white', linestyle='--',
            alpha=0.4, linewidth=1, label='Pool hit rate')
style_ax(ax2, 'Hit rate per round\n(actives / batch size)')
ax2.set_xlabel('Round')
ax2.set_ylabel('Hit rate')
ax2.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 3: AUC over rounds
ax3 = fig.add_subplot(gs[0, 2])
for name, res in results.items():
    ax3.plot(range(1, N_ROUNDS+1), res['round_aucs'],
             color=STRATEGY_COLORS[name], linewidth=2,
             label=name, marker='o', markersize=4)
style_ax(ax3, 'Model AUC over rounds\n(evaluated on full pool)')
ax3.set_xlabel('Round')
ax3.set_ylabel('AUC-ROC')
ax3.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 4: Final round — uncertainty vs probability for remaining pool
ax4 = fig.add_subplot(gs[1, 0])
# Refit model on full initial training set to show acquisition space
model_final = fit_model(X_train_init, y_train_init)
prob_pool, unc_pool = get_predictions(model_final, X_pool)

# Compute balanced acquisition score
prob_norm = (prob_pool - prob_pool.min()) / (prob_pool.max() - prob_pool.min() + 1e-8)
unc_norm  = (unc_pool - unc_pool.min()) / (unc_pool.max() - unc_pool.min() + 1e-8)
acq_score = 0.5 * prob_norm + 0.5 * unc_norm

top20_balanced = np.argsort(acq_score)[::-1][:20]

sc = ax4.scatter(prob_pool, unc_pool, c=y_pool, cmap='RdYlGn',
                 alpha=0.4, s=15, edgecolors='none')
ax4.scatter(prob_pool[top20_balanced], unc_pool[top20_balanced],
            s=80, marker='*', color=AMBER, zorder=5,
            edgecolors='white', linewidth=0.5, label='Top 20 balanced picks')
ax4.axvline(x=0.5, color='white', linestyle='--', alpha=0.3, linewidth=1)
style_ax(ax4, 'Acquisition space — round 1\n★ = balanced strategy top 20 picks')
ax4.set_xlabel('P(active)')
ax4.set_ylabel('Uncertainty')
ax4.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)
plt.colorbar(sc, ax=ax4).set_label('True label', color=LIGHT, fontsize=7)

# Plot 5: Total actives found — bar chart comparison
ax5 = fig.add_subplot(gs[1, 1])
names   = list(results.keys())
totals  = [results[n]['total_actives'] for n in names]
colors  = [STRATEGY_COLORS[n] for n in names]
bars    = ax5.bar(names, totals, color=colors, edgecolor='none',
                  alpha=0.85, width=0.6)
ax5.axhline(y=expected_random, color='white', linestyle='--',
            alpha=0.5, linewidth=1.5)
style_ax(ax5, f'Total actives found\n({N_ROUNDS} rounds × {BATCH_SIZE} compounds)')
ax5.set_ylabel('Actives found')
for bar, val in zip(bars, totals):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             str(val), ha='center', color=LIGHT, fontsize=9, fontweight='bold')

# Plot 6: Enrichment factor per strategy
ax6 = fig.add_subplot(gs[1, 2])
enrichments = [res['total_actives'] / expected_random for res in results.values()]
bars6 = ax6.bar(names, enrichments, color=colors, edgecolor='none',
                alpha=0.85, width=0.6)
ax6.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
style_ax(ax6, 'Enrichment factor\n(actives found / expected random)')
ax6.set_ylabel('Enrichment factor')
for bar, val in zip(bars6, enrichments):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.2f}x', ha='center', color=LIGHT,
             fontsize=9, fontweight='bold')

fig.suptitle('ChEMBL A2A — Active Learning: Experimental Prioritization',
             color='white', fontsize=11, fontweight='bold', y=0.99)

plt.savefig('results/active_learning_results.png', dpi=150,
            bbox_inches='tight', facecolor=DARK)
print("\nSaved: results/active_learning_results.png")