"""
Section 5: Modeling — Random Forest Baseline
=============================================
Trains a Random Forest classifier on the curated A2A dataset.
Evaluates on both scaffold and temporal splits.

Primary metric: AUC-ROC (measures ranking quality across all thresholds)
Secondary metrics: precision, recall, F1 at 0.5 threshold

Run from project root:
    python modeling/random_forest.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 65)
print("Section 5: Random Forest Baseline")
print("=" * 65)

fps      = np.load('data/processed/morgan_fps.npy')
desc     = pd.read_csv('data/processed/descriptors.csv')
meta     = pd.read_csv('data/processed/compounds_meta.csv')

# Combine features
X = np.hstack([fps, desc.values])
y = meta['active'].values

print(f"\nFeature matrix: {X.shape}")
print(f"Labels:         {y.shape}  ({y.mean()*100:.1f}% active)")

def load_split(split_csv, meta, X, y):
    """Load train/test indices from a split CSV."""
    split_df = pd.read_csv(split_csv)
    train_smiles = set(split_df[split_df['split']=='train']['canonical_smiles_clean'])
    test_smiles  = set(split_df[split_df['split']=='test']['canonical_smiles_clean'])

    train_idx = meta[meta['canonical_smiles_clean'].isin(train_smiles)].index
    test_idx  = meta[meta['canonical_smiles_clean'].isin(test_smiles)].index

    return (X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            train_idx, test_idx)

# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================
def train_and_evaluate(X_train, y_train, X_test, y_test, split_name):
    """
    Train a calibrated Random Forest and evaluate on test set.

    Calibration: raw RF probabilities tend to be overconfident
    (pushed toward 0 and 1). CalibratedClassifierCV corrects this
    so predicted probabilities are meaningful for ranking and
    uncertainty estimation.

    class_weight='balanced': compensates for class imbalance by
    weighting minority class samples more heavily during training.
    """
    print(f"\n── {split_name} ─────────────────────────────────────────────")
    print(f"  Train: {len(X_train):,}  ({y_train.mean()*100:.1f}% active)")
    print(f"  Test:  {len(X_test):,}   ({y_test.mean()*100:.1f}% active)")

    # Base Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        max_features='sqrt',      # standard for classification
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Calibrate probabilities using cross-validation
    model = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    model.fit(X_train, y_train)

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  AUC-ROC: {auc:.3f}")
    print(f"\n  Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred,
                                target_names=['Inactive', 'Active'],
                                zero_division=0))

    # Uncertainty via tree variance
    rf_base = model.calibrated_classifiers_[0].estimator
    tree_probs = np.array([
        tree.predict_proba(X_test)[:, 1]
        for tree in rf_base.estimators_
    ])
    uncertainty = tree_probs.std(axis=0)

    print(f"  Mean uncertainty: {uncertainty.mean():.3f}")
    print(f"  High uncertainty compounds (>0.3): {(uncertainty > 0.3).sum():,}")

    return model, y_prob, y_pred, uncertainty, auc

# Train on both splits
(X_train_s, y_train_s, X_test_s, y_test_s,
 train_idx_s, test_idx_s) = load_split(
    'data/processed/split_scaffold.csv', meta, X, y)

(X_train_t, y_train_t, X_test_t, y_test_t,
 train_idx_t, test_idx_t) = load_split(
    'data/processed/split_temporal.csv', meta, X, y)

model_s, prob_s, pred_s, unc_s, auc_s = train_and_evaluate(
    X_train_s, y_train_s, X_test_s, y_test_s, 'Scaffold Split')

model_t, prob_t, pred_t, unc_t, auc_t = train_and_evaluate(
    X_train_t, y_train_t, X_test_t, y_test_t, 'Temporal Split')

print(f"\n── AUC comparison ───────────────────────────────────────────────")
print(f"  Scaffold split AUC: {auc_s:.3f}")
print(f"  Temporal split AUC: {auc_t:.3f}")
print(f"  → Difference reflects distribution shift over time")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print(f"\n── Feature importance (scaffold split) ─────────────────────────")

# Get feature importances from the underlying RF
rf_base_s = model_s.calibrated_classifiers_[0].estimator
importances = rf_base_s.feature_importances_

# Descriptor importances (last 10 features)
desc_names = list(desc.columns)
desc_importances = importances[-len(desc_names):]
fp_importances   = importances[:-len(desc_names)]

print(f"\n  Physicochemical descriptor importances:")
for name, imp in sorted(zip(desc_names, desc_importances),
                         key=lambda x: x[1], reverse=True):
    print(f"    {name:<15} {imp:.4f}")

print(f"\n  Top 10 fingerprint bit importances:")
top_fp_idx = np.argsort(fp_importances)[::-1][:10]
for i, idx in enumerate(top_fp_idx):
    print(f"    fp_{idx:<6} {fp_importances[idx]:.4f}")

print(f"\n  Total importance — fingerprints: {fp_importances.sum():.3f}  "
      f"descriptors: {desc_importances.sum():.3f}")

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================
# Save scaffold split predictions for active learning script
scaffold_test_meta = meta.loc[test_idx_s].copy()
scaffold_test_meta['predicted_prob'] = prob_s
scaffold_test_meta['predicted_label'] = pred_s
scaffold_test_meta['uncertainty'] = unc_s
scaffold_test_meta.to_csv('data/processed/scaffold_test_predictions.csv', index=False)
print(f"\nSaved: data/processed/scaffold_test_predictions.csv")

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

# Plot 1: ROC curves for both splits
ax1 = fig.add_subplot(gs[0, 0])
fpr_s, tpr_s, _ = roc_curve(y_test_s, prob_s)
fpr_t, tpr_t, _ = roc_curve(y_test_t, prob_t)
ax1.plot(fpr_s, tpr_s, color=GREEN, linewidth=1.5,
         label=f'Scaffold (AUC={auc_s:.3f})')
ax1.plot(fpr_t, tpr_t, color=AMBER, linewidth=1.5,
         label=f'Temporal (AUC={auc_t:.3f})')
ax1.plot([0,1],[0,1], color='white', linestyle='--', alpha=0.3, linewidth=1)
style_ax(ax1, 'ROC curves')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 2: Predicted probability distribution — scaffold split
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(prob_s[y_test_s==0], bins=40, color=BLUE,
         alpha=0.7, label='Inactive', density=True)
ax2.hist(prob_s[y_test_s==1], bins=40, color=GREEN,
         alpha=0.7, label='Active', density=True)
ax2.axvline(x=0.5, color='white', linestyle='--', alpha=0.4, linewidth=1)
style_ax(ax2, 'Predicted probabilities\nscaffold split test set')
ax2.set_xlabel('P(active)')
ax2.set_ylabel('Density')
ax2.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 3: Predicted probability distribution — temporal split
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(prob_t[y_test_t==0], bins=40, color=BLUE,
         alpha=0.7, label='Inactive', density=True)
ax3.hist(prob_t[y_test_t==1], bins=40, color=GREEN,
         alpha=0.7, label='Active', density=True)
ax3.axvline(x=0.5, color='white', linestyle='--', alpha=0.4, linewidth=1)
style_ax(ax3, 'Predicted probabilities\ntemporal split test set')
ax3.set_xlabel('P(active)')
ax3.set_ylabel('Density')
ax3.legend(fontsize=7, labelcolor=LIGHT, facecolor=PANEL)

# Plot 4: Uncertainty vs predicted probability — scaffold split
ax4 = fig.add_subplot(gs[1, 0])
sc = ax4.scatter(prob_s, unc_s, c=y_test_s, cmap='RdYlGn',
                 alpha=0.5, s=15, edgecolors='none')
ax4.axvline(x=0.5, color='white', linestyle='--', alpha=0.3, linewidth=1)
style_ax(ax4, 'Uncertainty vs predicted probability\nscaffold split')
ax4.set_xlabel('P(active)')
ax4.set_ylabel('Uncertainty (std across trees)')
plt.colorbar(sc, ax=ax4).set_label('True label', color=LIGHT, fontsize=7)

# Plot 5: Descriptor importances
ax5 = fig.add_subplot(gs[1, 1])
sorted_idx = np.argsort(desc_importances)
ax5.barh([desc_names[i] for i in sorted_idx],
         [desc_importances[i] for i in sorted_idx],
         color=BLUE, edgecolor='none', alpha=0.85)
style_ax(ax5, 'Descriptor feature importances\n(scaffold split)')
ax5.set_xlabel('Importance')

# Plot 6: Confusion matrix — scaffold split
ax6 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test_s, pred_s)
im = ax6.imshow(cm, cmap='Blues')
ax6.set_xticks([0, 1])
ax6.set_yticks([0, 1])
ax6.set_xticklabels(['Pred Inactive', 'Pred Active'], fontsize=7)
ax6.set_yticklabels(['True Inactive', 'True Active'], fontsize=7)
for i in range(2):
    for j in range(2):
        ax6.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')
style_ax(ax6, 'Confusion matrix\nscaffold split test set')

fig.suptitle('ChEMBL A2A — Random Forest Baseline',
             color='white', fontsize=11, fontweight='bold', y=0.99)

plt.savefig('results/random_forest_results.png', dpi=150,
            bbox_inches='tight', facecolor=DARK)
print("Saved: results/random_forest_results.png")