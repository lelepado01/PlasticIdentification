"""
train_classifier.py — Local plastic classifier on AlphaEarth embeddings

Inputs (from step3_download_patches.py):
  X_embeddings.npy   (N, 64)  float32
  y_labels.npy       (N,)     int8
  meta.npy           (N,)     object

Outputs:
  plastic_classifier.joblib   — trained model, ready for inference
  results/                    — confusion matrices, feature importance plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. Load data
# =============================================================================

print("Loading data...")
X    = np.load('npy/X_embeddings.npy')
y    = np.load('npy/y_labels.npy')
meta = np.load('npy/meta.npy', allow_pickle=True)

envs  = np.array([m['environment'] for m in meta])
sites = np.array([m['site_name']   for m in meta])

print(f"  Total pixels : {len(y)}")
print(f"  Plastic (1)  : {y.sum()}")
print(f"  Clean   (0)  : {(y == 0).sum()}")
print(f"  Features     : {X.shape}")
print()

for env in ['beach', 'river', 'ocean']:
    mask = envs == env
    print(f"  {env:6s}: {mask.sum():4d} pixels  "
          f"({y[mask].sum()} plastic / {(y[mask]==0).sum()} clean)")
print()

norms = np.linalg.norm(X, axis=1)
print(f"  Embedding norm -- mean: {norms.mean():.3f}  std: {norms.std():.3f}")
print()

# =============================================================================
# 2. Define candidate models
# =============================================================================

MODELS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=4,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        ))
    ]),
}

# =============================================================================
# 3. Stratified 5-fold cross-validation
# =============================================================================

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 60)
print("Cross-validation (5-fold stratified)")
print("=" * 60)

cv_results = {}

for name, model in MODELS.items():
    print(f"\n-- {name} --")

    probs = cross_val_predict(model, X, y, cv=CV, method='predict_proba')[:, 1]
    preds = (probs >= 0.5).astype(int)

    print(classification_report(y, preds, target_names=['clean', 'plastic']))

    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y, probs)
    print(f"  ROC-AUC      : {roc_auc:.4f}")
    print(f"  Avg precision: {ap:.4f}")

    cv_results[name] = {
        'probs': probs, 'preds': preds,
        'fpr': fpr, 'tpr': tpr,
        'roc_auc': roc_auc, 'ap': ap,
    }

# =============================================================================
# 4. Pick best model by ROC-AUC, retrain on full dataset
# =============================================================================

best_name  = max(cv_results, key=lambda n: cv_results[n]['roc_auc'])
best_model = MODELS[best_name]

print(f"\n{'='*60}")
print(f"Best model: {best_name}  (ROC-AUC={cv_results[best_name]['roc_auc']:.4f})")
print(f"{'='*60}")
print("Retraining on full dataset...")
best_model.fit(X, y)
joblib.dump(best_model, 'plastic_classifier.joblib')
print("Saved -> plastic_classifier.joblib")

# =============================================================================
# 5. Plots
# =============================================================================

# 5a. ROC + PR + confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
for name, res in cv_results.items():
    ax.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (5-fold CV)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for name, res in cv_results.items():
    prec, rec, _ = precision_recall_curve(y, res['probs'])
    ax.plot(rec, prec, label=f"{name} (AP={res['ap']:.3f})")
ax.axhline(y.mean(), color='k', linestyle='--', alpha=0.4,
           label=f'Baseline ({y.mean():.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves (5-fold CV)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
cm = confusion_matrix(y, cv_results[best_name]['preds'])
ConfusionMatrixDisplay(cm, display_labels=['clean', 'plastic']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix -- {best_name}')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=150)
plt.close()
print("Saved -> results/model_comparison.png")

# 5b. Permutation feature importance
print("Computing permutation importance (may take ~1 min)...")
perm = permutation_importance(
    best_model, X, y,
    n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc'
)

top_idx = np.argsort(perm.importances_mean)[-20:][::-1]
dim_labels = [f'B{i}' for i in range(X.shape[1])]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(20), perm.importances_mean[top_idx],
       yerr=perm.importances_std[top_idx],
       color='steelblue', alpha=0.8, capsize=3)
ax.set_xticks(range(20))
ax.set_xticklabels([dim_labels[i] for i in top_idx], rotation=45, ha='right')
ax.set_ylabel('Permutation Importance (ROC-AUC drop)')
ax.set_title(f'Top 20 Embedding Dimensions -- {best_name}')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=150)
plt.close()
print("Saved -> results/feature_importance.png")

# 5c. Per-environment confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, env in zip(axes, ['beach', 'river', 'ocean']):
    mask = envs == env
    if mask.sum() == 0:
        ax.set_visible(False)
        continue
    y_env  = y[mask]
    probs  = cv_results[best_name]['probs'][mask]
    preds  = cv_results[best_name]['preds'][mask]
    cm     = confusion_matrix(y_env, preds)
    ConfusionMatrixDisplay(cm, display_labels=['clean', 'plastic']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    roc = auc(*roc_curve(y_env, probs)[:2])
    ax.set_title(f'{env.capitalize()}  (AUC={roc:.3f}, n={mask.sum()})')

plt.suptitle(f'Per-environment -- {best_name}', y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'per_environment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> results/per_environment.png")

# 5d. Probability score distribution
fig, ax = plt.subplots(figsize=(8, 4))
probs_best = cv_results[best_name]['probs']
ax.hist(probs_best[y == 0], bins=40, alpha=0.6, color='royalblue',
        label='Clean (0)', density=True)
ax.hist(probs_best[y == 1], bins=40, alpha=0.6, color='tomato',
        label='Plastic (1)', density=True)
ax.axvline(0.5, color='k',      linestyle='--', alpha=0.6, label='Threshold 0.5')
ax.axvline(0.8, color='orange', linestyle='--', alpha=0.6, label='Threshold 0.8')
ax.set_xlabel('P(plastic)')
ax.set_ylabel('Density')
ax.set_title(f'Predicted Probability Distribution -- {best_name}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'probability_distribution.png', dpi=150)
plt.close()
print("Saved -> results/probability_distribution.png")

# =============================================================================
# 6. Summary
# =============================================================================

print(f"""
{'='*60}
Training Complete
{'='*60}
Best model : {best_name}
ROC-AUC    : {cv_results[best_name]['roc_auc']:.4f}  (5-fold CV)
Avg Prec   : {cv_results[best_name]['ap']:.4f}  (5-fold CV)

Saved:
  plastic_classifier.joblib
  results/model_comparison.png
  results/feature_importance.png
  results/per_environment.png
  results/probability_distribution.png

Inference on new embeddings:
  import numpy as np, joblib
  clf   = joblib.load('plastic_classifier.joblib')
  X_new = np.load('X_new.npy')            # shape (N, 64)
  proba = clf.predict_proba(X_new)[:, 1]  # P(plastic) per pixel
  flags = proba > 0.8                     # high-confidence detections
""")
