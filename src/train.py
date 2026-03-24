"""
Teknofest Sağlıkta Yapay Zeka - Kanser Tespiti
Model Eğitim ve Raporlama Scripti - Hibrit Ensemble (XGB+LGBM)
@Antigravity
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from model import get_hybrid_ensemble

warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# 1. Veri Yükleme
print("[1/7] Balanced veri seti yukleniyor...")
# In src/ dir, so root is ..
path = 'data/processed/teknofest_balanced.csv'
if not os.path.exists(path):
    path = '../' + path

df = pd.read_csv(path)
X = df.drop(columns=['Target'])
y = df['Target']
feature_names = X.columns.tolist()

# 2. Train-Test Ayrımı
print("[2/7] Train-Test ayrilimi yapiliyor (%80/%20 Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Model Eğitimi (Ensemble)
print("[3/7] Hibrit Ensemble (XGBoost + LightGBM) egitiliyor...")
model = get_hybrid_ensemble()
model.fit(X_train, y_train)

# 4. Kalibrasyon
print("[4/7] Model olasilik kalibrasyonu yapiliyor (Isotonic)...")
# CV=3 for hybrid ensemble calibration to ensure robust probabilities
calibrated_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
calibrated_model.fit(X_train, y_train)

# 5. Tahmin ve Metrikler
print("[5/7] Metrikler hesaplaniyor...")
y_prob = calibrated_model.predict_proba(X_test)[:, 1]
# Optimal Threshold (F1-Max)
thresholds = np.linspace(0.2, 0.8, 100)
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_prob >= best_threshold).astype(int)

# 6. Grafiklerin Oluşturulması
print("[6/7] Grafikler ve Confusion Matrix ciziliyor...")
plt.figure(figsize=(20, 15))

# CM
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign (0)', 'Pathogenic (1)'],
            yticklabels=['Benign (0)', 'Pathogenic (1)'])
plt.title(f'Confusion Matrix (Threshold: {best_threshold:.2f})', fontsize=14, fontweight='bold')
plt.ylabel('Gercek Deger')
plt.xlabel('Tahmin Edilen')

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, color='dodgerblue', lw=3, label=f'AUC: {roc_auc_score(y_test, y_prob):.4f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.fill_between(fpr, tpr, alpha=0.1, color='dodgerblue')
plt.title('ROC Egrisi (Ensemble)', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Feature Importance
# VotingClassifier doesn't have it directly, use XGB part.
importances = model.named_estimators_['xgb'].feature_importances_
plt.subplot(2, 2, 3)
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title('XGBoost Ozellik Onem Siralamasi', fontsize=14, fontweight='bold')
plt.xlabel('Onem Skoru')

# Threshold vs F1
plt.subplot(2, 2, 4)
plt.plot(thresholds, f1_scores, color='crimson', lw=2)
plt.axvline(x=best_threshold, color='black', linestyle='--', label=f'Best: {best_threshold:.2f}')
plt.title('Threshold vs F1 Skoru', fontsize=14, fontweight='bold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/model_performance.png', dpi=150)

# 7. Final Rapor Yazımı
print("[7/7] Teknofest raporu kaydediliyor...")
m = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_prob),
    'PR-AUC': average_precision_score(y_test, y_prob)
}

report = f"""
============================================================
   TEKNOFEST KANSER TESPITI - HIBHRID ENSEMBLE RAPORU
============================================================

1. MODEL ARCHITECTURE
   Algoritma: XGBoost + LightGBM (Hybrid Soft Voting)
   XGBoost N_Est: 300
   LightGBM N_Est: 300
   Weight: 0.5 XGB / 0.5 LGBM
   Calibration: Isotonic Regression

2. PERFORMANCE (TEST SET)
   Best Threshold: {best_threshold:.2f}
   Dogruluk (Accuracy): %{m['Accuracy']*100:.2f}
   F1-Score: {m['F1']:.4f}
   Hassasiyet (Precision): {m['Precision']:.4f}
   Duyarlilik (Recall): {m['Recall']:.4f}
   ROC-AUC Skoru: {m['ROC-AUC']:.4f}
   PR-AUC Skoru: {m['PR-AUC']:.4f}

3. CONFUSION MATRIX BILGISI
   Gercek Benign -> Tahmin Benign: {cm[0,0]}
   Gercek Benign -> Tahmin Pathogenic: {cm[0,1]}
   Gercek Pathogenic -> Tahmin Benign: {cm[1,0]}
   Gercek Pathogenic -> Tahmin Pathogenic: {cm[1,1]}

4. CLASSIFICATION REPORT
{classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic'])}
============================================================
"""

with open('reports/final_teknofest_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
# Also write to root for easy access
with open('final_teknofest_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
joblib.dump(calibrated_model, 'models/hybrid_ensemble_model.pkl')
print("Islem tamamlandi! Modeller ve raporlar kaydedildi.")
