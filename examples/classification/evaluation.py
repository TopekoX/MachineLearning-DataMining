from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("docs/assets/img/classification")
OUT.mkdir(parents=True, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)

# 1. Prediksi kelas dan probabilitas
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # P(kelas 1 = jinak)

# 2. Confusion matrix (baris = label asli, kolom = prediksi; urutan kelas 0, 1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

# 3. Laporan metrik per kelas dan metrik berbasis probabilitas
print(classification_report(y_test, y_pred, target_names=["ganas (0)", "jinak (1)"], digits=3))
print("ROC-AUC           : %.3f" % roc_auc_score(y_test, y_proba))
print("PR-AUC (avg. prec): %.3f" % average_precision_score(y_test, y_proba))

# 4. Ambang keputusan: pindahkan dari 0,5 untuk menukar precision dan recall.
#    Di sini kelas "positif" yang dijaga adalah GANAS (0), jadi pakai P(ganas) = 1 - proba.
y_malignant = (y_test == 0).astype(int)
p_malignant = 1 - y_proba
print("\nAmbang P(ganas)  precision_ganas  recall_ganas  ganas_terlewat")
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred_m = (p_malignant >= t).astype(int)
    missed = int(((pred_m == 0) & (y_malignant == 1)).sum())
    print(f"{t:<16} {precision_score(y_malignant, pred_m):.3f}           "
          f"{recall_score(y_malignant, pred_m):.3f}         {missed}")

# 5. Gambar: confusion matrix, kurva ROC, kurva Precision-Recall
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["ganas", "jinak"], ax=axes[0], colorbar=False
)
axes[0].set_title("Confusion matrix")
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].set_title("Kurva ROC")
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[2])
axes[2].set_title("Kurva Precision-Recall")
fig.tight_layout()
fig.savefig(OUT / "evaluation.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "evaluation.png").as_posix())
