"""Proyek klasifikasi end-to-end: memprediksi churn pelanggan (data SINTETIS)."""
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Pembangkit data ada di examples/fundamentals/ (lihat lesson pandas)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fundamentals"))
from customer_data import make_customer_data  # noqa: E402

SEED = 42

# 1. Data, pembersihan duplikat, dan pembagian data
df = make_customer_data().drop_duplicates().reset_index(drop=True)
X, y = df.drop(columns="churn"), df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("latih:", X_train.shape, "| uji:", X_test.shape, "| churn latih: %.3f" % y_train.mean())

# 2. Praproses (di dalam pipeline agar tidak bocor)
numeric = ["age", "tenure_months", "income", "n_orders", "avg_spend"]
categorical = ["city", "plan"]


def make_pipeline_with(clf, scale=True):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


# 3. Kandidat model, termasuk baseline
candidates = {
    "Baseline (mayoritas)": make_pipeline_with(DummyClassifier(strategy="most_frequent")),
    "Logistic Regression": make_pipeline_with(LogisticRegression(max_iter=1000)),
    "KNN (k=15)": make_pipeline_with(KNeighborsClassifier(n_neighbors=15)),
    "Random Forest": make_pipeline_with(
        RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED), scale=False),
    "SVM (RBF)": make_pipeline_with(SVC()),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
print("\nmodel                  ROC-AUC  PR-AUC   F1(ambang 0.5)")
for name, model in candidates.items():
    s = cross_validate(model, X_train, y_train, cv=cv,
                       scoring=["roc_auc", "average_precision", "f1"])
    print(f"{name:<22} {s['test_roc_auc'].mean():.3f}    {s['test_average_precision'].mean():.3f}    "
          f"{s['test_f1'].mean():.3f}")

# 4. Optimasi ringan untuk Logistic Regression (dipilih karena sederhana dan skornya
#    hampir setara dengan Random Forest; lihat pembahasan di lesson)
search = GridSearchCV(
    make_pipeline_with(LogisticRegression(max_iter=1000)),
    {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc",
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
).fit(X_train, y_train)
best = search.best_estimator_
print("\nC terbaik:", search.best_params_["clf__C"], "| ROC-AUC CV: %.3f" % search.best_score_)

# 5. Ambang keputusan dipilih dari prediksi validasi silang pada data LATIH (bukan data uji)
oof = cross_val_predict(best, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba")[:, 1]
thresholds = np.arange(0.2, 0.71, 0.05)
f1s = [f1_score(y_train, oof >= t) for t in thresholds]
threshold = float(thresholds[int(np.argmax(f1s))])
print("ambang terpilih (F1 tertinggi pada prediksi CV): %.2f (F1 CV %.3f)" % (threshold, max(f1s)))

# 6. Evaluasi akhir pada data uji, satu kali
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
print("\nROC-AUC uji: %.3f | PR-AUC uji: %.3f" % (roc_auc_score(y_test, proba), average_precision_score(y_test, proba)))
print("Confusion matrix (baris = asli, kolom = prediksi):\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["bertahan", "churn"], digits=3))

# 7. Simpan pipeline dan uji muat ulang
path = Path(tempfile.mkdtemp()) / "churn_model.joblib"
joblib.dump({"model": best, "threshold": threshold}, path)
restored = joblib.load(path)
same = np.array_equal(pred, (restored["model"].predict_proba(X_test)[:, 1] >= restored["threshold"]).astype(int))
print("model tersimpan dan prediksi identik setelah dimuat:", same)
