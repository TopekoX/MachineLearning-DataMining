from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# Data SINTETIS tidak seimbang: 2000 sampel, sekitar 5% kelas positif
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5, weights=[0.95, 0.05],
    flip_y=0.01, class_sep=0.8, random_state=SEED,
)
print("Jumlah positif: %d dari %d (%.1f%%)" % (y.sum(), len(y), 100 * y.mean()))

# 1. Mengapa stratified: jumlah positif per lipatan validasi
for name, splitter in [("KFold", KFold(5, shuffle=True, random_state=SEED)),
                       ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=SEED))]:
    counts = [int(y[val].sum()) for _, val in splitter.split(X, y)]
    print(f"{name:<16} positif per lipatan validasi: {counts}")

# 2. Bandingkan strategi pada validasi silang berstrata
models = {
    "Selalu tebak mayoritas": DummyClassifier(strategy="most_frequent"),
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "LogReg class_weight=balanced": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "Random Forest balanced": RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample", random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scoring = {  # zero_division=0: bila model tidak memprediksi positif sama sekali, precision/F1 = 0
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "average_precision": "average_precision",
}

print("\nmodel                          acc    prec   recall F1     PR-AUC")
for name, model in models.items():
    s = cross_validate(model, X, y, cv=cv, scoring=scoring)
    vals = [s[f"test_{m}"].mean() for m in scoring]
    print(f"{name:<30} " + " ".join(f"{v:.3f}" for v in vals))
