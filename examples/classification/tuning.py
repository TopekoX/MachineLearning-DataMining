import pandas as pd
from scipy.stats import randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# 1. Grid Search untuk SVM. Nama parameter pipeline: <langkah>__<parameter>
svm = Pipeline([("scale", StandardScaler()), ("svc", SVC())])
grid = GridSearchCV(
    svm,
    param_grid={"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, "scale"]},
    scoring="f1", cv=cv,
)
grid.fit(X_train, y_train)  # pencarian hanya memakai data latih
print("Grid Search SVM")
print("  kombinasi dicoba :", len(grid.cv_results_["params"]))
print("  parameter terbaik:", grid.best_params_)
print("  F1 CV terbaik    : %.3f" % grid.best_score_)
top = pd.DataFrame(grid.cv_results_).nsmallest(3, "rank_test_score")
print(top[["param_svc__C", "param_svc__gamma", "mean_test_score", "std_test_score"]].round(3).to_string(index=False))

default_cv = cross_val_score(svm, X_train, y_train, cv=cv, scoring="f1").mean()
print("  F1 CV SVM default: %.3f" % default_cv)

# 2. Random Search untuk Random Forest (ruang parameter lebih besar)
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED),
    param_distributions={
        "n_estimators": randint(100, 400),
        "max_depth": [None, 3, 5, 10],
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", 0.5],
    },
    n_iter=15, scoring="f1", cv=cv, random_state=SEED,
)
rf_search.fit(X_train, y_train)
print("\nRandom Search Random Forest (15 percobaan)")
print("  parameter terbaik:", rf_search.best_params_)
print("  F1 CV terbaik    : %.3f" % rf_search.best_score_)

# 3. Estimasi jujur: nested cross-validation (pencarian di dalam lipatan luar)
nested = cross_val_score(grid, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="f1")
print("\nNested CV F1 untuk prosedur 'Grid Search SVM': %.3f (+/- %.3f)" % (nested.mean(), nested.std()))

# 4. Data uji dipakai SEKALI untuk model SVM terpilih
pred = grid.predict(X_test)
print("Data uji -> accuracy: %.3f | F1: %.3f" % (accuracy_score(y_test, pred), f1_score(y_test, pred)))
