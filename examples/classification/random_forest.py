import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 1. Satu pohon vs hutan (validasi silang 5 lipatan)
tree_cv = cross_val_score(DecisionTreeClassifier(random_state=SEED), X_train, y_train, cv=5)
print("1 pohon           : akurasi CV %.3f (+/- %.3f)" % (tree_cv.mean(), tree_cv.std()))

# 2. Jumlah pohon (n_estimators)
print("\nn_estimators  akurasi_CV")
for n in [1, 5, 25, 100, 300]:
    rf = RandomForestClassifier(n_estimators=n, random_state=SEED)
    print(f"{n:<13} {cross_val_score(rf, X_train, y_train, cv=5).mean():.3f}")

# 3. Jumlah fitur yang dipertimbangkan per pembagian (max_features)
print("\nmax_features  akurasi_CV")
for mf in ["sqrt", 0.5, 1.0]:
    rf = RandomForestClassifier(n_estimators=200, max_features=mf, random_state=SEED)
    print(f"{str(mf):<13} {cross_val_score(rf, X_train, y_train, cv=5).mean():.3f}")

# 4. Model akhir, skor out-of-bag (OOB), dan evaluasi uji
rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=SEED).fit(X_train, y_train)
print("\nOOB score: %.3f" % rf.oob_score_)
print(classification_report(y_test, rf.predict(X_test), digits=3))

# 5. Pentingnya fitur: impurity vs permutation (dihitung pada data uji)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=SEED)
perm = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("Top 3 impurity   :", list(imp.head(3).index))
print("Top 3 permutation:", list(perm.head(3).index))
