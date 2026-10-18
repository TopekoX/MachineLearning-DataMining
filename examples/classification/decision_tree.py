import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 1. Pengaruh max_depth (pohon tidak memerlukan scaling)
print("max_depth  akurasi_latih  akurasi_CV  jumlah_daun")
for depth in [1, 2, 3, 4, 6, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    cv = cross_val_score(tree, X_train, y_train, cv=5).mean()
    tree.fit(X_train, y_train)
    print(f"{str(depth):<9}  {tree.score(X_train, y_train):.3f}          {cv:.3f}       {tree.get_n_leaves()}")

# 2. Pengaruh min_samples_leaf pada pohon tanpa batas kedalaman
print("\nmin_samples_leaf  akurasi_CV")
for leaf in [1, 5, 10, 20]:
    tree = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=SEED)
    print(f"{leaf:<16}  {cross_val_score(tree, X_train, y_train, cv=5).mean():.3f}")

# 3. Pohon kecil yang dapat dibaca manusia (max_depth=3)
small = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)
print("\n" + export_text(small, feature_names=list(X.columns)))

# 4. Pentingnya fitur (berdasarkan penurunan impurity) dan evaluasi uji
imp = pd.Series(small.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(3).round(3).to_string())
print(classification_report(y_test, small.predict(X_test), digits=3))
