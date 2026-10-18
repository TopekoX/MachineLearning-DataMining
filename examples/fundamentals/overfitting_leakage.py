import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# ---------- 1. Overfitting pada klasifikasi: kedalaman pohon ----------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

print("max_depth  akurasi_latih  akurasi_uji")
for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X_train, y_train)
    print(f"{str(depth):<9}  {tree.score(X_train, y_train):.3f}          {tree.score(X_test, y_test):.3f}")

# ---------- 2. Underfitting vs overfitting pada regresi ----------
# Data SINTETIS: y = sin(2*pi*x) + noise, hanya 30 titik latih
rng = np.random.default_rng(SEED)
x_all = rng.uniform(0, 1, 130)
y_all = np.sin(2 * np.pi * x_all) + rng.normal(0, 0.3, 130)
x_tr, y_tr = x_all[:30].reshape(-1, 1), y_all[:30]
x_te, y_te = x_all[30:].reshape(-1, 1), y_all[30:]

print("\nderajat  R2_latih  R2_uji")
for degree in [1, 4, 15]:
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    model.fit(x_tr, y_tr)
    print(f"{degree:<7}  {model.score(x_tr, y_tr):.3f}     {model.score(x_te, y_te):.3f}")

# ---------- 3. Kebocoran data: seleksi fitur di luar validasi silang ----------
# Data SINTETIS: 100 sampel, 2000 fitur acak, label acak -> tidak ada pola nyata.
rng = np.random.default_rng(SEED)
X_noise = rng.normal(size=(100, 2000))
y_noise = rng.integers(0, 2, size=100)

# SALAH: memilih 20 fitur "terbaik" memakai SELURUH data, baru validasi silang
X_selected = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y_noise, cv=5).mean()

# BENAR: seleksi fitur di dalam Pipeline, sehingga dilatih ulang pada tiap lipatan
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
correct = cross_val_score(pipe, X_noise, y_noise, cv=5).mean()

print("\nAkurasi CV pada data acak murni (harapan teoretis: sekitar 0.50)")
print("seleksi di luar CV (bocor)   : %.3f" % leaky)
print("seleksi di dalam Pipeline    : %.3f" % correct)
