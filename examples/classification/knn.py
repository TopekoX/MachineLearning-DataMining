from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 1. Pengaruh k: akurasi latih vs validasi silang
print("k    akurasi_latih  akurasi_CV")
for k in [1, 3, 5, 11, 21, 51, 101]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    train_acc = model.fit(X_train, y_train).score(X_train, y_train)
    print(f"{k:<4} {train_acc:.3f}          {cv:.3f}")

# 2. Pengaruh scaling dan pembobotan jarak (k=11)
print("\nvarian (k=11)                 akurasi_CV")
variants = {
    "tanpa scaling": KNeighborsClassifier(n_neighbors=11),
    "scaling, bobot seragam": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11)),
    "scaling, bobot jarak": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11, weights="distance")),
}
for name, m in variants.items():
    print(f"{name:<29} {cross_val_score(m, X_train, y_train, cv=5).mean():.3f}")

# 3. Evaluasi akhir model terpilih (k=11, scaling)
final = variants["scaling, bobot seragam"].fit(X_train, y_train)
print("\n", classification_report(y_test, final.predict(X_test), digits=3))
