from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


def cv_acc(model):
    return cross_val_score(model, X_train, y_train, cv=5).mean()


# 1. Kernel linear vs RBF, dengan dan tanpa scaling
print("model                      akurasi_CV")
print(f"linear + scaling           {cv_acc(make_pipeline(StandardScaler(), SVC(kernel='linear'))):.3f}")
print(f"RBF + scaling              {cv_acc(make_pipeline(StandardScaler(), SVC(kernel='rbf'))):.3f}")
print(f"RBF tanpa scaling          {cv_acc(SVC(kernel='rbf')):.3f}")

# 2. Pengaruh C dan gamma pada kernel RBF (gamma besar = pengaruh titik sangat lokal)
print("\nC      gamma   akurasi_latih  akurasi_CV")
for C, gamma in [(1, 0.001), (1, "scale"), (1, 1.0), (100, 0.001), (100, "scale"), (100, 1.0)]:
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))
    train_acc = model.fit(X_train, y_train).score(X_train, y_train)
    print(f"{C:<6} {str(gamma):<7} {train_acc:.3f}          {cv_acc(model):.3f}")

# 3. Evaluasi akhir (RBF, C=1, gamma="scale") dan jumlah support vector
final = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")).fit(X_train, y_train)
print("\njumlah support vector per kelas:", final[-1].n_support_.tolist(), "dari", len(X_train), "data latih")
print(classification_report(y_test, final.predict(X_test), digits=3))
