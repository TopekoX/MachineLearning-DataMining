from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 1. Pisahkan data uji lebih dulu (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("latih:", X_train.shape, "| uji:", X_test.shape)
print("proporsi kelas 1 -> latih: %.3f | uji: %.3f" % (y_train.mean(), y_test.mean()))

# 2. Baseline 1: selalu menebak kelas mayoritas
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
pred = dummy.predict(X_test)
print("\nBaseline mayoritas -> accuracy: %.3f | F1 kelas 1: %.3f"
      % (accuracy_score(y_test, pred), f1_score(y_test, pred)))

# 3. Baseline 2: model sederhana (scaling + Logistic Regression) pada validasi silang
simple = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv_f1 = cross_val_score(simple, X_train, y_train, cv=5, scoring="f1")
print("Logistic Regression (CV 5 lipatan) -> F1: %.3f (+/- %.3f)" % (cv_f1.mean(), cv_f1.std()))

# 4. Pengaruh scaling pada Logistic Regression
no_scale = LogisticRegression(max_iter=10000)
cv_raw = cross_val_score(no_scale, X_train, y_train, cv=5, scoring="f1")
print("Tanpa scaling (CV)                 -> F1: %.3f (+/- %.3f)" % (cv_raw.mean(), cv_raw.std()))
