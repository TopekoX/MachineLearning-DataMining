import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 1. Latih dan evaluasi
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), digits=3))

# 2. Probabilitas untuk 3 sampel uji
proba = model.predict_proba(X_test.iloc[:3])[:, 1]
print("P(kelas 1) 3 sampel pertama:", proba.round(3), "| label asli:", y_test.iloc[:3].tolist())

# 3. Koefisien (fitur sudah distandardisasi, sehingga dapat dibandingkan)
coef = pd.Series(model[-1].coef_[0], index=X.columns).sort_values()
print("\nKoefisien paling negatif (mendorong ke kelas 0 = ganas):")
print(coef.head(3).round(2).to_string())
print("Koefisien paling positif (mendorong ke kelas 1 = jinak):")
print(coef.tail(3).round(2).to_string())

# 4. Pengaruh kekuatan regularisasi C (kecil = regularisasi kuat)
print("\nC        F1 (CV 5 lipatan)")
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    print(f"{C:<8} {cross_val_score(m, X_train, y_train, cv=5, scoring='f1').mean():.3f}")
