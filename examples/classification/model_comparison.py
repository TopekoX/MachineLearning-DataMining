import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# 1. Data: Breast Cancer Wisconsin (bawaan scikit-learn), 569 sampel, 30 fitur
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print("Distribusi kelas:", y.value_counts().sort_index().to_dict())

# 2. Pisahkan data uji SEBELUM membandingkan model (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 3. Kandidat model; scaler hanya untuk model berbasis jarak/gradien
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "KNN (k=5, tanpa scaling)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "Naive Bayes (Gaussian)": GaussianNB(),
}

# 4. Validasi silang berulang pada data latih; semua model memakai lipatan yang sama
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scoring = ["accuracy", "f1", "roc_auc"]

rows = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows.append(
        {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean(),
            "Std F1": scores["test_f1"].std(),
        }
    )

results = pd.DataFrame(rows).set_index("Model")
print(results.round(3).to_string())

# 5. Evaluasi akhir: hanya SATU model terpilih, satu kali, pada data uji
best_name = results["F1"].idxmax()
print("\nModel terpilih (F1 CV tertinggi):", best_name)
best_model = models[best_name].fit(X_train, y_train)
print(classification_report(y_test, best_model.predict(X_test), digits=3))
