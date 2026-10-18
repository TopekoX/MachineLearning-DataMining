import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
cv = KFold(5, shuffle=True, random_state=SEED)


def cv_rmse(model):
    return -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()


# 1. Fitur polinomial derajat 2 membuat 10 fitur menjadi 65: model linear mudah overfitting
poly = lambda est: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), est)
n_features = poly(LinearRegression()).fit(X_train, y_train)[0].n_output_features_
print("jumlah fitur setelah polinomial derajat 2:", n_features)
print("\nLinear (10 fitur asli)        RMSE CV: %.2f" % cv_rmse(make_pipeline(StandardScaler(), LinearRegression())))
print("Linear (65 fitur polinomial)  RMSE CV: %.2f" % cv_rmse(poly(LinearRegression())))

# 2. Ridge: penalti L2 (alpha besar = koefisien menyusut kuat)
print("\nRidge alpha   RMSE CV")
for a in [0.1, 1, 10, 100, 300, 1000]:
    print(f"{a:<12} {cv_rmse(poly(Ridge(alpha=a))):.2f}")

# 3. Lasso: penalti L1 (sebagian koefisien menjadi tepat nol)
print("\nLasso alpha  RMSE CV   koefisien tidak nol (dari 65)")
for a in [0.01, 0.1, 0.5, 1, 3, 10]:
    model = poly(Lasso(alpha=a, max_iter=50000)).fit(X_train, y_train)
    nonzero = int((model[-1].coef_ != 0).sum())
    print(f"{a:<11} {cv_rmse(poly(Lasso(alpha=a, max_iter=50000))):.2f}      {nonzero}")

# 4. Evaluasi uji untuk pilihan yang wajar (satu kali)
for name, model in {
    "Linear 65 fitur": poly(LinearRegression()),
    "Ridge alpha=100": poly(Ridge(alpha=100)),
    "Lasso alpha=1": poly(Lasso(alpha=1, max_iter=50000)),
}.items():
    pred = model.fit(X_train, y_train).predict(X_test)
    print(f"{name:<16} RMSE uji: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
