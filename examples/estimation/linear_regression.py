import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print("latih:", X_train.shape, "| uji:", X_test.shape)


def report(name, model):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:<22} MAE={mean_absolute_error(y_test, pred):6.2f}  RMSE={rmse:6.2f}  R2={r2_score(y_test, pred):6.3f}")


# 1. Baseline: selalu memprediksi rata-rata target data latih
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
report("Baseline (rata-rata)", baseline)

# 2. Regresi linear
lin = LinearRegression().fit(X_train, y_train)
report("Linear Regression", lin)
cv_rmse = -cross_val_score(lin, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
print("RMSE CV (5 lipatan): %.2f (+/- %.2f)" % (cv_rmse.mean(), cv_rmse.std()))

# 3. Koefisien (fitur sudah satu skala, sehingga dapat dibandingkan)
coef = pd.Series(lin.coef_, index=X.columns).sort_values()
print("\nintercept: %.1f" % lin.intercept_)
print(coef.round(1).to_string())

# 4. Pemeriksaan: persamaan normal OLS memberi koefisien yang sama
A = np.column_stack([np.ones(len(X_train)), X_train.to_numpy()])
beta = np.linalg.solve(A.T @ A, A.T @ y_train.to_numpy())
print("\nselisih maks. dengan scikit-learn: %.2e" % np.abs(beta[1:] - lin.coef_).max())
