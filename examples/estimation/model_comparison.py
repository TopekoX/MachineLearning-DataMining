import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

models = {
    "Baseline (rata-rata)": DummyRegressor(strategy="mean"),
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge (alpha=10)": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "Lasso (alpha=0.5)": make_pipeline(StandardScaler(), Lasso(alpha=0.5)),
    "KNN (k=20)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=20)),
    "Decision Tree (depth=3)": DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED),
}

# Lipatan yang sama untuk semua model: 5 lipatan x 3 pengulangan = 15 skor per model
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

rows = []
for name, model in models.items():
    s = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows.append({"Model": name, "MAE": -s["test_mae"].mean(), "RMSE": -s["test_rmse"].mean(),
                 "R2": s["test_r2"].mean(), "Std RMSE": s["test_rmse"].std()})
results = pd.DataFrame(rows).set_index("Model")
print(results.round(2).to_string())

# Model terpilih: RMSE CV terendah; data uji dipakai satu kali
best = results["RMSE"].idxmin()
pred = models[best].fit(X_train, y_train).predict(X_test)
print("\nModel terpilih:", best)
print("Uji -> MAE %.2f | RMSE %.2f | R2 %.3f" % (
    mean_absolute_error(y_test, pred), mean_squared_error(y_test, pred) ** 0.5, r2_score(y_test, pred)))
