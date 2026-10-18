import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
cv = KFold(5, shuffle=True, random_state=SEED)
scoring = "neg_root_mean_squared_error"

# 1. Validasi silang tunggal: estimasi tanpa penyetelan
ridge = Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                  ("scale", StandardScaler()), ("ridge", Ridge())])
base = -cross_val_score(ridge, X_train, y_train, cv=cv, scoring=scoring).mean()
print("Ridge + polinomial, alpha=1 (bawaan): RMSE CV %.2f" % base)

# 2. Grid Search untuk alpha (skala logaritmik)
grid = GridSearchCV(ridge, {"ridge__alpha": np.logspace(-1, 4, 11)}, scoring=scoring, cv=cv)
grid.fit(X_train, y_train)
print("\nGrid Search Ridge (11 nilai alpha)")
print("  alpha terbaik: %.1f | RMSE CV: %.2f" % (grid.best_params_["ridge__alpha"], -grid.best_score_))

# 3. Random Search untuk Random Forest
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=SEED),
    {"n_estimators": randint(100, 400), "max_depth": [None, 3, 5, 8],
     "min_samples_leaf": randint(2, 20), "max_features": [0.3, 0.6, 1.0]},
    n_iter=15, scoring=scoring, cv=cv, random_state=SEED,
).fit(X_train, y_train)
print("\nRandom Search Random Forest (15 percobaan)")
print("  parameter terbaik:", rf_search.best_params_)
print("  RMSE CV: %.2f" % -rf_search.best_score_)

# 4. Nested CV: estimasi jujur untuk prosedur 'Grid Search Ridge'
nested = -cross_val_score(grid, X_train, y_train, cv=KFold(5, shuffle=True, random_state=1), scoring=scoring)
print("\nNested CV RMSE: %.2f (+/- %.2f)" % (nested.mean(), nested.std()))

# 5. Data uji, satu kali, untuk kedua model terpilih
for name, search in [("Ridge", grid), ("Random Forest", rf_search)]:
    pred = search.predict(X_test)
    print(f"Uji {name:<14} MAE {mean_absolute_error(y_test, pred):.2f} | RMSE {mean_squared_error(y_test, pred) ** 0.5:.2f}")

top = pd.DataFrame(grid.cv_results_).nsmallest(3, "rank_test_score")[["param_ridge__alpha", "mean_test_score"]]
print("\nTiga alpha teratas (RMSE CV):")
print(top.assign(rmse=-top["mean_test_score"].round(2)).drop(columns="mean_test_score").to_string(index=False))
