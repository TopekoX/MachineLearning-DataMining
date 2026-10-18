"""Proyek regresi end-to-end: memperkirakan harga rumah (data SINTETIS)."""
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


def make_housing_data(n=800, seed=42):
    """Harga rumah (juta Rupiah) SINTETIS dengan hubungan multiplikatif + derau."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(30, 300, n)                                   # m2
    bedrooms = np.clip(np.round(area / 50 + rng.normal(0, 0.7, n)), 1, 6)
    age = rng.integers(0, 40, n).astype(float)                       # tahun
    distance = rng.gamma(2.0, 4.0, n)                                # km ke pusat kota
    district = rng.choice(["Pusat", "Barat", "Timur", "Selatan"], n, p=[0.2, 0.3, 0.3, 0.2])
    condition = rng.choice(["baik", "sedang", "kurang"], n, p=[0.4, 0.4, 0.2])
    d_eff = pd.Series(district).map({"Pusat": 0.35, "Barat": 0.1, "Timur": 0.0, "Selatan": 0.15}).to_numpy()
    c_eff = pd.Series(condition).map({"baik": 0.1, "sedang": 0.0, "kurang": -0.15}).to_numpy()
    log_price = (2.0 + 0.9 * np.log(area) - 0.012 * age - 0.03 * distance + d_eff + c_eff
                 + rng.normal(0, 0.2, n))
    df = pd.DataFrame({"area_m2": area.round(1), "bedrooms": bedrooms, "age_years": age,
                       "distance_km": distance.round(1), "district": district,
                       "condition": condition, "price": np.exp(log_price).round(1)})
    df.loc[rng.choice(n, int(0.05 * n), replace=False), "age_years"] = np.nan
    df.loc[rng.choice(n, int(0.04 * n), replace=False), "condition"] = np.nan
    return df


# 1. Data dan EDA singkat
df = make_housing_data()
print("Ukuran:", df.shape, "| hilang:", df.isna().sum()[lambda s: s > 0].to_dict())
print("Harga (juta Rp): median %.0f | rata-rata %.0f | maks %.0f | skew %.2f"
      % (df["price"].median(), df["price"].mean(), df["price"].max(), df["price"].skew()))
print("skew log(harga): %.2f" % np.log(df["price"]).skew())

X, y = df.drop(columns="price"), df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# 2. Praproses di dalam pipeline
numeric = ["area_m2", "bedrooms", "age_years", "distance_km"]
categorical = ["district", "condition"]
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])


def with_log_target(est):
    return TransformedTargetRegressor(Pipeline([("prep", prep), ("model", est)]),
                                      func=np.log, inverse_func=np.exp)


candidates = {
    "Baseline (rata-rata)": Pipeline([("prep", prep), ("model", DummyRegressor())]),
    "Ridge, target asli": Pipeline([("prep", prep), ("model", Ridge(alpha=1))]),
    "Ridge, target log": with_log_target(Ridge(alpha=1)),
    "Random Forest, target asli": Pipeline([("prep", prep), ("model", RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=SEED))]),
    "Random Forest, target log": with_log_target(RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=SEED)),
}

# 3. Perbandingan dengan validasi silang pada data latih
cv = KFold(5, shuffle=True, random_state=SEED)
print("\nmodel                       MAE     RMSE    R2")
for name, model in candidates.items():
    s = cross_validate(model, X_train, y_train, cv=cv,
                       scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    print(f"{name:<27} {-s['test_neg_mean_absolute_error'].mean():<7.1f} "
          f"{-s['test_neg_root_mean_squared_error'].mean():<7.1f} {s['test_r2'].mean():.3f}")

# 4. Optimasi ringan Ridge pada target log
search = GridSearchCV(with_log_target(Ridge()), {"regressor__model__alpha": [0.01, 0.1, 1, 10, 100]},
                      scoring="neg_root_mean_squared_error", cv=cv).fit(X_train, y_train)
best = search.best_estimator_
print("\nalpha terbaik:", search.best_params_["regressor__model__alpha"], "| RMSE CV: %.1f" % -search.best_score_)

# 5. Evaluasi akhir pada data uji (satu kali) dan galat per kelompok
pred = best.predict(X_test)
print("\nUji -> MAE %.1f | RMSE %.1f | R2 %.3f (satuan: juta Rp)" % (
    mean_absolute_error(y_test, pred), mean_squared_error(y_test, pred) ** 0.5, r2_score(y_test, pred)))
err = pd.DataFrame({"district": X_test["district"], "abs_err": (y_test - pred).abs(),
                    "rel_err_pct": (y_test - pred).abs() / y_test * 100})
print(err.groupby("district").agg(MAE=("abs_err", "mean"), rel_err_pct=("rel_err_pct", "mean")).round(1))

# 6. Simpan dan uji muat ulang
path = Path(tempfile.mkdtemp()) / "house_price_model.joblib"
joblib.dump(best, path)
print("prediksi identik setelah dimuat:", np.allclose(pred, joblib.load(path).predict(X_test)))
