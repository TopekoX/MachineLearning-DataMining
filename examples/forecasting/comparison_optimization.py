import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_utils import mae, mase, rmse, seasonal_naive, load_co2_monthly

warnings.filterwarnings("ignore")

y = load_co2_monthly()
LAGS = [1, 2, 3, 12]


def ml_forecast(make_model, history, horizon):
    """Prediksi rekursif dengan fitur lag dari selisih bulanan + bulan (hanya masa lalu)."""
    d = history.diff()
    X = pd.DataFrame({f"lag_{k}": d.shift(k) for k in LAGS})
    X["sin"], X["cos"] = np.sin(2 * np.pi * X.index.month / 12), np.cos(2 * np.pi * X.index.month / 12)
    ok = X.notna().all(axis=1) & d.notna()
    model = make_model().fit(X[ok], d[ok])
    diffs, level, date, out = list(d.dropna()), history.iloc[-1], history.index[-1], []
    for _ in range(horizon):
        date = date + pd.offsets.MonthBegin(1)
        row = [diffs[-k] for k in LAGS] + [np.sin(2 * np.pi * date.month / 12), np.cos(2 * np.pi * date.month / 12)]
        step = model.predict(pd.DataFrame([row], columns=X.columns))[0]
        diffs.append(step)
        level += step
        out.append(level)
    return np.array(out)


def hw(trend, damped, seasonal):
    return lambda tr, h: ExponentialSmoothing(tr, trend=trend, damped_trend=damped, seasonal=seasonal, seasonal_periods=12,
                                              initialization_method="estimated").fit().forecast(h).to_numpy()


methods = {
    "Seasonal naive": lambda tr, h: seasonal_naive(tr, h),
    "Drift": lambda tr, h: tr.iloc[-1] + (tr.iloc[-1] - tr.iloc[0]) / (len(tr) - 1) * np.arange(1, h + 1),
    "HW aditif": hw("add", False, "add"),
    "HW aditif, tren teredam": hw("add", True, "add"),
    "HW musiman multiplikatif": hw("add", False, "mul"),
    "SARIMA (1,1,1)(0,1,1,12)": lambda tr, h: SARIMAX(tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(h).to_numpy(),
    "Ridge alpha=0.1 (selisih)": lambda tr, h: ml_forecast(lambda: make_pipeline(StandardScaler(), Ridge(alpha=0.1)), tr, h),
    "Ridge alpha=10 (selisih)": lambda tr, h: ml_forecast(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)), tr, h),
    "Random Forest (selisih)": lambda tr, h: ml_forecast(lambda: RandomForestRegressor(n_estimators=100, random_state=42), tr, h),
}

# Data uji: 24 bulan terakhir (disimpan). Pemilihan memakai titik asal yang prediksinya TIDAK menyentuh data uji.
n_dev_end = len(y) - 24
dev_origins = [n_dev_end - 12 - 12 * k for k in range(5, -1, -1)]  # 6 titik asal, horizon 12
print("titik asal pengembangan:", y.index[dev_origins[0] - 1].strftime("%Y-%m"), "...", y.index[dev_origins[-1] - 1].strftime("%Y-%m"))

rows = []
for name, fn in methods.items():
    maes, rmses, mases = [], [], []
    for end in dev_origins:
        tr, actual = y.iloc[:end], y.iloc[end:end + 12]
        pred = fn(tr, 12)
        maes.append(mae(actual, pred)); rmses.append(rmse(actual, pred)); mases.append(mase(actual, pred, tr))
    rows.append({"Metode": name, "MAE": np.mean(maes), "Std MAE": np.std(maes), "RMSE": np.mean(rmses), "MASE": np.mean(mases)})
table = pd.DataFrame(rows).set_index("Metode").sort_values("MAE")
print(table.round(2).to_string())

# Evaluasi akhir: SATU kali pada 24 bulan terakhir untuk metode terpilih dan seasonal naive
best = table.index[0]
train, test = y.iloc[:n_dev_end], y.iloc[n_dev_end:]
print("\nMetode terpilih (MAE pengembangan terendah):", best)
for name in [best, "Seasonal naive"]:
    pred = methods[name](train, 24)
    print(f"Uji 24 bulan  {name:<26} MAE {mae(test, pred):.2f} | RMSE {rmse(test, pred):.2f} | MASE {mase(test, pred, train):.2f}")
