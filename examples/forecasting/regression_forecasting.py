import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast_utils import mae, mase, rmse, seasonal_naive, load_co2_monthly

y = load_co2_monthly()
test, valid, train = y.iloc[-24:], y.iloc[-48:-24], y.iloc[:-48]
LAGS = [1, 2, 3, 12]


def make_row(history, date):
    """Fitur untuk bulan `date` memakai HANYA data di `history` (sebelum date)."""
    v = history.to_numpy()
    row = [v[-k] for k in LAGS]
    row += [np.sin(2 * np.pi * date.month / 12), np.cos(2 * np.pi * date.month / 12)]
    return row


def build_xy(series, use_diff):
    """Tabel fitur-target dari seluruh `series`. Target: level, atau selisih terhadap bulan lalu."""
    X, t = [], []
    for i in range(max(LAGS), len(series)):
        X.append(make_row(series.iloc[:i], series.index[i]))
        t.append(series.iloc[i] - series.iloc[i - 1] if use_diff else series.iloc[i])
    return np.array(X), np.array(t)


def recursive_forecast(model, history, horizon, use_diff):
    """Prediksi bertahap: hasil bulan ini dimasukkan sebagai lag bulan berikutnya."""
    hist = history.copy()
    out = []
    for _ in range(horizon):
        date = hist.index[-1] + pd.offsets.MonthBegin(1)
        x = np.array([make_row(hist, date)])
        p = model.predict(x)[0]
        value = hist.iloc[-1] + p if use_diff else p
        out.append(value)
        hist = pd.concat([hist, pd.Series([value], index=[date])])
    return np.array(out)


def evaluate(name, model, use_diff):
    X, t = build_xy(train, use_diff)
    model.fit(X, t)
    one_step = []  # prediksi satu langkah: lag diambil dari data AKTUAL
    full = pd.concat([train, valid])
    for i in range(len(train), len(full)):
        x = np.array([make_row(full.iloc[:i], full.index[i])])
        p = model.predict(x)[0]
        one_step.append(full.iloc[i - 1] + p if use_diff else p)
    multi = recursive_forecast(model, train, len(valid), use_diff)
    print(f"{name:<34} 1-langkah MAE {mae(valid, one_step):5.2f} | 24-langkah rekursif MAE {mae(valid, multi):5.2f} "
          f"| MASE {mase(valid, multi, train):.2f}")
    return multi


print("Validasi 24 bulan, metrik dalam ppm")
evaluate("Ridge, target level", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), use_diff=False)
evaluate("Random Forest, target level", RandomForestRegressor(n_estimators=200, random_state=42), use_diff=False)
evaluate("Ridge, target selisih", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), use_diff=True)
evaluate("Random Forest, target selisih", RandomForestRegressor(n_estimators=200, random_state=42), use_diff=True)

sn = seasonal_naive(train, 24)
print(f"{'Seasonal naive (pembanding)':<34} {'':<26}  24-langkah MAE {mae(valid, sn):5.2f} | MASE {mase(valid, sn, train):.2f}")
print("Rentang CO2 data latih: %.1f - %.1f | rentang validasi: %.1f - %.1f" % (
    train.min(), train.max(), valid.min(), valid.max()))
