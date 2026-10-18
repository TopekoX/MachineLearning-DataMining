import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_utils import seasonal_naive, load_co2_monthly

warnings.filterwarnings("ignore")

y = load_co2_monthly()
H = 12
# 8 titik asal (rolling origin, expanding window). Prediksi tiap asal TIDAK menyentuh data setelah asal itu.
origins = [len(y) - H - 12 * k for k in range(7, -1, -1)]  # asal paling awal -> paling akhir
print("titik asal (data latih s.d.):", y.index[origins[0] - 1].strftime("%Y-%m"), "...", y.index[origins[-1] - 1].strftime("%Y-%m"),
      "| jumlah:", len(origins))

abs_err = {name: [] for name in ["Seasonal naive", "Holt-Winters", "SARIMA"]}
covered, width = [], []
for end in origins:
    train, actual = y.iloc[:end], y.iloc[end:end + H].to_numpy()
    abs_err["Seasonal naive"].append(np.abs(actual - seasonal_naive(train, H)))
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                              initialization_method="estimated").fit().forecast(H).to_numpy()
    abs_err["Holt-Winters"].append(np.abs(actual - hw))
    fc = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).get_forecast(H)
    abs_err["SARIMA"].append(np.abs(actual - fc.predicted_mean.to_numpy()))
    ci = fc.conf_int(alpha=0.05).to_numpy()
    covered.append((actual >= ci[:, 0]) & (actual <= ci[:, 1]))
    width.append(ci[:, 1] - ci[:, 0])

# 1. Galat menurut horizon (rata-rata atas titik asal)
print("\nMAE (ppm) menurut horizon, rata-rata %d titik asal" % len(origins))
table = pd.DataFrame({k: np.mean(v, axis=0) for k, v in abs_err.items()}, index=range(1, H + 1))
table.index.name = "horizon (bulan)"
print(table.loc[[1, 3, 6, 9, 12]].round(2).to_string())
print("rata-rata semua horizon:", table.mean().round(2).to_dict())

# 2. Interval prediksi 95% SARIMA: lebar dan cakupan empiris
cov = np.mean(covered, axis=0)
print("\nInterval prediksi 95%% SARIMA: lebar rata-rata horizon 1: %.2f | horizon 12: %.2f ppm" % (
    np.mean(width, axis=0)[0], np.mean(width, axis=0)[-1]))
print("cakupan empiris (nominal 95%%): horizon 1-4: %.2f | 5-8: %.2f | 9-12: %.2f | keseluruhan: %.2f" % (
    cov[:4].mean(), cov[4:8].mean(), cov[8:].mean(), cov.mean()))
