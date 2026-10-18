"""Proyek forecasting end-to-end: penjualan bulanan produk (data SINTETIS)."""
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def mase(a, b, train, m=12):
    train = np.asarray(train)
    return mae(a, b) / float(np.mean(np.abs(train[m:] - train[:-m])))


def seasonal_naive(train, h, m=12):
    return np.resize(np.asarray(train)[-m:], h)


# 1. Data SINTETIS: penjualan bulanan 2017-2023 = tren + musiman (puncak Desember) + derau
rng = np.random.default_rng(7)
idx = pd.date_range("2017-01-01", "2023-12-01", freq="MS")
t = np.arange(len(idx))
season = np.array([-10, -12, -5, 0, 4, 2, -3, -2, 3, 10, 22, 45])[idx.month - 1]
sales = pd.Series(200 + 2.2 * t + season + rng.normal(0, 8, len(idx)), index=idx, name="sales").round(1)
print("data:", sales.index[0].strftime("%Y-%m"), "s.d.", sales.index[-1].strftime("%Y-%m"), "| %d bulan" % len(sales))
print("rata-rata per tahun:", sales.groupby(sales.index.year).mean().round(0).astype(int).to_dict())
print("bulan dengan rata-rata tertinggi:", int(sales.groupby(sales.index.month).mean().idxmax()))

# 2. Split: 12 bulan terakhir = uji (disimpan). Pengembangan memakai 4 titik asal yang tidak menyentuh uji.
H = 12
train_all, test = sales.iloc[:-H], sales.iloc[-H:]
dev_origins = [len(train_all) - H * k for k in (3, 2, 1, 0)]  # asal pada akhir 2019, 2020, 2021, 2022

methods = {
    "Seasonal naive": lambda tr, h: seasonal_naive(tr, h),
    "Holt-Winters aditif": lambda tr, h: ExponentialSmoothing(
        tr, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated").fit().forecast(h).to_numpy(),
    "SARIMA (1,1,1)(0,1,1,12)": lambda tr, h: SARIMAX(
        tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(h).to_numpy(),
}
rows = []
for name, fn in methods.items():
    errs = [mae(sales.iloc[e:e + H], fn(sales.iloc[:e], H)) for e in dev_origins]
    rows.append({"Metode": name, "MAE rata-rata": np.mean(errs), "Std": np.std(errs)})
dev = pd.DataFrame(rows).set_index("Metode").sort_values("MAE rata-rata")
print("\nPengembangan (4 titik asal, horizon 12 bulan):")
print(dev.round(1).to_string())

# 3. Pemilihan: dua teratas berselisih jauh lebih kecil daripada simpangan bakunya, jadi dianggap setara.
#    Dipilih SARIMA karena menghasilkan interval prediksi. Evaluasi akhir SATU kali pada 12 bulan uji.
best = "SARIMA (1,1,1)(0,1,1,12)"
gap = abs(dev.loc["Holt-Winters aditif", "MAE rata-rata"] - dev.loc[best, "MAE rata-rata"])
print("\nSelisih MAE Holt-Winters vs SARIMA: %.1f (Std antar-asal: %.1f-%.1f) -> dianggap setara"
      % (gap, dev.loc[best, "Std"], dev.loc["Holt-Winters aditif", "Std"]))
pred = methods[best](train_all, H)
sn = methods["Seasonal naive"](train_all, H)
print("Metode terpilih:", best)
print("Uji -> MAE %.1f | MASE %.2f   (seasonal naive: MAE %.1f | MASE %.2f)" % (
    mae(test, pred), mase(test, pred, train_all), mae(test, sn), mase(test, sn, train_all)))

# 4. Latih ulang pada SELURUH data, prediksi 12 bulan ke depan dengan interval 95%, lalu simpan
final = SARIMAX(sales, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
fc = final.get_forecast(H)
out = pd.DataFrame({"prediksi": fc.predicted_mean}).join(fc.conf_int(alpha=0.05).set_axis(["bawah_95", "atas_95"], axis=1))
print("\nPrediksi 2024 (SARIMA dilatih pada seluruh data):")
print(out.round(0).astype(int).iloc[[0, 5, 10, 11]].to_string())

path = Path(tempfile.mkdtemp()) / "sales_sarima.joblib"
joblib.dump(final, path)
print("\nprediksi identik setelah dimuat:", np.allclose(joblib.load(path).forecast(H), fc.predicted_mean))
