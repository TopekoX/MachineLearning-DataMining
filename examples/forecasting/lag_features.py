import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from forecast_utils import load_co2_monthly

y = load_co2_monthly()
dy = y.diff().rename("dy")  # target: perubahan bulanan (menghindari tren yang mendominasi)

# 1. Fitur yang HANYA memakai masa lalu: lag, statistik bergulir (digeser 1), kalender
feat = pd.DataFrame({"dy": dy})
for k in [1, 2, 3, 12]:
    feat[f"lag_{k}"] = dy.shift(k)                       # nilai k bulan lalu
feat["roll_mean_3"] = dy.shift(1).rolling(3).mean()      # rata-rata 3 bulan SEBELUM t
feat["month_sin"] = np.sin(2 * np.pi * feat.index.month / 12)
feat["month_cos"] = np.cos(2 * np.pi * feat.index.month / 12)
print("fitur:", [c for c in feat.columns if c != "dy"])
n_before = len(feat)
feat = feat.dropna()
print("baris sebelum/sesudah dropna: %d / %d (lag membuat baris awal kosong)" % (n_before, len(feat)))
print(feat.head(2).round(2).to_string())

# 2. Kebocoran: statistik bergulir yang IKUT memakai nilai saat ini
feat["roll_mean_3_LEAK"] = dy.rolling(3).mean().reindex(feat.index)  # termasuk dy_t
honest = ["lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "month_sin", "month_cos"]

# Split kronologis: 24 bulan terakhir sebagai data uji
train, test = feat.iloc[:-24], feat.iloc[-24:]


def fit_eval(columns):
    model = LinearRegression().fit(train[columns], train["dy"])
    rmse = mean_squared_error(test["dy"], model.predict(test[columns])) ** 0.5
    return rmse


print("\nRMSE prediksi perubahan bulanan (ppm) pada 24 bulan terakhir")
print("simpangan baku target uji (acuan)      : %.3f" % test["dy"].std())
print("fitur jujur (hanya masa lalu)          : %.3f" % fit_eval(honest))
print("ditambah fitur bocor (termasuk saat t) : %.3f" % fit_eval(honest + ["roll_mean_3_LEAK"]))
print("korelasi fitur bocor dengan target     : %.2f" % feat["roll_mean_3_LEAK"].corr(feat["dy"]))
print("korelasi fitur jujur dengan target     : %.2f" % feat["roll_mean_3"].corr(feat["dy"]))
