import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_utils import mae, mase, rmse, load_co2_monthly

warnings.filterwarnings("ignore")  # peringatan konvergensi disembunyikan agar keluaran ringkas

y = load_co2_monthly()
test, valid, train = y.iloc[-24:], y.iloc[-48:-24], y.iloc[:-48]

# 1. Bandingkan beberapa orde dengan AIC pada data LATIH (AIC lebih kecil = lebih baik)
#    d=1 (diferensiasi tren) dan D=1 pada periode 12 (diferensiasi musiman)
candidates = [
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 0), (0, 1, 1, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
]
rows = []
for order, seasonal in candidates:
    res = SARIMAX(train, order=order, seasonal_order=seasonal).fit(disp=False)
    rows.append({"order": order, "seasonal_order": seasonal, "AIC": res.aic})
aic = pd.DataFrame(rows).sort_values("AIC")
print(aic.round(1).to_string(index=False))

# 2. Model terpilih: prediksi 24 bulan validasi beserta interval prediksi 95%
best = aic.iloc[0]
res = SARIMAX(train, order=best["order"], seasonal_order=best["seasonal_order"]).fit(disp=False)
fc = res.get_forecast(24)
pred, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
print("\nModel terpilih:", best["order"], best["seasonal_order"])
print("Validasi -> MAE %.2f | RMSE %.2f | MASE %.2f" % (mae(valid, pred), rmse(valid, pred), mase(valid, pred, train)))
inside = ((valid.to_numpy() >= ci.iloc[:, 0].to_numpy()) & (valid.to_numpy() <= ci.iloc[:, 1].to_numpy())).mean()
print("lebar interval 95%%: bulan ke-1 %.2f ppm | bulan ke-12 %.2f | bulan ke-24 %.2f" % (
    ci.iloc[0, 1] - ci.iloc[0, 0], ci.iloc[11, 1] - ci.iloc[11, 0], ci.iloc[23, 1] - ci.iloc[23, 0]))
print("proporsi nilai aktual dalam interval 95%%: %.2f (24 titik)" % inside)

# 3. Diagnostik residual: apakah masih ada autokorelasi? (uji Ljung-Box pada lag 12 dan 24)
resid = res.resid.iloc[13:]  # buang residual awal yang dipengaruhi inisialisasi diferensiasi
lb = acorr_ljungbox(resid, lags=[12, 24])
print("\nLjung-Box p-value lag 12: %.3f | lag 24: %.3f (p kecil = masih ada pola)" % (
    lb["lb_pvalue"].iloc[0], lb["lb_pvalue"].iloc[1]))
print("rata-rata residual: %.3f | std residual: %.3f" % (resid.mean(), resid.std()))
