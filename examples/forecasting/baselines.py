import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

from forecast_utils import mae, mase, rmse, seasonal_naive, load_co2_monthly

warnings.filterwarnings("ignore")  # peringatan konvergensi statsmodels disembunyikan agar keluaran ringkas

y = load_co2_monthly()
test, valid, train = y.iloc[-24:], y.iloc[-48:-24], y.iloc[:-48]
H = len(valid)


def forecasts(history, horizon):
    """Prediksi horizon langkah dari satu titik asal, hanya memakai `history`."""
    last = history.iloc[-1]
    drift = (history.iloc[-1] - history.iloc[0]) / (len(history) - 1)
    steps = np.arange(1, horizon + 1)
    out = {
        "Naive (nilai terakhir)": np.repeat(last, horizon),
        "Rata-rata 12 bulan terakhir": np.repeat(history.iloc[-12:].mean(), horizon),
        "Drift": last + drift * steps,
        "Seasonal naive": seasonal_naive(history, horizon),
        "Simple Exp. Smoothing": SimpleExpSmoothing(history, initialization_method="estimated").fit().forecast(horizon).to_numpy(),
        "Holt (tren)": Holt(history, initialization_method="estimated").fit().forecast(horizon).to_numpy(),
        "Holt (tren teredam)": Holt(history, damped_trend=True, initialization_method="estimated").fit().forecast(horizon).to_numpy(),
        "Holt-Winters (tren+musiman)": ExponentialSmoothing(
            history, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated"
        ).fit().forecast(horizon).to_numpy(),
    }
    return out


# 1. Evaluasi pada data VALIDASI (24 bulan), dari satu titik asal: akhir data latih
preds = forecasts(train, H)
rows = [{"Metode": k, "MAE": mae(valid, v), "RMSE": rmse(valid, v), "MASE": mase(valid, v, train)}
        for k, v in preds.items()]
table = pd.DataFrame(rows).set_index("Metode").round(2)
print(table.sort_values("MASE").to_string())

# 2. Pilih berdasarkan validasi, lalu evaluasi SATU KALI pada uji dengan data latih + validasi
best = table["MASE"].idxmin()
trainval = pd.concat([train, valid])
pred_test = forecasts(trainval, len(test))[best]
sn_test = forecasts(trainval, len(test))["Seasonal naive"]
print("\nMetode terpilih (MASE validasi terendah):", best)
print("Uji   -> MAE %.2f | RMSE %.2f | MASE %.2f" % (mae(test, pred_test), rmse(test, pred_test), mase(test, pred_test, trainval)))
print("Uji (pembanding seasonal naive) -> MAE %.2f | MASE %.2f" % (mae(test, sn_test), mase(test, sn_test, trainval)))
