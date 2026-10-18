import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_utils import mae, load_co2_monthly

warnings.filterwarnings("ignore")

y = load_co2_monthly()
train, new_data = y.iloc[:-24], y.iloc[-24:]  # 'new_data' = data yang datang SETELAH model dipasang

# 1. Latih dan simpan hasil model (objek hasil statsmodels dapat di-pickle dengan joblib)
res = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
path = Path(tempfile.mkdtemp()) / "co2_sarima.joblib"
joblib.dump(res, path)

# 2. Muat ulang dan pastikan prediksi identik
loaded = joblib.load(path)
f_orig, f_load = res.forecast(24), loaded.forecast(24)
print("prediksi identik setelah dimuat:", np.allclose(f_orig, f_load))
print("model dilatih sampai:", train.index[-1].strftime("%Y-%m"), "| prediksi mulai:", f_load.index[0].strftime("%Y-%m"))

# 3. Nilai acuan galat dari periode pengembangan (rolling origin horizon 12 sebelum data baru)
ref_maes = []
for end in range(len(train) - 12 - 60, len(train) - 11, 12):
    r = SARIMAX(y.iloc[:end], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    ref_maes.append(mae(y.iloc[end:end + 12], r.forecast(12)))
ref = float(np.mean(ref_maes))
threshold = 2 * ref
print("\nMAE acuan (rolling origin, horizon 12): %.2f ppm | ambang peringatan (2x acuan): %.2f ppm" % (ref, threshold))


# 4. Pemantauan: bandingkan prediksi dengan data aktual yang masuk (MAE 6 bulan bergulir)
def monitor(actual, forecast, label):
    err = (actual - forecast).abs()
    roll = err.rolling(6).mean().dropna()
    flag = roll > threshold
    print(f"{label:<22} MAE 6 bulan terakhir: {roll.iloc[-1]:.2f} | bulan pertama melewati ambang:",
          flag.idxmax().strftime("%Y-%m") if flag.any() else "tidak ada")


monitor(new_data, f_load, "data aktual")

# 5. Simulasi SINTETIS: perubahan struktural, aktual naik +3 ppm mulai 2000-07
shifted = new_data.copy()
shifted.loc["2000-07-01":] += 3
monitor(shifted, f_load, "aktual + pergeseran 3")
print("bias rata-rata galat (aktual - prediksi): data aktual %.2f | simulasi pergeseran %.2f" % (
    (new_data - f_load).mean(), (shifted - f_load).mean()))
