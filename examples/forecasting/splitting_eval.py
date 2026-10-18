import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from forecast_utils import mae, mase, rmse, seasonal_naive, smape, load_co2_monthly

y = load_co2_monthly()

# 1. Split kronologis: latih -> validasi -> uji, tanpa diacak
test, valid, train = y.iloc[-24:], y.iloc[-48:-24], y.iloc[:-48]
for name, part in [("latih", train), ("validasi", valid), ("uji", test)]:
    print(f"{name:<9}: {part.index[0].strftime('%Y-%m')} s.d. {part.index[-1].strftime('%Y-%m')} ({len(part)} bulan)")

# 2. Mengapa tidak boleh diacak: model berbasis pohon pada fitur lag
lags = pd.DataFrame({f"lag_{k}": y.shift(k) for k in [1, 2, 3, 12]})
data = lags.assign(target=y).dropna()
X_all, y_all = data.drop(columns="target"), data["target"]


def rf():
    return RandomForestRegressor(n_estimators=200, random_state=42)


# (a) validasi silang acak (SALAH untuk deret waktu)
errs = []
for tr, te in KFold(5, shuffle=True, random_state=42).split(X_all):
    errs.append(rmse(y_all.iloc[te], rf().fit(X_all.iloc[tr], y_all.iloc[tr]).predict(X_all.iloc[te])))
print("\nRMSE Random Forest, KFold acak          : %.2f ppm" % np.mean(errs))

# (b) evaluasi kronologis: latih sebelum 1998, uji 24 bulan setelahnya (prediksi satu langkah)
cut = len(X_all) - 48
model = rf().fit(X_all.iloc[:cut], y_all.iloc[:cut])
pred = model.predict(X_all.iloc[cut:cut + 24])
print("RMSE Random Forest, kronologis (validasi): %.2f ppm" % rmse(y_all.iloc[cut:cut + 24], pred))
print("Rentang target latih: %.1f - %.1f | rentang target validasi: %.1f - %.1f" % (
    y_all.iloc[:cut].min(), y_all.iloc[:cut].max(), y_all.iloc[cut:cut + 24].min(), y_all.iloc[cut:cut + 24].max()))

# 3. Metrik pada prediksi seasonal naive untuk data validasi
pred_sn = seasonal_naive(train, 24)
print("\nSeasonal naive pada validasi -> MAE %.2f | RMSE %.2f | sMAPE %.3f%% | MASE %.2f" % (
    mae(valid, pred_sn), rmse(valid, pred_sn), smape(valid, pred_sn), mase(valid, pred_sn, train)))

# 4. Rolling-origin (expanding window): prediksi 12 bulan ke depan dari beberapa titik asal
print("\nRolling origin, seasonal naive, horizon 12 bulan")
print("asal (data latih s.d.)   MAE")
maes = []
for end in range(len(y) - 12 - 5 * 12, len(y) - 11, 12):  # 6 titik asal, mundur tiap 12 bulan
    tr, te = y.iloc[:end], y.iloc[end:end + 12]
    m = mae(te, seasonal_naive(tr, 12))
    maes.append(m)
    print(f"{tr.index[-1].strftime('%Y-%m')}                  {m:.2f}")
print("rata-rata MAE: %.2f (+/- %.2f)" % (np.mean(maes), np.std(maes)))
