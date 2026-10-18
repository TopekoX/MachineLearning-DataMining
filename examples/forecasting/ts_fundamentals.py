from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

OUT = Path("docs/assets/img/forecasting")
OUT.mkdir(parents=True, exist_ok=True)

# Deret SINTETIS bulanan: komponennya diketahui sehingga hasil dekomposisi dapat diperiksa
rng = np.random.default_rng(42)
idx = pd.date_range("2015-01-01", periods=96, freq="MS")
t = np.arange(96)
trend_true = 100 + 1.5 * t                                   # naik 1,5 per bulan
season_true = 12 * np.sin(2 * np.pi * (idx.month - 1) / 12)  # amplitudo 12, periode 12 bulan
noise_true = rng.normal(0, 3, 96)                            # derau, simpangan baku 3
y = pd.Series(trend_true + season_true + noise_true, index=idx, name="sales")

print(y.head(3).round(1).to_string())
print("frekuensi indeks:", y.index.freqstr, "| jumlah:", len(y))

# 1. Dekomposisi aditif: y = tren + musiman + sisa
res = seasonal_decompose(y, model="additive", period=12)
slope = np.polyfit(t[6:-6], res.trend.dropna(), 1)[0]  # 6 titik di tiap ujung hilang pada trend
print("\nkemiringan tren terestimasi: %.2f (sebenarnya 1.50)" % slope)
print("amplitudo musiman terestimasi: %.1f (sebenarnya 12.0)" % res.seasonal.max())
print("simpangan baku sisa: %.2f (derau sebenarnya 3.00)" % res.resid.std())

# 2. Autokorelasi: korelasi deret dengan dirinya sendiri pada lag tertentu
r = acf(y, nlags=12)
print("\nACF lag 1: %.2f | lag 6: %.2f | lag 12: %.2f" % (r[1], r[6], r[12]))

# 3. Diferensiasi menghilangkan tren dan musiman
d1 = y.diff().dropna()                  # selisih 1 bulan: menghilangkan tren
d12 = y.diff(12).dropna()               # selisih 12 bulan: menghilangkan musiman
print("\nstd deret asli: %.1f | std setelah diff(1): %.1f | std setelah diff(12): %.1f"
      % (y.std(), d1.std(), d12.std()))
print("rata-rata diff(12): %.1f (tren 1,5/bulan x 12 = 18)" % d12.mean())

# 4. Gambar komponen
fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
for ax, (name, series) in zip(axes, [("Asli", y), ("Tren", res.trend), ("Musiman", res.seasonal), ("Sisa", res.resid)]):
    ax.plot(series, lw=1.2)
    ax.set_ylabel(name)
fig.tight_layout()
fig.savefig(OUT / "decomposition.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "decomposition.png").as_posix())
