from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.datasets import co2

OUT = Path("docs/assets/img/forecasting")
OUT.mkdir(parents=True, exist_ok=True)

weekly = co2.load_pandas().data["co2"]

# 1. Struktur waktu: indeks, frekuensi, rentang
print("rentang:", weekly.index.min().date(), "-", weekly.index.max().date(), "| jumlah baris:", len(weekly))
print("selisih antar-tanggal (hari):", weekly.index.to_series().diff().dt.days.dropna().unique().tolist())

# 2. Nilai hilang (NaN) pada deret mingguan
print("\nnilai hilang mingguan:", int(weekly.isna().sum()))
print("per dekade:", weekly[weekly.isna()].groupby(weekly[weekly.isna()].index.year // 10 * 10).size().to_dict())

# 3. Timestamp yang hilang berbeda dengan nilai NaN. Simulasi (BUATAN): hapus 3 baris, lalu deteksi.
gappy = weekly.drop(weekly.index[[100, 101, 500]])
full_index = pd.date_range(gappy.index.min(), gappy.index.max(), freq="7D")
missing_ts = full_index.difference(gappy.index)
print("\n[simulasi] timestamp hilang terdeteksi:", [d.date().isoformat() for d in missing_ts])
print("setelah asfreq -> baris:", len(gappy.asfreq("7D")), "| NaN baru:", int(gappy.asfreq("7D").isna().sum() - gappy.isna().sum()))

# 4. Agregasi ke bulanan dan penanganan bulan tanpa data
monthly_raw = weekly.resample("MS").mean()
print("\nbulan tanpa data:", [d.strftime("%Y-%m") for d in monthly_raw[monthly_raw.isna()].index])
linear = monthly_raw.interpolate(method="time")
ffill = monthly_raw.ffill()
gap = monthly_raw[monthly_raw.isna()].index[2:]  # tiga bulan beruntun 1964
print("bulan 1964 terisi -> interpolasi waktu:", linear[gap].round(2).tolist(), "| ffill:", ffill[gap].round(2).tolist())

# 5. Ringkasan pola
m = linear
annual = m.groupby(m.index.year).mean()
print("\nrata-rata 1960: %.1f | 1980: %.1f | 2000: %.1f ppm" % (annual[1960], annual[1980], annual[2000]))
print("kenaikan rata-rata per tahun 1960-1980: %.2f | 1980-2000: %.2f" % (
    (annual[1980] - annual[1960]) / 20, (annual[2000] - annual[1980]) / 20))
detr = m - m.groupby(m.index.year).transform("mean")
seasonal_profile = detr.groupby(detr.index.month).mean()
print("bulan puncak: %d (%.1f ppm) | bulan terendah: %d (%.1f ppm)" % (
    seasonal_profile.idxmax(), seasonal_profile.max(), seasonal_profile.idxmin(), seasonal_profile.min()))

# 6. Gambar
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(m, lw=1)
axes[0].set(title="CO2 bulanan di Mauna Loa", ylabel="ppm")
axes[1].bar(seasonal_profile.index, seasonal_profile.values)
axes[1].set(title="Pola musiman rata-rata (selisih terhadap rata-rata tahunan)", xlabel="bulan", ylabel="ppm")
fig.tight_layout()
fig.savefig(OUT / "co2-eda.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "co2-eda.png").as_posix())
