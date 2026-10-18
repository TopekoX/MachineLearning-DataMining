from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes

OUT = Path("docs/assets/img/estimation")
OUT.mkdir(parents=True, exist_ok=True)

data = load_diabetes(as_frame=True)
X, y = data.data, data.target

# 1. Bentuk data dan tipe
print("Ukuran fitur:", X.shape, "| fitur:", list(X.columns))
print("Ada data hilang?", bool(X.isna().any().any()), "| duplikat:", int(X.duplicated().sum()))

# 2. Target: kontinu, bukan kelas
print("\nTarget -> min %.0f | median %.0f | rata-rata %.1f | maks %.0f | std %.1f | skew %.2f"
      % (y.min(), y.median(), y.mean(), y.max(), y.std(), y.skew()))

# 3. Fitur sudah dipusatkan dan diskalakan oleh scikit-learn
print("\nRata-rata fitur (maks. nilai mutlak): %.1e" % X.mean().abs().max())
print("Jumlah kuadrat tiap fitur:", X.pow(2).sum().round(2).unique())

# 4. Korelasi fitur dengan target
corr = X.corrwith(y).sort_values(ascending=False)
print("\nKorelasi dengan target:\n", corr.round(2).to_string())

# 5. Pasangan fitur yang sangat berkorelasi (multikolinearitas)
c = X.corr().abs().where(~np.eye(X.shape[1], dtype=bool))
pair = c.stack().sort_values(ascending=False).index[0]
print("\nPasangan fitur paling berkorelasi:", pair, "| r = %.2f" % X.corr().loc[pair])

# 6. Gambar: distribusi target dan hubungan BMI dengan target
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(y, bins=25, ax=axes[0])
axes[0].set_title("Distribusi target (progres penyakit)")
axes[0].set_xlabel("target")
sns.regplot(x=X["bmi"], y=y, scatter_kws={"s": 12, "alpha": 0.5}, line_kws={"color": "C3"}, ax=axes[1])
axes[1].set_title("BMI vs target (fitur sudah diskalakan)")
fig.tight_layout()
fig.savefig(OUT / "diabetes-eda.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "diabetes-eda.png").as_posix())
