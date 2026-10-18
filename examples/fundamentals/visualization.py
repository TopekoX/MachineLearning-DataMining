from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine

from customer_data import make_customer_data

OUT = Path("docs/assets/img/fundamentals")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

df = make_customer_data()
wine = load_wine(as_frame=True).frame  # kolom 'target' = kelas anggur (0,1,2)

# 1. Histogram: bentuk distribusi. Outlier meratakan skala sumbu-x,
#    sehingga transformasi log membantu melihat bentuk sebenarnya.
income_juta = df["income"].dropna() / 1e6
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
sns.histplot(income_juta, bins=30, ax=axes[0])
axes[0].set_title("Skala asli")
axes[0].set_xlabel("income (juta Rp/bulan)")
sns.histplot(np.log10(income_juta), bins=30, ax=axes[1])
axes[1].set_title("Setelah log10")
axes[1].set_xlabel("log10(income dalam juta Rp)")
fig.tight_layout()
fig.savefig(OUT / "histogram-income.png", dpi=130)

# 2. Boxplot: perbandingan distribusi antar kelompok + outlier
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.boxplot(data=wine, x="target", y="alcohol", ax=ax)
ax.set_title("Kadar alkohol menurut kelas anggur")
fig.tight_layout()
fig.savefig(OUT / "boxplot-alcohol.png", dpi=130)

# 3. Scatter plot: hubungan dua variabel numerik
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.scatterplot(data=wine, x="alcohol", y="flavanoids", hue="target",
                palette="deep", ax=ax)
ax.set_title("Alkohol vs flavanoid")
fig.tight_layout()
fig.savefig(OUT / "scatter-wine.png", dpi=130)

# 4. Heatmap korelasi
fig, ax = plt.subplots(figsize=(6, 4.5))
cols = ["alcohol", "malic_acid", "flavanoids", "total_phenols", "color_intensity", "proline"]
sns.heatmap(wine[cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Korelasi antar fitur (wine)")
fig.tight_layout()
fig.savefig(OUT / "heatmap-wine.png", dpi=130)

print("Gambar tersimpan:", sorted(p.name for p in OUT.glob("*.png")))
print("Korelasi alcohol-flavanoids: %.2f" % wine["alcohol"].corr(wine["flavanoids"]))
print("Skewness income: %.2f" % df["income"].skew())
