from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from segment_data import make_customer_segments

SEED = 42
OUT = Path("docs/assets/img/clustering")
OUT.mkdir(parents=True, exist_ok=True)

df = make_customer_segments()
features = ["annual_spend", "visits_per_month", "avg_basket", "recency_days", "age"]
skewed = ["annual_spend", "recency_days"]  # miring ke kanan -> log

# 1. Pipeline: log pada fitur miring, standardisasi, k-Means (k=4)
prep = ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), skewed)],
    remainder="passthrough",
)
pipe = Pipeline([("prep", prep), ("scale", StandardScaler()),
                 ("km", KMeans(n_clusters=4, n_init=10, random_state=SEED))])
pipe.fit(df[features])
labels = pipe.named_steps["km"].labels_
Zt = pipe[:-1].transform(df[features])
print("silhouette: %.3f | ARI terhadap kelompok sebenarnya: %.3f" % (
    silhouette_score(Zt, labels), adjusted_rand_score(df["segment_true"], labels)))

# 2. Profil klaster dalam satuan ASLI
df["cluster"] = labels
profile = df.groupby("cluster")[features].mean().round(1)
profile.insert(0, "n", df.groupby("cluster").size())
profile.insert(1, "n%", (profile["n"] / len(df) * 100).round(0))
print("\nProfil klaster (rata-rata):\n", profile.to_string())

# 3. Indeks terhadap rata-rata keseluruhan (100 = rata-rata semua pelanggan)
index = (profile[features] / df[features].mean() * 100).round(0).astype(int)
print("\nIndeks terhadap rata-rata keseluruhan:\n", index.to_string())

# 4. Penamaan berbasis aturan dari profil (nama sebenarnya diputuskan manusia)
names = {}
remaining = set(profile.index)
inactive = profile.loc[list(remaining), "recency_days"].idxmax(); names[inactive] = "tidak aktif"; remaining.discard(inactive)
loyal = profile.loc[list(remaining), "annual_spend"].idxmax(); names[loyal] = "loyal"; remaining.discard(loyal)
bargain = profile.loc[list(remaining), "visits_per_month"].idxmax(); names[bargain] = "pemburu diskon"; remaining.discard(bargain)
names[remaining.pop()] = "musiman"
print("\nNama segmen:", {int(k): v for k, v in sorted(names.items())})

# 5. Kesesuaian dengan kelompok sebenarnya (hanya mungkin pada data sintetis)
print("\nCrosstab klaster vs kelompok sebenarnya:")
print(pd.crosstab(df["cluster"].map(names), df["segment_true"]).to_string())

# 6. Gambar
fig, ax = plt.subplots(figsize=(6.5, 4.2))
sc = ax.scatter(df["visits_per_month"], df["annual_spend"], c=df["cluster"], cmap="viridis", s=14)
ax.set(xlabel="kunjungan per bulan", ylabel="belanja tahunan (juta Rp)", yscale="log", title="Segmen pelanggan")
fig.tight_layout()
fig.savefig(OUT / "segments.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "segments.png").as_posix())
