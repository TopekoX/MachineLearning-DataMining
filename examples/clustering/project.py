"""Proyek segmentasi pelanggan toko online (data SINTETIS bergaya RFM)."""
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42

# 1. Data: Recency (hari sejak pembelian terakhir), Frequency (jumlah pesanan), Monetary (total belanja, ribu Rp)
rng = np.random.default_rng(SEED)
groups = {"VIP": (0.12, 10, 30, 14.0), "reguler": (0.5, 40, 6, 5.0), "berisiko pergi": (0.38, 220, 2, 3.0)}
rows = []
for name, (share, rec, freq, mon) in groups.items():
    n = int(900 * share)
    rows.append(pd.DataFrame({
        "recency": rng.gamma(4, rec / 4, n).round(0) + 1,
        "frequency": rng.poisson(freq, n) + 1,
        "monetary": rng.lognormal(np.log(mon * 100), 0.5, n).round(0),
        "group_true": name}))
df = pd.concat(rows, ignore_index=True)
# beberapa pelanggan ekstrem (misalnya reseller)
df.loc[rng.choice(len(df), 8, replace=False), "monetary"] *= 15
features = ["recency", "frequency", "monetary"]
print("jumlah pelanggan:", len(df), "| skewness:", df[features].skew().round(2).to_dict())

# 2. Praproses: log1p (data miring dan outlier) + standardisasi, semuanya di dalam pipeline
prep = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
Z = prep.fit_transform(df[features])
print("skewness setelah log:", pd.DataFrame(np.log1p(df[features])).skew().round(2).to_dict())

# 3. Memilih k: metrik internal + kestabilan bootstrap
print("\nk   silhouette  Davies-Bouldin  stabilitas(ARI)")
boot_rng = np.random.default_rng(SEED)
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    aris = []
    for _ in range(20):
        idx = np.unique(boot_rng.integers(0, len(Z), len(Z)))
        b = KMeans(n_clusters=k, n_init=10, random_state=int(boot_rng.integers(1_000_000))).fit_predict(Z[idx])
        aris.append(adjusted_rand_score(km.labels_[idx], b))
    print(f"{k:<3} {silhouette_score(Z, km.labels_):<11.3f} {davies_bouldin_score(Z, km.labels_):<15.3f} {np.mean(aris):.3f}")

# 4. Model akhir (k=3) dan pembanding algoritma lain
km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Z)
ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Z)
print("\nARI k-Means vs Ward: %.3f | ARI k-Means vs kelompok sebenarnya: %.3f" % (
    adjusted_rand_score(km.labels_, ward), adjusted_rand_score(df["group_true"], km.labels_)))

# 5. Profil dan tindakan
df["cluster"] = km.labels_
prof = df.groupby("cluster")[features].median().round(0)
prof.insert(0, "n", df.groupby("cluster").size())
names = {}
names[prof["monetary"].idxmax()] = "VIP"
names[prof["recency"].idxmax()] = "berisiko pergi"
names[next(c for c in prof.index if c not in names)] = "reguler"
prof["segmen"] = prof.index.map(names)
print("\nProfil (median):\n", prof.to_string())
actions = {"VIP": "program loyalitas dan akses awal", "reguler": "dorong frekuensi dengan promo bundel",
           "berisiko pergi": "kampanye reaktivasi"}
print("\nTindakan yang diusulkan:", {names[c]: actions[names[c]] for c in sorted(names)})

# 6. Simpan pipeline (praproses + k-Means + nama segmen) dan tempatkan pelanggan baru
final = Pipeline([("prep", prep), ("km", KMeans(n_clusters=3, n_init=10, random_state=SEED))]).fit(df[features])
path = Path(tempfile.mkdtemp()) / "rfm_segments.joblib"
joblib.dump({"pipeline": final, "names": names}, path)
loaded = joblib.load(path)
new = pd.DataFrame([[5, 25, 4200], [180, 2, 250]], columns=features)
print("\npelanggan baru:", [loaded["names"][int(c)] for c in loaded["pipeline"].predict(new)])
print("hasil identik setelah dimuat:", np.array_equal(final.predict(df[features]), loaded["pipeline"].predict(df[features])))
