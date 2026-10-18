from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("docs/assets/img/clustering")
OUT.mkdir(parents=True, exist_ok=True)

X, y = load_wine(return_X_y=True, as_frame=True)  # y: kultivar, hanya untuk validasi eksternal
scaler = StandardScaler().fit(X)
Z = scaler.transform(X)

# 1. k-Means dengan k=3
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=SEED).fit(Z)
print("ukuran klaster:", np.bincount(km.labels_).tolist(), "| inertia: %.1f" % km.inertia_)
print("ARI terhadap kultivar: %.3f" % adjusted_rand_score(y, km.labels_))

# 2. Pusat klaster dalam satuan asli (inverse transform)
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns)
print("\nPusat klaster (satuan asli):\n", centers[["alcohol", "flavanoids", "proline"]].round(2).to_string())

# 3. Pusat = rata-rata anggota (pemeriksaan sifat Lloyd)
manual = np.array([Z[km.labels_ == k].mean(axis=0) for k in range(3)])
print("\nselisih maksimum pusat vs rata-rata anggota: %.1e" % np.abs(manual - km.cluster_centers_).max())

# 4. Sensitif terhadap inisialisasi: satu kali inisialisasi acak per seed
print("\nseed  inertia  ARI   (init acak, n_init=1)")
for seed in range(5):
    m = KMeans(n_clusters=3, init="random", n_init=1, random_state=seed).fit(Z)
    print(f"{seed:<5} {m.inertia_:<8.1f} {adjusted_rand_score(y, m.labels_):.3f}")

# 5. Keterbatasan: klaster non-bulat (dua bulan sabit, data SINTETIS)
Xm, ym = make_moons(n_samples=400, noise=0.08, random_state=SEED)
lm = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(Xm)
print("\nk-Means pada dua bulan sabit: ARI = %.3f" % adjusted_rand_score(ym, lm))

# 6. Gambar: Wine pada 2 komponen utama (PCA, hanya untuk visualisasi)
P = PCA(n_components=2, random_state=SEED).fit_transform(Z)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
for ax, labels, title in [(axes[0], km.labels_, "Klaster k-Means (k=3)"), (axes[1], y, "Kultivar sebenarnya")]:
    ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="viridis", s=18)
    ax.set(title=title, xlabel="komponen utama 1")
axes[0].set_ylabel("komponen utama 2")
fig.tight_layout()
fig.savefig(OUT / "kmeans-wine.png", dpi=130)
print("Gambar tersimpan:", (OUT / "kmeans-wine.png").as_posix())
