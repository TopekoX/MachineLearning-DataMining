from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_wine, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("docs/assets/img/clustering")
OUT.mkdir(parents=True, exist_ok=True)

# Data SINTETIS: dua bulan sabit (non-bulat) dengan sedikit derau
X, y = make_moons(n_samples=400, noise=0.08, random_state=SEED)


def summarize(labels):
    n_clusters = len(set(labels) - {-1})
    return n_clusters, 100 * np.mean(labels == -1)


# 1. Pengaruh eps (min_samples=5)
print("eps   klaster  noise(%)  ARI")
for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    n, noise = summarize(labels)
    print(f"{eps:<5} {n:<8} {noise:<9.1f} {adjusted_rand_score(y, labels):.3f}")

# 2. Memilih eps dengan jarak ke tetangga ke-k (k = min_samples)
k = 5
dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
kdist = np.sort(dist[:, -1])
print("\njarak ke tetangga ke-5 (persentil 50/90/95/99): %s" % np.percentile(kdist, [50, 90, 95, 99]).round(3))

# 3. Perbandingan dengan k-Means
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(X)
print("\nARI pada bulan sabit: DBSCAN(eps=0.2) %.3f | k-Means %.3f" % (
    adjusted_rand_score(y, db), adjusted_rand_score(y, km)))
print("DBSCAN: klaster %d, titik noise %d" % (len(set(db) - {-1}), int((db == -1).sum())))

# 4. DBSCAN pada data berdimensi tinggi: Wine (13 fitur, terstandardisasi)
Xw, yw = load_wine(return_X_y=True)
Zw = StandardScaler().fit_transform(Xw)
print("\nWine terstandardisasi, min_samples=5")
print("eps   klaster  noise(%)  ARI")
for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(Zw)
    n, noise = summarize(labels)
    print(f"{eps:<5} {n:<8} {noise:<9.1f} {adjusted_rand_score(yw, labels):.3f}")

# 5. Gambar
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
for ax, labels, title in [(axes[0], y, "Kelompok sebenarnya"), (axes[1], km, "k-Means (k=2)"),
                          (axes[2], db, "DBSCAN (eps=0,2)")]:
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=14)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(OUT / "moons.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "moons.png").as_posix())
