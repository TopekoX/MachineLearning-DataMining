from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

OUT = Path("docs/assets/img/clustering")
OUT.mkdir(parents=True, exist_ok=True)

X, y = load_wine(return_X_y=True)
Z = StandardScaler().fit_transform(X)

# 1. Pengaruh jenis linkage (k=3)
print("linkage   ARI vs kultivar  silhouette  ukuran klaster")
for method in ["ward", "complete", "average", "single"]:
    labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(Z)
    print(f"{method:<9} {adjusted_rand_score(y, labels):<16.3f} {silhouette_score(Z, labels):<11.3f} {np.bincount(labels).tolist()}")

# 2. Matriks linkage (scipy) dan lonjakan jarak penggabungan pada Ward
L = linkage(Z, method="ward")
print("\n8 jarak penggabungan terakhir (Ward):", L[-8:, 2].round(1).tolist())
gaps = np.diff(L[-8:, 2])
print("selisih antar penggabungan terakhir:", gaps.round(1).tolist())

# 3. Memotong dendrogram dengan ambang jarak, bukan dengan k
labels_thr = AgglomerativeClustering(n_clusters=None, distance_threshold=20, linkage="ward").fit_predict(Z)
print("ambang jarak 20 -> %d klaster" % (labels_thr.max() + 1))

# 4. Gambar dendrogram (dipangkas: 20 penggabungan terakhir)
fig, ax = plt.subplots(figsize=(9, 3.8))
dendrogram(L, truncate_mode="lastp", p=20, ax=ax, color_threshold=20, no_labels=True)
ax.axhline(20, color="gray", linestyle="--")
ax.set(title="Dendrogram Ward (dipangkas)", ylabel="jarak penggabungan")
fig.tight_layout()
fig.savefig(OUT / "dendrogram.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "dendrogram.png").as_posix())
