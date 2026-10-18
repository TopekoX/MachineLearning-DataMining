import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
X, y = load_wine(return_X_y=True)
Z = StandardScaler().fit_transform(X)


def internal(labels):
    mask = labels != -1  # titik noise DBSCAN dikeluarkan dari metrik internal
    if len(set(labels[mask])) < 2:
        return np.nan, np.nan
    return silhouette_score(Z[mask], labels[mask]), davies_bouldin_score(Z[mask], labels[mask])


# 1. Optimasi parameter TANPA label: DBSCAN dengan silhouette (hanya titik non-noise) pada grid kecil
best = None
for eps in [2.0, 2.5, 3.0, 3.5, 4.0]:
    for ms in [3, 5, 10]:
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
        sil, _ = internal(lab)
        if not np.isnan(sil) and (best is None or sil > best[0]):
            best = (sil, eps, ms, lab)
print("DBSCAN terbaik menurut silhouette: eps=%.1f, min_samples=%d (silhouette %.3f, klaster %d, noise %.0f%%)" % (
    best[1], best[2], best[0], len(set(best[3]) - {-1}), 100 * np.mean(best[3] == -1)))

# 2. Perbandingan algoritma pada data yang sama
models = {
    "k-Means (k=3)": KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(Z),
    "Agglomerative Ward (k=3)": AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Z),
    "Gaussian Mixture (k=3)": GaussianMixture(n_components=3, random_state=SEED).fit_predict(Z),
    "DBSCAN (parameter di atas)": best[3],
}
rows = []
for name, lab in models.items():
    sil, db = internal(lab)
    rows.append({"Metode": name, "klaster": len(set(lab) - {-1}), "noise%": 100 * np.mean(lab == -1),
                 "silhouette": sil, "DB": db, "ARI": adjusted_rand_score(y, lab),
                 "NMI": normalized_mutual_info_score(y, lab)})
print("\n" + pd.DataFrame(rows).set_index("Metode").round(3).to_string())

# 3. Kesepakatan antar-algoritma
a, b = models["k-Means (k=3)"], models["Agglomerative Ward (k=3)"]
print("\nARI antara k-Means dan Ward: %.3f" % adjusted_rand_score(a, b))
print("ARI antara k-Means dan Gaussian Mixture: %.3f" % adjusted_rand_score(a, models["Gaussian Mixture (k=3)"]))
