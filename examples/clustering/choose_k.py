from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("docs/assets/img/clustering")
OUT.mkdir(parents=True, exist_ok=True)

X, y = load_wine(return_X_y=True)
Z = StandardScaler().fit_transform(X)

# 1. Metrik internal untuk k = 2..8 (tidak memerlukan label)
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(Z, km.labels_),
                 "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                 "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
                 "ARI_vs_kultivar": adjusted_rand_score(y, km.labels_)})
table = pd.DataFrame(rows).set_index("k")
print(table.round(3).to_string())
print("\nk terbaik menurut silhouette (maks.): %d | Davies-Bouldin (min.): %d | Calinski-Harabasz (maks.): %d" % (
    table["silhouette"].idxmax(), table["davies_bouldin"].idxmin(), table["calinski_harabasz"].idxmax()))

# 2. Penurunan inertia (siku): selisih relatif antar k
drop = (-table["inertia"].diff() / table["inertia"].shift(1) * 100).dropna().round(1)
print("penurunan inertia (%) dari k-1 ke k:", drop.to_dict())

# 3. Kestabilan: klaster pada sampel bootstrap vs klaster data penuh (ARI pada titik yang terambil)
rng = np.random.default_rng(SEED)
print("\nk   rata-rata ARI kestabilan (30 bootstrap)")
for k in range(2, 7):
    full = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
    aris = []
    for _ in range(30):
        idx = np.unique(rng.integers(0, len(Z), len(Z)))
        boot = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1_000_000))).fit_predict(Z[idx])
        aris.append(adjusted_rand_score(full[idx], boot))
    print(f"{k:<3} {np.mean(aris):.3f}")

# 4. Gambar: elbow dan silhouette
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(table.index, table["inertia"], marker="o")
axes[0].set(title="Elbow (inertia)", xlabel="k", ylabel="inertia")
axes[1].plot(table.index, table["silhouette"], marker="o", color="C1")
axes[1].set(title="Silhouette rata-rata", xlabel="k")
fig.tight_layout()
fig.savefig(OUT / "choose-k.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "choose-k.png").as_posix())
