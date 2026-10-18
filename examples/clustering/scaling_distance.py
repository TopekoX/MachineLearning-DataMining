import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SEED = 42
X, y = load_wine(return_X_y=True, as_frame=True)  # y = kultivar (hanya untuk validasi eksternal)

# 1. Skala fitur sangat berbeda
rng = X.agg(["min", "max"]).T
print("rentang proline: %.0f - %.0f | rentang hue: %.2f - %.2f" % (
    rng.loc["proline", "min"], rng.loc["proline", "max"], rng.loc["hue", "min"], rng.loc["hue", "max"]))
print("std per fitur (terbesar/terkecil):", X.std().idxmax(), "=%.1f" % X.std().max(), "|", X.std().idxmin(), "=%.2f" % X.std().min())

# 2. Jarak Euclidean antara dua sampel, sebelum dan sesudah scaling
a, b = X.iloc[0].to_numpy(), X.iloc[100].to_numpy()
raw = np.sqrt(((a - b) ** 2).sum())
contrib = ((a - b) ** 2) / ((a - b) ** 2).sum()
print("\njarak mentah sampel 0 dan 100: %.1f | kontribusi proline terhadap jarak^2: %.1f%%" % (
    raw, 100 * contrib[X.columns.get_loc("proline")]))
Z = StandardScaler().fit_transform(X)
za, zb = Z[0], Z[100]
print("jarak setelah standardisasi: %.2f | kontribusi proline: %.1f%%" % (
    np.sqrt(((za - zb) ** 2).sum()), 100 * ((za - zb) ** 2 / ((za - zb) ** 2).sum())[X.columns.get_loc("proline")]))

# 3. Ukuran jarak lain pada data terstandardisasi
print("\njarak sampel 0-100 (standar): Euclidean %.2f | Manhattan %.2f | cosine %.3f" % (
    cdist([za], [zb], "euclidean")[0, 0], cdist([za], [zb], "cityblock")[0, 0], cdist([za], [zb], "cosine")[0, 0]))

# 4. Dampak scaling pada k-Means (k=3): ARI terhadap kultivar dan silhouette
print("\nperlakuan         ARI vs kultivar   silhouette")
for name, data in [("tanpa scaling", X.to_numpy()),
                   ("StandardScaler", StandardScaler().fit_transform(X)),
                   ("MinMaxScaler", MinMaxScaler().fit_transform(X)),
                   ("RobustScaler", RobustScaler().fit_transform(X))]:
    labels = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(data)
    print(f"{name:<17} {adjusted_rand_score(y, labels):.3f}             {silhouette_score(data, labels):.3f}")
