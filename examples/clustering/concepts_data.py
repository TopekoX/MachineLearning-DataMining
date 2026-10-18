import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from segment_data import make_customer_segments

SEED = 42
df = make_customer_segments()
features = ["annual_spend", "visits_per_month", "avg_basket", "recency_days", "age"]
X = df[features]

# 1. EDA: ukuran, data hilang, rentang nilai (skala sangat berbeda!)
print("Ukuran:", X.shape, "| hilang:", int(X.isna().sum().sum()), "| duplikat:", int(X.duplicated().sum()))
print(X.describe().T[["mean", "std", "min", "max"]].round(1).to_string())
print("skewness annual_spend: %.2f" % X["annual_spend"].skew())

# 2. Kelompok tersembunyi hanya untuk validasi eksternal pada materi ini
print("\nUkuran kelompok sebenarnya:", df["segment_true"].value_counts().sort_index().to_dict())

# 3. Mengapa tidak ada "data uji berlabel": kita tidak punya jawaban benar.
#    Yang dapat diuji adalah KESTABILAN: apakah struktur yang sama ditemukan pada data lain?
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED)
scaler = StandardScaler().fit(X.iloc[train_idx])            # praproses dipelajari dari data latih
km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(scaler.transform(X.iloc[train_idx]))

assigned = km.predict(scaler.transform(X.iloc[test_idx]))   # data uji ditempatkan ke klaster latih
own = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit_predict(
    StandardScaler().fit_transform(X.iloc[test_idx]))       # data uji diklaster sendiri
print("\nKestabilan (ARI antara dua cara mengelompokkan data uji): %.3f" % adjusted_rand_score(assigned, own))

# 4. Validasi eksternal (hanya mungkin karena data sintetis ini punya kelompok sebenarnya)
print("ARI terhadap kelompok sebenarnya (data uji, penempatan dari model latih): %.3f"
      % adjusted_rand_score(df["segment_true"].iloc[test_idx], assigned))
