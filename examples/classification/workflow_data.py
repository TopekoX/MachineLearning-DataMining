from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# 1. Bentuk data dan arti label
print("Ukuran fitur:", X.shape)
print("Nama kelas  :", dict(enumerate(map(str, data.target_names))))
print(y.value_counts().rename(index=dict(enumerate(map(str, data.target_names)))))
print("Proporsi kelas:", y.value_counts(normalize=True).round(3).to_dict())

# 2. Contoh fitur (30 fitur = 10 pengukuran x {mean, error, worst})
print("\nContoh nama fitur:", list(X.columns[:3]), "...", list(X.columns[-3:]))
print("Ada data hilang?", bool(X.isna().any().any()), "| duplikat:", int(X.duplicated().sum()))

# 3. Rentang nilai sangat berbeda antar fitur (penting untuk scaling)
summary = X[["mean radius", "mean area", "mean smoothness"]].describe().loc[["min", "max"]].round(3)
print("\n", summary)

# 4. Perbedaan rata-rata fitur antar kelas
print(X.groupby(y.map(dict(enumerate(map(str, data.target_names)))))[["mean radius", "mean concavity"]].mean().round(3))
