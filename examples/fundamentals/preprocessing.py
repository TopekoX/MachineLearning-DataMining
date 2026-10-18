import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from customer_data import make_customer_data

df = make_customer_data()

# 1. Duplikat: hapus baris yang persis sama
print("sebelum:", len(df), "| duplikat:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("sesudah:", len(df))

# 2. Imputasi data hilang. (Di sini dilakukan pada seluruh data hanya untuk
#    demonstrasi; pada proyek nyata lakukan di dalam Pipeline setelah split.)
#    Median tahan terhadap outlier; modus untuk kategori.
income_imputer = SimpleImputer(strategy="median")
city_imputer = SimpleImputer(strategy="most_frequent")
df["income"] = income_imputer.fit_transform(df[["income"]]).ravel()
df["city"] = city_imputer.fit_transform(df[["city"]]).ravel()
print("sisa NaN:", int(df.isna().sum().sum()), "| median income:", income_imputer.statistics_[0])

# 3. Outlier: batasi (winsorize) pada batas IQR atas, bukan menghapus
q1, q3 = df["income"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print("income > batas atas:", int((df["income"] > upper).sum()), "baris")
df["income_capped"] = df["income"].clip(upper=upper)

# 4. Encoding kategorik
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
city_encoded = encoder.fit_transform(df[["city"]])
print("kategori kota:", list(encoder.categories_[0]))
print("hasil one-hot (3 baris pertama):\n", city_encoded[:3])

# 5. Scaling: bandingkan StandardScaler dan MinMaxScaler
cols = ["age", "income_capped"]
z = StandardScaler().fit_transform(df[cols])
m = MinMaxScaler().fit_transform(df[cols])
print("StandardScaler -> mean:", z.mean(axis=0).round(2), "std:", z.std(axis=0).round(2))
print("MinMaxScaler   -> min :", m.min(axis=0), "max:", m.max(axis=0))
