import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from segment_data import make_customer_segments

SEED = 42
features = ["annual_spend", "visits_per_month", "avg_basket", "recency_days", "age"]
skewed = ["annual_spend", "recency_days"]

train = make_customer_segments(n=600, seed=42)
pipe = Pipeline([
    ("prep", ColumnTransformer([("log", FunctionTransformer(np.log1p), skewed)], remainder="passthrough")),
    ("scale", StandardScaler()),
    ("km", KMeans(n_clusters=4, n_init=10, random_state=SEED)),
]).fit(train[features])

# 1. Nama segmen berdasarkan pusat klaster (disimpan bersama pipeline agar tidak berubah)
centers = pd.DataFrame(pipe[:-1].named_steps["scale"].inverse_transform(pipe["km"].cluster_centers_),
                       columns=skewed + [c for c in features if c not in skewed])
names = {}
left = set(range(4))
for rule, label in [(lambda i: centers.loc[i, "recency_days"], "tidak aktif"),
                    (lambda i: centers.loc[i, "annual_spend"], "loyal"),
                    (lambda i: centers.loc[i, "visits_per_month"], "pemburu diskon")]:
    pick = max(left, key=rule)
    names[pick] = label
    left.discard(pick)
names[left.pop()] = "musiman"

path = Path(tempfile.mkdtemp()) / "segment_model.joblib"
joblib.dump({"pipeline": pipe, "names": names, "features": features}, path)
bundle = joblib.load(path)
print("nama segmen tersimpan:", {int(k): v for k, v in sorted(bundle["names"].items())})

# 2. Menempatkan pelanggan BARU: tidak ada fit ulang; praproses memakai parameter data latih
new = pd.DataFrame([[35.0, 8.0, 250.0, 7.0, 36], [3.0, 0.5, 100.0, 300.0, 50], [14.0, 1.0, 450.0, 60.0, 47]],
                   columns=features)
assigned = bundle["pipeline"].predict(new[features])
print("pelanggan baru ->", [bundle["names"][int(c)] for c in assigned])

# 3. Deteksi pelanggan yang tidak cocok dengan segmen mana pun: jarak ke pusat terdekat
train_dist = bundle["pipeline"].transform(train[features]).min(axis=1)
limit = np.percentile(train_dist, 99)
odd = pd.DataFrame([[400.0, 20.0, 5000.0, 1.0, 30]], columns=features)  # nilai ekstrem buatan
d_new = bundle["pipeline"].transform(pd.concat([new, odd]).reset_index(drop=True)[features]).min(axis=1)
print("\nbatas jarak (persentil 99 data latih): %.2f" % limit)
print("jarak pelanggan baru ke pusat terdekat:", d_new.round(2).tolist(), "| melebihi batas:", (d_new > limit).tolist())

# 4. Pemantauan: porsi segmen pada batch baru vs data latih
print("\nporsi segmen (%)  latih | batch sama distribusi | batch SINTETIS bergeser")
same = make_customer_segments(n=600, seed=7)
shifted = make_customer_segments(n=600, seed=8)
shifted["recency_days"] = shifted["recency_days"] * 3   # simulasi: pelanggan makin jarang kembali
shifted["visits_per_month"] = shifted["visits_per_month"] * 0.5


def share(df):
    return pd.Series(bundle["pipeline"].predict(df[features])).map(bundle["names"]).value_counts(normalize=True) * 100


table = pd.DataFrame({"latih": share(train), "batch_sama": share(same), "batch_bergeser": share(shifted)}).fillna(0).round(0)
print(table.to_string())
for col in ["batch_sama", "batch_bergeser"]:
    print(f"selisih porsi terbesar {col:<15}: {(table[col] - table['latih']).abs().max():.0f} poin persen")

# 5. Sinyal lain: rata-rata jarak ke pusat terdekat (makin besar = data makin "jauh" dari segmen yang dikenal)
for name, batch in [("latih", train), ("batch_sama", same), ("batch_bergeser", shifted)]:
    print(f"rata-rata jarak ke pusat terdekat {name:<15}: {bundle['pipeline'].transform(batch[features]).min(axis=1).mean():.2f}")
