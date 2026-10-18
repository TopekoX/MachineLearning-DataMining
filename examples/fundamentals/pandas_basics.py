from customer_data import make_customer_data

# Pada proyek nyata data dibaca dari berkas, misalnya: df = pd.read_csv("data.csv")
df = make_customer_data()

# 1. Mengenal data
print(df.shape)
print(df.head(3))
df.info()

# 2. Series vs DataFrame
age = df["age"]            # satu kurung siku -> Series (satu kolom)
age_frame = df[["age"]]    # dua kurung siku  -> DataFrame (tabel berisi satu kolom)
print(type(age).__name__, age.shape, "|", type(age_frame).__name__, age_frame.shape)

# 3. Memilih baris: syarat boolean, loc (label), iloc (posisi)
print(df["plan"].value_counts())
premium_senior = df[(df["plan"] == "premium") & (df["age"] >= 50)]
print("premium & usia>=50:", len(premium_senior))
print(df.loc[0:2, ["age", "city"]])   # loc: 0:2 TERMASUK baris berlabel 2
print(df.iloc[0:2, [0, 1]])           # iloc: 0:2 TIDAK termasuk posisi 2

# 4. Agregasi per kelompok
print(df.groupby("plan")["churn"].mean().round(3))
print(df.groupby("city").agg(n=("churn", "size"), churn_rate=("churn", "mean")).round(3))

# 5. Data hilang dan duplikat
print(df.isna().sum())
print("duplikat:", df.duplicated().sum())

# 6. Kolom baru dan pengurutan
df["spend_total"] = df["n_orders"] * df["avg_spend"]
print(df.sort_values("spend_total", ascending=False)[["n_orders", "avg_spend", "spend_total"]].head(3))
