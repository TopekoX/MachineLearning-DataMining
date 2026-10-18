from customer_data import make_customer_data

df = make_customer_data()

# 1. Ukuran, tipe, dan contoh data
print("Ukuran:", df.shape)
print(df.dtypes.to_string())

# 2. Data hilang (jumlah dan persentase)
missing = df.isna().sum()
summary = missing[missing > 0].to_frame("n_missing")
summary["pct"] = (summary["n_missing"] / len(df) * 100).round(1)
print(summary)

# 3. Duplikat
print("Baris duplikat:", df.duplicated().sum())

# 4. Ringkasan statistik numerik
numeric = ["age", "tenure_months", "income", "n_orders", "avg_spend"]
print(df[numeric].describe().T[["count", "mean", "50%", "max"]].round(0).to_string())

# 5. Distribusi target: seimbang atau tidak?
print(df["churn"].value_counts(normalize=True).round(3))

# 6. Hubungan fitur dengan target
print(df.groupby("plan")["churn"].mean().round(3))
print(df.groupby(df["tenure_months"] // 12 * 12)["churn"].mean().round(3).rename("churn per 12 bulan"))
print(df[["tenure_months", "n_orders", "age", "avg_spend", "churn"]].corr()["churn"].round(2))
