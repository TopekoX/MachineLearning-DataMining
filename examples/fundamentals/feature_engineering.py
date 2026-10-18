import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

from customer_data import make_customer_data

df = make_customer_data().drop_duplicates().reset_index(drop=True)

# 1. Fitur turunan dari pengetahuan domain
df["spend_total"] = df["n_orders"] * df["avg_spend"]
df["orders_per_month"] = df["n_orders"] / df["tenure_months"]
df["is_new"] = (df["tenure_months"] <= 6).astype(int)

# 2. Transformasi log untuk data miring (income dapat bernilai hilang -> NaN tetap NaN)
df["log_income"] = np.log1p(df["income"])
print("skew income     : %.2f" % df["income"].skew())
print("skew log_income : %.2f" % df["log_income"].skew())

# 3. Binning usia menjadi kelompok
df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 60, 100],
                         labels=["18-30", "31-45", "46-60", "61+"])
print(df.groupby("age_group", observed=True)["churn"].agg(["size", "mean"]).round(3))

# 4. Fitur polinomial dan interaksi
X = df[["tenure_months", "n_orders"]]
poly = PolynomialFeatures(degree=2, include_bias=False)
print("fitur asli     :", list(X.columns))
print("fitur polinomial:", list(poly.fit(X).get_feature_names_out()))

# 5. Apakah fitur baru membantu? Bandingkan dengan validasi silang (5 lipatan)
y = df["churn"]
def cv_auc(columns):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_score(model, df[columns], y, cv=5, scoring="roc_auc").mean()

base = ["tenure_months", "n_orders", "age", "avg_spend"]
print("ROC-AUC fitur asli        : %.3f" % cv_auc(base))
print("ROC-AUC + fitur turunan   : %.3f" % cv_auc(base + ["orders_per_month", "is_new"]))
