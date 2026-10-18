import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from customer_data import make_customer_data

df = make_customer_data()
income = df["income"].dropna()

# 1. Pemusatan dan sebaran
print("mean   : %.0f" % income.mean())
print("median : %.0f" % income.median())
print("std    : %.0f" % income.std())
print("skew   : %.2f" % income.skew())

# 2. Kuartil dan aturan IQR untuk outlier
q1, q3 = income.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = income[(income < lower) | (income > upper)]
print("Q1=%.0f  Q3=%.0f  IQR=%.0f" % (q1, q3, iqr))
print("batas: [%.0f, %.0f] -> %d outlier" % (lower, upper, len(outliers)))

# 3. Korelasi dan batasannya
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 200)
y = x ** 2 + rng.normal(0, 0.3, 200)   # hubungan jelas, tetapi tidak linear
print("korelasi Pearson x vs x^2: %.2f" % np.corrcoef(x, y)[0, 1])

diabetes = load_diabetes(as_frame=True).frame
print(diabetes.corr()["target"].drop("target").sort_values(ascending=False).round(2).head(3))

# 4. Probabilitas bersyarat dari data: P(churn | plan)
print(pd.crosstab(df["plan"], df["churn"], normalize="index").round(3))
print("P(churn) keseluruhan: %.3f" % df["churn"].mean())
