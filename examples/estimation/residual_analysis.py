from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path("docs/assets/img/estimation")
OUT.mkdir(parents=True, exist_ok=True)

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
resid = y_test - pred  # residual = nilai sebenarnya - prediksi

# 1. Metrik ringkas
mse = mean_squared_error(y_test, pred)
print("MAE %.2f | MSE %.1f | RMSE %.2f | R2 %.3f" % (mean_absolute_error(y_test, pred), mse, np.sqrt(mse), r2_score(y_test, pred)))

# 2. Ringkasan residual
print("\nResidual -> rata-rata %.2f | std %.2f | skew %.2f | min %.1f | maks %.1f"
      % (resid.mean(), resid.std(), resid.skew(), resid.min(), resid.max()))
print("Korelasi |residual| dengan nilai prediksi: %.2f" % np.corrcoef(np.abs(resid), pred)[0, 1])

# 3. Galat per rentang prediksi (apakah galat sama besar di semua rentang?)
bins = pd.qcut(pd.Series(pred, index=resid.index), 3, labels=["rendah", "sedang", "tinggi"])
print("\nMAE per tertil prediksi:\n", resid.abs().groupby(bins, observed=True).mean().round(1).to_string())

# 4. Tiga galat terbesar
worst = resid.abs().sort_values(ascending=False).head(3).index
print("\nTiga galat terbesar (aktual, prediksi, residual):")
for i in worst:
    print(f"  idx {i}: {y_test[i]:.0f}, {pred[list(y_test.index).index(i)]:.0f}, {resid[i]:+.0f}")

# 5. Gambar diagnostik
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].scatter(pred, y_test, s=14, alpha=0.7)
lims = [min(pred.min(), y_test.min()), max(pred.max(), y_test.max())]
axes[0].plot(lims, lims, color="C3")
axes[0].set(title="Aktual vs prediksi", xlabel="prediksi", ylabel="aktual")
axes[1].scatter(pred, resid, s=14, alpha=0.7)
axes[1].axhline(0, color="C3")
axes[1].set(title="Residual vs prediksi", xlabel="prediksi", ylabel="residual")
sns.histplot(resid, bins=20, ax=axes[2])
axes[2].set(title="Distribusi residual", xlabel="residual")
fig.tight_layout()
fig.savefig(OUT / "residuals.png", dpi=130)
print("\nGambar tersimpan:", (OUT / "residuals.png").as_posix())
