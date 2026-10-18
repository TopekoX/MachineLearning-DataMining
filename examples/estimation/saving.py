import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X_train, y_train)

# 1. Simpan pipeline + metadata, termasuk rentang fitur data latih untuk pemeriksaan masukan
out_dir = Path(tempfile.mkdtemp())
meta = {
    "sklearn_version": sklearn.__version__,
    "features": list(X.columns),
    "train_min": X_train.min().to_dict(),
    "train_max": X_train.max().to_dict(),
    "train_mae_reference": float(mean_absolute_error(y_test, model.predict(X_test))),
}
joblib.dump(model, out_dir / "diabetes_model.joblib")
(out_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

# 2. Muat ulang dan pastikan prediksi identik
loaded = joblib.load(out_dir / "diabetes_model.joblib")
print("prediksi identik setelah dimuat:", np.allclose(model.predict(X_test), loaded.predict(X_test)))


# 3. Fungsi prediksi dengan pemeriksaan masukan di luar rentang latih (risiko ekstrapolasi)
def predict_checked(df):
    lo, hi = pd.Series(meta["train_min"]), pd.Series(meta["train_max"])
    out_of_range = ((df < lo) | (df > hi)).any(axis=1)
    return loaded.predict(df), out_of_range.to_numpy()


sample = X_test.iloc[[0, 1]].copy()
sample.loc[sample.index[1], "bmi"] = X_train["bmi"].max() * 2  # nilai ekstrem buatan
pred, flag = predict_checked(sample)
print("prediksi:", pred.round(1), "| di luar rentang latih:", flag)

# 4. Pemantauan galat saat label sebenarnya tersedia (batch SINTETIS: target digeser +30)
print("\nbatch               MAE")
print(f"data uji            {mean_absolute_error(y_test, loaded.predict(X_test)):.2f}")
shifted = y_test + 30
print(f"target bergeser +30 {mean_absolute_error(shifted, loaded.predict(X_test)):.2f}")
print("bias (rata-rata galat) batch bergeser: %.1f" % (shifted - loaded.predict(X_test)).mean())
