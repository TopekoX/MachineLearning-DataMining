import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)

# 1. Simpan pipeline utuh (praproses + model) dan metadata
out_dir = Path(tempfile.mkdtemp())
joblib.dump(model, out_dir / "breast_cancer_model.joblib")
meta = {
    "sklearn_version": sklearn.__version__,
    "features": list(X.columns),
    "classes": {"0": "ganas", "1": "jinak"},
    "metric_note": "dilatih pada data latih (stratified split, seed 42)",
}
(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("berkas tersimpan:", sorted(p.name for p in out_dir.iterdir()))

# 2. Muat kembali dan pastikan prediksinya identik
loaded = joblib.load(out_dir / "breast_cancer_model.joblib")
same = np.array_equal(model.predict(X_test), loaded.predict(X_test))
print("prediksi model asli == model termuat:", same)

# 3. Prediksi untuk satu sampel baru (harus berupa DataFrame dengan nama kolom yang sama)
new_sample = X_test.iloc[[0]]
print("P(ganas, jinak):", loaded.predict_proba(new_sample)[0].round(3), "| label asli:", int(y_test.iloc[0]))

# 4. Pemantauan sederhana: apakah distribusi data baru bergeser dari data latih?
#    Batch "baru" di bawah ini SINTETIS: salinan data uji dengan 'mean radius' digeser 30%.
new_batch = X_test.copy()
new_batch["mean radius"] = new_batch["mean radius"] * 1.3
print("\nfitur              p-value KS (data latih vs batch baru)")
for col in ["mean radius", "mean texture", "mean smoothness"]:
    p = ks_2samp(X_train[col], new_batch[col]).pvalue
    print(f"{col:<18} {p:.4f}", "<- bergeser" if p < 0.01 else "")

# 5. Pemantauan keluaran: porsi prediksi 'ganas' sebelum dan sesudah pergeseran
share_before = (loaded.predict(X_test) == 0).mean()
share_after = (loaded.predict(new_batch) == 0).mean()
print("\nporsi diprediksi ganas -> data uji: %.3f | batch bergeser: %.3f" % (share_before, share_after))
