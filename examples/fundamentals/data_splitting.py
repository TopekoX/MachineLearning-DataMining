import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42

# Data SINTETIS tidak seimbang: sekitar 10% kelas positif
X, y = make_classification(n_samples=300, n_features=5, weights=[0.9, 0.1],
                           random_state=SEED)
print("Proporsi positif seluruh data: %.3f" % y.mean())

# 1. Split biasa vs stratified: proporsi kelas pada data uji
for strat in (None, y):
    _, _, _, y_te = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=SEED)
    label = "stratified" if strat is not None else "acak biasa"
    print(f"{label:<11}: positif di data uji = {int(y_te.sum())} dari {len(y_te)} ({y_te.mean():.3f})")

# 2. Tiga bagian: latih 60% / validasi 20% / uji 20%
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25,
                                                  stratify=y_tmp, random_state=SEED)
print("ukuran latih/validasi/uji:", len(X_train), len(X_val), len(X_test))

# 3. Stratified K-Fold: setiap lipatan menjaga proporsi kelas
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for i, (tr, te) in enumerate(skf.split(X_tmp, y_tmp), start=1):
    print(f"lipatan {i}: latih={len(tr)}, validasi={len(te)}, positif validasi={int(y_tmp[te].sum())}")
