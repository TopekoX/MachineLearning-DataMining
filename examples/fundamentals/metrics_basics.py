import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)

# ---------- Klasifikasi: contoh kecil yang bisa dihitung tangan ----------
y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print("accuracy : %.3f" % accuracy_score(y_true, y_pred))
print("precision: %.3f" % precision_score(y_true, y_pred))
print("recall   : %.3f" % recall_score(y_true, y_pred))
print("F1       : %.3f" % f1_score(y_true, y_pred))

# Jebakan accuracy pada data tidak seimbang: 95 negatif, 5 positif
y_imb = np.array([0] * 95 + [1] * 5)
dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((100, 1)), y_imb)
pred = dummy.predict(np.zeros((100, 1)))
print("\nModel yang selalu menebak 0 -> accuracy: %.2f | recall: %.2f"
      % (accuracy_score(y_imb, pred), recall_score(y_imb, pred)))

# ---------- Regresi ----------
y_real = np.array([100, 150, 200, 250, 300])
y_hat = np.array([110, 140, 210, 230, 330])
mse = mean_squared_error(y_real, y_hat)
print("\nMAE : %.1f" % mean_absolute_error(y_real, y_hat))
print("MSE : %.1f" % mse)
print("RMSE: %.2f" % np.sqrt(mse))
print("R2  : %.3f" % r2_score(y_real, y_hat))
