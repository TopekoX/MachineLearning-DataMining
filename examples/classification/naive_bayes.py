import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = 42

# ---------- Bagian 1: GaussianNB pada fitur numerik (Breast Cancer) ----------
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
gnb = GaussianNB()
print("GaussianNB akurasi CV: %.3f" % cross_val_score(gnb, X_train, y_train, cv=5).mean())
gnb.fit(X_train, y_train)
print("prior kelas [ganas, jinak]:", gnb.class_prior_.round(3))
print(classification_report(y_test, gnb.predict(X_test), digits=3))

# Asumsi independensi: banyak fitur sangat berkorelasi
corr = X_train[["mean radius", "mean perimeter", "mean area"]].corr().round(3)
print("Korelasi antar fitur ukuran tumor:\n", corr)

# ---------- Bagian 2: MultinomialNB pada teks (contoh SINTETIS kecil) ----------
docs = [
    "diskon besar gratis hadiah klik sekarang", "menang hadiah gratis hubungi segera",
    "promo gratis ongkir diskon besar", "klik tautan hadiah menang uang",
    "rapat proyek dijadwalkan besok pagi", "mohon kirim laporan proyek hari ini",
    "jadwal kuliah berubah mohon cek", "laporan keuangan rapat besok",
]
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1 = spam, 0 = bukan spam

text_model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
text_model.fit(docs, labels)
tests = ["hadiah gratis klik", "laporan rapat proyek", "diskon laporan"]
for t, p in zip(tests, text_model.predict_proba(tests)[:, 1]):
    print(f"P(spam | '{t}') = {p:.3f}")
