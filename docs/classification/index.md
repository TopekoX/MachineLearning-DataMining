# Klasifikasi (Classification)

Klasifikasi memprediksi **kelas (label kategorik)** dari sebuah data, misalnya email *spam* atau bukan, atau tumor ganas atau jinak. Ini adalah *supervised learning*: model belajar dari data yang labelnya sudah diketahui.

## Tujuan pembelajaran

- Merumuskan masalah klasifikasi dan memilih metrik yang sesuai.
- Membangun *pipeline* klasifikasi tanpa kebocoran data.
- Membandingkan beberapa algoritma secara adil dan menafsirkan hasilnya.
- Menangani kelas tidak seimbang, menyetel hyperparameter, dan menyimpan model.

## Prasyarat

Bab [Dasar-dasar](../fundamentals/index.md), terutama [Praproses Data](../fundamentals/data-preprocessing.md), [Data Latih, Validasi, Uji](../fundamentals/data-splitting.md), [Metrik Evaluasi](../fundamentals/evaluation-metrics.md), dan [Pipeline](../fundamentals/sklearn-pipeline.md).

## Tahap alur kerja yang relevan

| Tahap | Status | Catatan |
|---|---|---|
| Definisi masalah, pengumpulan data, EDA, praproses | Wajib | Periksa distribusi kelas |
| Rekayasa fitur | Opsional | |
| Pembagian data | Wajib | *Stratified*, validasi silang |
| Pemilihan, pelatihan model | Wajib | Mulai dari *baseline* |
| Evaluasi | Wajib | *Confusion matrix*, precision/recall/F1, ROC-AUC, PR-AUC |
| Optimasi | Wajib | Grid/Random Search dengan validasi silang |
| Penyimpanan model | Wajib | `joblib` |
| Deployment dan monitoring | Opsional | Konsep, contoh sederhana, pemantauan pergeseran |

## Pelajaran

| # | Pelajaran | Isi utama |
|---|---|---|
| 1 | [Alur Kerja dan Data](workflow-and-data.md) | Rumusan masalah, dataset, label, pemeriksaan awal |
| 2 | [Praproses, Pembagian Data, Baseline](preprocessing-baseline.md) | Keputusan praproses, *stratified split*, *baseline* |
| 3 | [Logistic Regression](logistic-regression.md) | Sigmoid, regularisasi `C`, koefisien |
| 4 | [KNN](knn.md) | Jarak, `k`, *scaling* |
| 5 | [Decision Tree](decision-tree.md) | Gini, kedalaman, aturan terbaca |
| 6 | [Random Forest](random-forest.md) | *Bagging*, OOB, *feature importance* |
| 7 | [SVM](svm.md) | Margin, `C`, kernel RBF, `gamma` |
| 8 | [Naive Bayes](naive-bayes.md) | Teorema Bayes, asumsi independensi, teks |
| 9 | [Perbandingan dan Pemilihan Model](model-selection.md) | Validasi silang berulang, interpretasi |
| 10 | [Evaluasi Model](evaluation.md) | *Confusion matrix*, ROC, PR, ambang |
| 11 | [Kelas Tidak Seimbang dan Validasi Silang](imbalance-cv.md) | Metrik, `class_weight`, stratifikasi |
| 12 | [Optimasi Hyperparameter](hyperparameter-tuning.md) | Grid/Random Search, *nested CV* |
| 13 | [Menyimpan Model, Deployment, Monitoring](saving-deployment.md) | `joblib`, keamanan, pergeseran data |
| 14 | [Proyek End-to-End](project.md) | Prediksi churn dari awal sampai model tersimpan |

## Data yang dipakai

- **Breast Cancer Wisconsin (Diagnostic)**, data publik bawaan scikit-learn, untuk pelajaran 1–13.
- **Data sintetis** (pelanggan churn dan data tidak seimbang) untuk bagian yang membutuhkan data campuran atau timpang; selalu ditandai sebagai sintetis.

Contoh kode ada di `examples/classification/`; keluaran yang ditampilkan berasal dari program yang dijalankan.
