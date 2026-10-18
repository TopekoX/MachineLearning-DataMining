# Dasar-dasar

Topik fondasi yang dipakai di semua bab. Setiap topik diajarkan sekali di sini, lalu bab-bab merujuknya.

## Prasyarat

Pemahaman dasar pemrograman dan matematika tingkat SMA. Python dijelaskan dari awal.

## Daftar pelajaran

| # | Pelajaran | Isi utama | Dipakai oleh |
|---|---|---|---|
| 1 | [Lingkungan Python](python-for-data-mining.md) | venv, instalasi, skrip vs notebook, reproduksibilitas | Semua bab |
| 2 | [NumPy](numpy.md) | array, `axis`, broadcasting, jarak Euclidean | Semua bab |
| 3 | [pandas](pandas.md) | `DataFrame`, seleksi, `groupby`, data hilang | Semua bab |
| 4 | [Visualisasi Data](data-visualization.md) | histogram, boxplot, scatter, heatmap | EDA, semua bab |
| 5 | [Statistik](statistics.md) | median, IQR, korelasi, probabilitas bersyarat | EDA, Estimasi, Forecasting |
| 6 | [EDA](eda.md) | daftar periksa EDA, keputusan praproses | Semua bab |
| 7 | [Pembersihan dan Praproses](data-preprocessing.md) | imputasi, *outlier*, *encoding*, *scaling* | Semua bab |
| 8 | [Feature Engineering](feature-engineering.md) | fitur turunan, log, *binning*, polinomial | Klasifikasi, Estimasi, Forecasting |
| 9 | [Data Latih, Validasi, Uji](data-splitting.md) | split, *stratified*, validasi silang | Semua bab |
| 10 | [Metrik Evaluasi](evaluation-metrics.md) | precision, recall, F1, MAE, RMSE, R² | Klasifikasi, Estimasi |
| 11 | [Workflow scikit-learn dan Pipeline](sklearn-pipeline.md) | `Pipeline`, `ColumnTransformer` | Semua bab |
| 12 | [Overfitting, Underfitting, Kebocoran Data](overfitting-leakage.md) | diagnosis dan pencegahan | Semua bab |

## Data dan reproduksibilitas

Contoh kode ada di folder `examples/fundamentals/`. Banyak lesson memakai **data pelanggan sintetis** (`customer_data.py`) dan beberapa dataset bawaan scikit-learn. Setiap contoh memakai seed tetap; keluaran yang ditampilkan berasal dari program yang benar-benar dijalankan (Python 3.14, scikit-learn 1.9).
