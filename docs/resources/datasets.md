# Dataset dan Sumber Belajar

## Prinsip pemilihan dataset

- Utamakan dataset bawaan pustaka atau data sintetis dengan seed tetap, agar semua contoh dapat dijalankan tanpa mengunduh apa pun.
- Setiap pelajaran menyebut apakah datanya **publik** atau **sintetis**. Data sintetis dipakai untuk memperlihatkan mekanisme, bukan untuk menyimpulkan dunia nyata.
- Keluaran yang ditampilkan berasal dari program yang dijalankan; lisensi dan sumber dataset publik dirujuk pada pelajaran yang memakainya.

## Dataset publik yang dipakai

| Dataset | Akses | Dipakai pada |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | `sklearn.datasets.load_breast_cancer` | [Klasifikasi](../classification/index.md), [Overfitting dan kebocoran](../fundamentals/overfitting-leakage.md) |
| Wine | `sklearn.datasets.load_wine` | [Klaster](../clustering/index.md), [Visualisasi](../fundamentals/data-visualization.md) |
| Diabetes | `sklearn.datasets.load_diabetes` | [Estimasi](../estimation/index.md), [Statistik](../fundamentals/statistics.md) |
| CO2 Mauna Loa mingguan (Keeling dan Whorf) | `statsmodels.datasets.co2` | [Forecasting](../forecasting/index.md) |

Dokumentasi dataset bawaan scikit-learn: [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html). Dataset CO2 dibundel dengan statsmodels dan dinyatakan domain publik pada dokumentasinya.

## Dataset sintetis yang dipakai

| Dataset | Pembangkit | Dipakai pada |
|---|---|---|
| Pelanggan berlangganan (churn) | `examples/fundamentals/customer_data.py` | Fundamentals, [proyek klasifikasi](../classification/project.md) |
| Harga rumah | `make_housing_data` pada proyek estimasi | [Proyek estimasi](../estimation/project.md) |
| Deret waktu penjualan | Skrip pengantar dan proyek forecasting | [Dasar deret waktu](../forecasting/ts-fundamentals.md), [proyek forecasting](../forecasting/project.md) |
| Pelanggan dan RFM | `examples/clustering/segment_data.py` dan proyek klaster | [Klaster](../clustering/index.md) |
| Transaksi keranjang belanja | `examples/association/basket_data.py` | [Aturan asosiasi](../association/index.md) |

## Sumber belajar

- [Dokumentasi scikit-learn](https://scikit-learn.org/stable/)
- [Dokumentasi pandas](https://pandas.pydata.org/docs/)
- [Dokumentasi NumPy](https://numpy.org/doc/)
- [Dokumentasi Matplotlib](https://matplotlib.org/stable/)
- [Dokumentasi Seaborn](https://seaborn.pydata.org/)
- [Dokumentasi statsmodels](https://www.statsmodels.org/stable/)
- [Dokumentasi mlxtend](https://rasbt.github.io/mlxtend/)

## Lihat juga

[Glosarium](glossary.md) · [Menjalankan Situs dan Contoh](python-environment.md)
