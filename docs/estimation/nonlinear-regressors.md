# KNN, Decision Tree, dan Random Forest Regressor

## Tujuan pembelajaran

- [x] Menjelaskan versi regresi dari KNN, pohon keputusan, dan Random Forest.
- [x] Mengatur kompleksitas masing-masing dengan hyperparameter kuncinya.
- [x] Menyadari batasan model berbasis pohon: prediksi konstan per daun dan tidak mengekstrapolasi.

!!! info "Prasyarat"
    Cara kerja dasar ketiga model dibahas pada bab Klasifikasi: [KNN](../classification/knn.md), [Decision Tree](../classification/decision-tree.md), [Random Forest](../classification/random-forest.md). Di sini hanya dibahas perbedaannya untuk regresi.

## Perbedaan untuk regresi

| Model | Prediksi | Kriteria pembagian | Hyperparameter kunci |
|---|---|---|---|
| KNN Regressor | Rata-rata target \(k\) tetangga terdekat | (jarak) | `n_neighbors`, `weights` |
| Decision Tree Regressor | Rata-rata target pada daun | Penurunan galat kuadrat (MSE) | `max_depth`, `min_samples_leaf` |
| Random Forest Regressor | Rata-rata prediksi semua pohon | Seperti pohon, dengan bootstrap dan subset fitur | `n_estimators`, `min_samples_leaf`, `max_features` |

Konsekuensi: prediksi pohon dan hutan **berbentuk tangga** (konstan di tiap daun) dan tidak pernah keluar dari rentang target latih, sehingga tidak dapat mengekstrapolasi tren. KNN juga perlu *scaling*.

## Implementasi langkah demi langkah

Dataset Diabetes dengan pembagian 80/20 dan validasi silang 5 lipatan (RMSE; makin kecil makin baik). Program `examples/estimation/nonlinear_regressors.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: fungsi bantu RMSE latih dan RMSE validasi silang

**Konsep.** Membandingkan galat pada data latih dan galat validasi silang menunjukkan *overfitting*: galat latih sangat kecil tetapi galat validasi besar.

```python title="nonlinear_regressors.py" linenums="11"
--8<-- "estimation/nonlinear_regressors.py:11:22"
```

`cv_rmse(model)` mengembalikan RMSE rata-rata validasi silang (tanda minus mengubah skor negatif menjadi positif). `train_rmse(model)` melatih model pada data latih, memprediksi **data latih yang sama**, dan menghitung RMSE-nya, yaitu galat pada data yang sudah dilihat model.

### Langkah 2: KNN Regressor, pengaruh `k`

**Konsep.** KNN regresi memprediksi **rata-rata target** dari `k` tetangga terdekat. `k=1` meniru tetangga tunggal (menghafal).

```python title="nonlinear_regressors.py" linenums="25"
--8<-- "estimation/nonlinear_regressors.py:25:29"
```

`KNeighborsRegressor(n_neighbors=k)` dipasang dalam pipeline dengan `StandardScaler` (jarak sensitif pada skala). Format `{train_rmse(m):<10.2f}` menampilkan angka dua desimal dengan lebar 10 rata kiri.

```text
--8<-- "expected/estimation/nonlinear_regressors.txt:1:6"
```

**Interpretasi.** `k=1` menghafal data latih (RMSE latih 0,00) tetapi RMSE CV 79,47, jauh lebih buruk daripada model sederhana (*overfitting*). RMSE CV terbaik pada `k=20` (58,71); `k=40` sedikit naik (59,53).

### Langkah 3: Decision Tree Regressor, pengaruh `max_depth`

**Konsep.** Pohon regresi membagi data dengan pertanyaan ya/tidak, dan prediksinya adalah **rata-rata target pada daun**. Prediksinya berbentuk tangga, hanya sebanyak jumlah daun.

```python title="nonlinear_regressors.py" linenums="31"
--8<-- "estimation/nonlinear_regressors.py:31:36"
```

`DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, ...)` membatasi kedalaman dan melarang daun berisi kurang dari 5 sampel. `m.fit(...).get_n_leaves()` melatih lalu menghitung jumlah daun.

```text
--8<-- "expected/estimation/nonlinear_regressors.txt:8:13"
```

**Interpretasi.** Makin dalam, galat latih turun terus (65,02 → 35,31) sedangkan RMSE CV terbaik pada `max_depth=2` (63,39) lalu naik menjadi 69,10 tanpa batas kedalaman (58 daun): pola *overfitting*. Pada tabel uji dipakai `max_depth=3` demi pohon yang tetap mudah dibaca; kedalaman 2 sedikit lebih baik pada validasi silang.

### Langkah 4: Random Forest Regressor, pengaruh jumlah pohon

**Konsep.** Hutan merata-ratakan prediksi banyak pohon yang dilatih pada sampel bootstrap, sehingga lebih stabil daripada satu pohon.

```python title="nonlinear_regressors.py" linenums="38"
--8<-- "estimation/nonlinear_regressors.py:38:42"
```

```text
--8<-- "expected/estimation/nonlinear_regressors.txt:15:19"
```

**Interpretasi.** Satu pohon memberi RMSE CV 70,83; 10 pohon 59,09; 100 pohon 57,88; 300 pohon 57,79. Menambah pohon memperbaiki lalu mendatar.

### Langkah 5: evaluasi pada data uji

```python title="nonlinear_regressors.py" linenums="44"
--8<-- "estimation/nonlinear_regressors.py:44:54"
```

Tiga model dengan pilihan hyperparameter yang wajar disimpan dalam kamus, dilatih pada data latih, dan diuji **sekali** pada data uji dengan MAE, RMSE, dan R². (Pilihan `k=20` dan `max_depth=3` diambil dari langkah sebelumnya.)

```text
--8<-- "expected/estimation/nonlinear_regressors.txt:21:24"
```

**Interpretasi.** KNN (`k=20`): MAE 45,45, RMSE 55,21, R² 0,425; pohon (`depth=3`): MAE 45,18, RMSE 55,63, R² 0,416; Random Forest: MAE 43,36, RMSE 53,52, R² 0,459. Ketiganya berdekatan dan hampir setara dengan regresi linear (RMSE uji 53,85). Pada dataset kecil dan berderau ini, fleksibilitas ekstra tidak memberi keuntungan besar.

??? example "Program lengkap (`nonlinear_regressors.py`)"
    ```python title="nonlinear_regressors.py" linenums="1"
    --8<-- "estimation/nonlinear_regressors.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai KNN tanpa *scaling* (di Diabetes fitur sudah diskalakan, tetapi tidak di data lain).
    - Mengira pohon atau hutan dapat memperkirakan nilai di luar rentang latih.
    - Memilih kedalaman pohon dari galat latih.
    - Membandingkan model berdasarkan satu angka uji pada data kecil.

## Latihan

1. Ubah `weights="distance"` pada KNN. Apakah RMSE CV berubah?
2. Tambahkan `max_features=0.5` pada Random Forest.
3. Buat prediksi tiap model untuk satu sampel dengan `bmi` jauh lebih tinggi dari data latih. Bagaimana perilaku pohon vs regresi linear?

## Ringkasan

Versi regresi KNN, pohon, dan Random Forest memprediksi rata-rata target lokal. Mereka menangkap non-linearitas tetapi tidak mengekstrapolasi, dan pada data kecil belum tentu mengalahkan model linear.

Berikutnya: [Perbandingan Model Regresi](model-comparison.md).
