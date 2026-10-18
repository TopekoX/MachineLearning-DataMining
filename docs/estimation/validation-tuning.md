# Validasi Silang dan Optimasi Hyperparameter

## Tujuan pembelajaran

- [x] Memakai validasi silang yang sesuai untuk regresi.
- [x] Menyetel Ridge (`alpha`) dan Random Forest dengan pencarian bervalidasi silang.
- [x] Membedakan skor terbaik pencarian dan estimasi jujur (*nested CV*).

!!! info "Prasyarat"
    Konsep Grid/Random Search dan *nested CV* dijelaskan di [Optimasi Hyperparameter (Klasifikasi)](../classification/hyperparameter-tuning.md). Di sini fokus pada hal yang spesifik regresi.

## Validasi silang untuk regresi

- Gunakan `KFold(shuffle=True, random_state=...)` atau `RepeatedKFold`; tidak ada stratifikasi kelas. Untuk target yang sangat miring, dapat distratifikasi pada kelompok (*bin*) target.
- Skor *scikit-learn* dibuat "makin besar makin baik", sehingga galat dinyatakan negatif: `neg_root_mean_squared_error`, `neg_mean_absolute_error`. Ubah tanda saat melaporkan.
- Pilih metrik pencarian sesuai tujuan (RMSE bila galat besar mahal, MAE bila galat tipikal).
- Data berurutan waktu **tidak** boleh memakai `KFold` acak (lihat bab [Forecasting](../forecasting/index.md)).

## Implementasi langkah demi langkah

Dataset Diabetes. Pencarian pertama menyetel `alpha` Ridge pada fitur polinomial derajat 2 (65 fitur); pencarian kedua menyetel Random Forest. Program `examples/estimation/tuning.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: patokan sebelum penyetelan

**Tujuan.** Mengetahui skor model dengan hyperparameter bawaan agar hasil penyetelan dapat dinilai.

```python title="tuning.py" linenums="18"
--8<-- "estimation/tuning.py:18:28"
```

| Kode | Penjelasan |
|---|---|
| `scoring = "neg_root_mean_squared_error"` | Skor scikit-learn selalu "lebih besar lebih baik", sehingga galat dinyatakan negatif. Metrik ini dipakai di semua pencarian. |
| `Pipeline([("poly", ...), ("scale", ...), ("ridge", Ridge())])` | Fitur polinomial derajat 2, standardisasi, lalu Ridge (bawaan `alpha=1`). Nama langkah dipakai untuk menunjuk parameter. |
| `-cross_val_score(...).mean()` | RMSE rata-rata validasi silang; minus mengembalikannya menjadi positif. |

```text
--8<-- "expected/estimation/tuning.txt:1:1"
```

**Hasil.** RMSE CV 60,47 dengan `alpha=1`.

### Langkah 2: Grid Search untuk `alpha`

```python title="tuning.py" linenums="30"
--8<-- "estimation/tuning.py:30:34"
```

`np.logspace(-1, 4, 11)` menghasilkan 11 nilai yang berjarak sama pada skala logaritmik dari \(10^{-1}\) sampai \(10^{4}\) (0,1; 0,32; 1; ...; 10.000). Skala logaritmik dipakai karena `alpha` bekerja secara multiplikatif. `"ridge__alpha"` menunjuk parameter `alpha` milik langkah `ridge`. `grid.best_params_["ridge__alpha"]` adalah `alpha` terbaik dan `-grid.best_score_` RMSE CV-nya.

```text
--8<-- "expected/estimation/tuning.txt:3:4"
```

**Hasil.** `alpha` terbaik 100 dengan RMSE CV 56,92, turun dari 60,47.

### Langkah 3: Random Search untuk Random Forest

```python title="tuning.py" linenums="36"
--8<-- "estimation/tuning.py:36:45"
```

`RandomizedSearchCV` mencoba 15 kombinasi **acak** dari ruang yang ditentukan: `randint(100, 400)` memilih bilangan bulat acak, sedangkan daftar seperti `[None, 3, 5, 8]` memilih salah satu elemennya. `max_features=[0.3, 0.6, 1.0]` berarti proporsi fitur yang dipertimbangkan pada tiap pembagian. `.fit(X_train, y_train)` dipanggil langsung di akhir rangkaian.

```text
--8<-- "expected/estimation/tuning.txt:6:8"
```

**Hasil.** Terbaik: `max_depth=8`, `max_features=0,6`, `min_samples_leaf=9`, `n_estimators=101`, RMSE CV 57,36, hampir sama dengan Ridge yang disetel.

### Langkah 4: *nested cross-validation*

**Konsep.** `best_score_` dipilih dari banyak kandidat pada lipatan yang sama, sehingga cenderung optimis. *Nested CV* menilai **prosedur** pencarian: pencarian diulang di dalam tiap lipatan luar yang tidak dilihat pencarian.

```python title="tuning.py" linenums="47"
--8<-- "estimation/tuning.py:47:49"
```

```text
--8<-- "expected/estimation/tuning.txt:10:10"
```

**Hasil.** RMSE 57,20 (± 3,30), sedikit lebih buruk daripada `best_score_` 56,92: skor terbaik yang dipilih dari banyak kandidat cenderung optimis.

### Langkah 5: data uji dan tiga `alpha` teratas

```python title="tuning.py" linenums="51"
--8<-- "estimation/tuning.py:51:58"
```

Data uji dipakai **sekali** untuk dua model terpilih (`search.predict` otomatis memakai model terbaik hasil `refit`). Baris terakhir menampilkan tiga `alpha` dengan RMSE CV terkecil dari `grid.cv_results_` untuk melihat seberapa landai kurvanya.

```text
--8<-- "expected/estimation/tuning.txt:11:18"
```

## Interpretasi keseluruhan

- **Penyetelan Ridge**: dengan `alpha=1` (bawaan), RMSE CV 60,47; Grid Search memilih `alpha=100` dengan RMSE CV 56,92. Tiga teratas: 100 (56,92), 31,6 (57,43), dan 10 (58,60): kurva cukup landai di sekitar optimum.
- **Random Forest**: RMSE CV 57,36, hampir sama dengan Ridge yang disetel.
- **Nested CV** untuk prosedur "Grid Search Ridge": 57,20 (± 3,30), sedikit lebih buruk daripada `best_score_` 56,92.
- **Data uji**, sekali: Ridge MAE 41,57 dan RMSE 51,57; Random Forest MAE 42,89 dan RMSE 52,80. Selisihnya kecil dan tidak cukup untuk menyatakan pemenang.
- **Konteks**: regresi linear biasa pada 10 fitur asli sudah memberi RMSE CV sekitar 55–56. Penyetelan memperbaiki model polinomial yang berlebihan dari 60,47 ke 56,92, tetapi **tidak** mengalahkan model sederhana. Menyetel model yang kurang cocok tidak menggantikan pemilihan model dan fitur yang baik.

??? example "Program lengkap (`tuning.py`)"
    ```python title="tuning.py" linenums="1"
    --8<-- "estimation/tuning.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Mencari `alpha` pada skala linear (gunakan `np.logspace`).
    - Melaporkan `best_score_` sebagai performa akhir.
    - Lupa mengubah tanda skor negatif.
    - Menyetel dengan data uji.
    - Menyetel banyak parameter pada data kecil sehingga *overfit* ke lipatan validasi.

## Latihan

1. Perluas grid `alpha` ke `np.logspace(-2, 5, 15)`. Apakah optimum bergeser?
2. Ganti metrik pencarian menjadi `neg_mean_absolute_error`.
3. Ulangi Random Search dengan `random_state` berbeda. Seberapa stabil parameter terbaiknya?

## Ringkasan

Gunakan `KFold` acak, metrik negatif untuk galat, skala logaritmik untuk `alpha`, dan *nested CV* atau data uji untuk estimasi jujur. Penyetelan membantu tetapi tidak menggantikan pemilihan model yang tepat.

Berikutnya: [Menyimpan Model, Deployment, dan Monitoring](saving-deployment.md).
