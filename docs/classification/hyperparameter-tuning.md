# Optimasi Hyperparameter

## Tujuan pembelajaran

- [x] Membedakan parameter (dipelajari) dan hyperparameter (ditetapkan sebelum pelatihan).
- [x] Memakai `GridSearchCV` dan `RandomizedSearchCV` dengan `Pipeline`.
- [x] Menjelaskan mengapa skor terbaik dari pencarian bersifat optimis dan bagaimana *nested cross-validation* memperbaikinya.

## Konsep dasar

**Parameter** (misalnya koefisien regresi logistik) dipelajari dari data. **Hyperparameter** (misalnya `C`, `gamma`, `max_depth`) ditetapkan sebelum pelatihan. Optimasi hyperparameter mencari kombinasi yang memberi skor validasi terbaik.

| Metode | Cara kerja | Kapan dipakai |
|---|---|---|
| *Grid Search* | Mencoba semua kombinasi pada kisi | Sedikit hyperparameter, ruang kecil |
| *Random Search* | Mencoba sampel acak kombinasi | Ruang besar; sering lebih efisien per percobaan |
| Pencarian lanjutan (Bayesian, *successive halving*) | Mengarahkan pencarian | Model mahal; di luar cakupan lesson ini |

Aturan penting:

1. **Pencarian hanya memakai data latih** (validasi silang di dalamnya). Data uji tetap disimpan.
2. Gunakan **`Pipeline`**, sehingga praproses ikut dilatih ulang per lipatan. Nama parameter memakai format `<nama_langkah>__<parameter>`, misalnya `svc__C`.
3. Tentukan `scoring` sesuai tujuan (F1, recall, PR-AUC), bukan hanya *accuracy*.
4. `refit=True` (bawaan) melatih ulang model terbaik pada seluruh data latih.

## Implementasi langkah demi langkah

Dataset Breast Cancer; pemuatan dan pembagian data sama dengan [pelajaran dasar](preprocessing-baseline.md#langkah-1-memuat-data-dan-memisahkan-data-uji). Program lengkap ada di akhir bagian ini.

### Langkah 1: Grid Search untuk SVM

**Tujuan.** Mencari kombinasi `C` dan `gamma` terbaik dari kisi 4 × 4 = 16 kombinasi, dinilai dengan validasi silang pada data latih.

```python title="tuning.py" linenums="17"
--8<-- "classification/tuning.py:17:31"
```

| Kode | Penjelasan |
|---|---|
| `Pipeline([("scale", StandardScaler()), ("svc", SVC())])` | Pipeline dengan **nama** langkah `scale` dan `svc`. Nama dipakai untuk menunjuk parameter. |
| `param_grid={"svc__C": [...], "svc__gamma": [...]}` | Kisi parameter. Format `<nama_langkah>__<parameter>` (dua garis bawah) menunjuk parameter `C` milik langkah `svc`. Semua kombinasi (16) dicoba. |
| `scoring="f1"` | Metrik yang dioptimalkan. Pilih sesuai tujuan, bukan sekadar akurasi. |
| `cv=cv` | Validasi silang 5 lipatan berstrata untuk setiap kombinasi (16 × 5 = 80 pelatihan). |
| `grid.fit(X_train, y_train)` | Menjalankan pencarian. **Hanya data latih** yang dipakai; data uji tidak tersentuh. Setelah selesai, kombinasi terbaik dilatih ulang pada seluruh data latih (`refit=True` bawaan). |

```text
--8<-- "expected/classification/tuning.txt:1:4"
```

### Langkah 2: membaca hasil pencarian

```python title="tuning.py" linenums="32"
--8<-- "classification/tuning.py:32:40"
```

`grid.best_params_` = kombinasi terbaik; `grid.best_score_` = skor validasi silang rata-rata kombinasi itu; `grid.cv_results_` = tabel hasil semua kombinasi, yang diubah menjadi `DataFrame` lalu `nsmallest(3, "rank_test_score")` mengambil tiga peringkat teratas (peringkat 1 = terbaik). Baris terakhir menghitung skor SVM **bawaan** (tanpa penyetelan) sebagai pembanding.

```text
--8<-- "expected/classification/tuning.txt:5:9"
```

**Interpretasi.** Kombinasi terbaik `C=10`, `gamma=0,01` dengan F1 CV 0,981, sedangkan SVM bawaan sudah memberi 0,976. **Peningkatan 0,005 jauh lebih kecil daripada simpangan baku antar lipatan (0,012)**, jadi penyetelan di sini tidak memberi keuntungan yang meyakinkan. Tiga kombinasi teratas hanya berbeda 0,003.

### Langkah 3: Random Search untuk Random Forest

**Konsep.** Bila ruang parameter besar, mencoba semua kombinasi tidak praktis. *Random Search* mengambil sejumlah kombinasi **acak** (`n_iter`) dari rentang yang ditentukan.

```python title="tuning.py" linenums="42"
--8<-- "classification/tuning.py:42:56"
```

| Kode | Penjelasan |
|---|---|
| `randint(100, 400)` | Distribusi bilangan bulat acak antara 100 dan 399 (dari `scipy.stats`). Daftar seperti `[None, 3, 5, 10]` berarti pilih acak salah satu. |
| `n_iter=15` | Hanya 15 kombinasi acak yang dicoba. |
| `random_state=SEED` | Membuat pengambilan acaknya dapat diulang. |

```text
--8<-- "expected/classification/tuning.txt:11:13"
```

**Interpretasi.** Skor terbaik 0,973 dengan `max_depth=10`, `n_estimators=121`; di bawah SVM pada data ini.

### Langkah 4: estimasi jujur dengan *nested cross-validation*

**Konsep.** `best_score_` dipilih dari banyak kombinasi pada lipatan yang sama, sehingga cenderung **optimis**. *Nested CV* menilai **prosedur** pencarian itu sendiri: pencarian diulang di dalam tiap lipatan luar yang tidak dilihat pencarian.

```python title="tuning.py" linenums="58"
--8<-- "classification/tuning.py:58:60"
```

`cross_val_score(grid, ...)` memperlakukan seluruh `GridSearchCV` sebagai satu model: untuk tiap lipatan luar, pencarian dijalankan pada lipatan latih luar, lalu hasilnya dinilai pada lipatan validasi luar. `StratifiedKFold(..., random_state=1)` memakai pembagian berbeda dari pencarian dalam.

```text
--8<-- "expected/classification/tuning.txt:15:15"
```

**Interpretasi.** Estimasi F1 0,979 (± 0,007) untuk prosedur "Grid Search SVM", sedikit di bawah skor terbaik grid (0,981) karena skor terbaik cenderung optimis.

### Langkah 5: data uji sekali

```python title="tuning.py" linenums="62"
--8<-- "classification/tuning.py:62:64"
```

`grid.predict(X_test)` otomatis memakai model terbaik (`best_estimator_`) yang sudah dilatih ulang pada seluruh data latih.

```text
--8<-- "expected/classification/tuning.txt:16:16"
```

**Interpretasi.** Akurasi 0,982 dan F1 0,986 untuk model terpilih. Pesan utama: penyetelan berguna terutama bila model awal jauh dari optimal (seperti `gamma` ekstrem pada [SVM](svm.md)); pada model yang sudah cukup baik, hasilnya sering marginal, dan waktu lebih baik dipakai memperbaiki data atau fitur.

??? example "Program lengkap (`tuning.py`)"
    ```python title="tuning.py" linenums="1"
    --8<-- "classification/tuning.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Menyetel hyperparameter dengan data uji.
    - Melaporkan `best_score_` sebagai estimasi performa akhir.
    - Melakukan *scaling* sebelum pencarian sehingga bocor ke lipatan validasi.
    - Menyetel terlalu banyak kombinasi pada data kecil sehingga "overfit ke validasi".
    - Membandingkan model yang disetel dengan model lain yang tidak disetel secara adil.

## Latihan

1. Perluas grid `C` ke `[0.01, 100, 1000]`. Apakah kombinasi terbaik berubah?
2. Ganti `scoring` menjadi `recall` untuk kelas ganas dengan `make_scorer(recall_score, pos_label=0)`.
3. Jalankan `RandomizedSearchCV` dengan `n_iter=5` dan `n_iter=40`. Seberapa jauh skor terbaiknya berbeda?
4. Jelaskan dengan kata-kata Anda mengapa `best_score_` optimis.

## Ringkasan

Hyperparameter disetel dengan pencarian bervalidasi silang pada data latih, di dalam `Pipeline`. Skor terbaik pencarian cenderung optimis; *nested CV* dan data uji (sekali) memberi gambaran lebih jujur. Peningkatan kecil dibanding ketidakpastian tidak perlu dianggap bermakna.

Berikutnya: [Menyimpan Model, Deployment, dan Monitoring](saving-deployment.md).
