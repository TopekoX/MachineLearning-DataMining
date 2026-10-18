# Praproses, Pembagian Data, dan Baseline

## Tujuan pembelajaran

- [x] Menentukan praproses yang diperlukan klasifikasi berdasarkan temuan EDA.
- [x] Membagi data dengan *stratified split* sebelum apa pun dipelajari dari data.
- [x] Membuat *baseline* dan memakainya sebagai pembanding.
- [x] Menunjukkan pengaruh *scaling* pada model sederhana.

## Praproses dan rekayasa fitur untuk dataset ini

Konsep dasarnya sudah dibahas di [Praproses Data](../fundamentals/data-preprocessing.md) dan [Feature Engineering](../fundamentals/feature-engineering.md). Untuk Breast Cancer:

| Temuan EDA | Keputusan |
|---|---|
| Tidak ada data hilang, tidak ada duplikat | Tidak perlu imputasi atau penghapusan |
| Semua fitur numerik | Tidak perlu *encoding* |
| Rentang fitur sangat berbeda | `StandardScaler` untuk model berbasis jarak/gradien (Logistic Regression, KNN, SVM) |
| Banyak fitur sangat berkorelasi (radius, perimeter, area) | Tidak dihapus di bab ini; model berbeda menanggapinya berbeda |
| Rekayasa fitur | Tidak dilakukan; fitur sudah berupa ukuran turunan dan tujuan bab ini membandingkan model |

*Scaling* dimasukkan ke dalam `Pipeline` agar dihitung dari data latih saja.

## Pembagian data

Data dibagi 80% latih dan 20% uji dengan `stratify=y` dan `random_state` tetap. Data uji **tidak disentuh** sampai evaluasi akhir; pemilihan model memakai validasi silang pada data latih.

## Baseline

*Baseline* adalah model paling sederhana yang masih masuk akal. Model lain hanya berguna bila mengalahkannya.

1. **Baseline naif**: selalu menebak kelas mayoritas (`DummyClassifier`).
2. **Baseline sederhana**: *scaling* + Logistic Regression.

## Implementasi langkah demi langkah

Program `examples/classification/preprocessing_baseline.py` menjadi **pola dasar** yang dipakai ulang oleh semua pelajaran model berikutnya (pemuatan data dan pembagian data yang sama), jadi bagian ini dijelaskan paling rinci.

### Langkah 1: memuat data dan memisahkan data uji

**Tujuan.** Menyisihkan data uji sebelum mempelajari apa pun dari data, agar estimasi akhir jujur.

```python title="preprocessing_baseline.py" linenums="1"
--8<-- "classification/preprocessing_baseline.py:1:17"
```

| Kode | Penjelasan |
|---|---|
| `SEED = 42` | Satu angka seed yang dipakai di seluruh program agar hasil sama setiap dijalankan. Nilai 42 hanya konvensi; angka lain juga sah. |
| `load_breast_cancer(return_X_y=True, as_frame=True)` | Langsung menghasilkan `X` (tabel fitur) dan `y` (target) sebagai objek pandas. |
| `train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)` | Membagi data. Mengembalikan **empat** bagian dalam urutan tetap: `X_train, X_test, y_train, y_test`. |
| `test_size=0.2` | 20% data (114 sampel) menjadi data uji; 80% (455) data latih. |
| `stratify=y` | Menjaga proporsi kelas pada latih dan uji sama dengan data asli. |
| `y_train.mean()` | Karena `y` bernilai 0/1, rata-ratanya adalah proporsi kelas 1. |

```text
--8<-- "expected/classification/preprocessing_baseline.txt:1:2"
```

**Interpretasi.** 455 data latih dan 114 data uji (masing-masing dengan 30 fitur). Proporsi kelas 1 hampir sama (0,626 latih dan 0,632 uji) berkat `stratify`. Mulai sini, **data uji tidak disentuh** sampai evaluasi akhir.

### Langkah 2: *baseline* pertama, menebak kelas mayoritas

**Konsep.** *Baseline* adalah model paling sederhana yang masih masuk akal; model lain hanya berguna bila mengalahkannya. Pelatihan model scikit-learn selalu melalui dua tahap: **inisialisasi** (membuat objek dengan pengaturan) lalu **`fit`** (belajar dari data).

```python title="preprocessing_baseline.py" linenums="19"
--8<-- "classification/preprocessing_baseline.py:19:24"
```

| Kode | Penjelasan |
|---|---|
| `DummyClassifier(strategy="most_frequent")` | Inisialisasi: model yang selalu memprediksi kelas terbanyak di data latih. Belum belajar apa pun. |
| `dummy.fit(X_train, y_train)` | **Belajar**: mencatat kelas mayoritas dari `y_train`. Hanya data **latih** yang boleh masuk `fit`. |
| `dummy.predict(X_test)` | **Prediksi** untuk data uji; hasilnya array label (semua kelas 1 di sini). |
| `accuracy_score(y_test, pred)` | Bandingkan label **sebenarnya** (`y_test`) dengan **prediksi** (`pred`); urutan: sebenarnya dulu. |
| `f1_score(y_test, pred)` | F1 untuk kelas 1 (kelas positif bawaan). |

```text
--8<-- "expected/classification/preprocessing_baseline.txt:4:4"
```

**Interpretasi.** *Accuracy* 0,632 sama dengan proporsi kelas jinak. Nilai **F1 kelas 1 = 0,774** tampak lumayan padahal model tidak mempelajari apa pun: karena kelas 1 adalah mayoritas, menebaknya selalu menghasilkan *recall* 1,0. Ini contoh mengapa metrik harus dibaca bersama konteksnya. Hitung F1 kelas 0 untuk melihat bahwa nilainya 0.

### Langkah 3: *baseline* kedua, model sederhana dalam `Pipeline`

**Konsep.** Regresi logistik memerlukan fitur berskala sebanding. Kita menggabungkan `StandardScaler` dan `LogisticRegression` dalam satu `Pipeline`, lalu menilainya dengan **validasi silang** pada data latih (data uji tetap disimpan).

```python title="preprocessing_baseline.py" linenums="26"
--8<-- "classification/preprocessing_baseline.py:26:29"
```

| Kode | Penjelasan |
|---|---|
| `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` | Merangkai dua langkah berurutan. `fit` mempelajari rata-rata/simpangan baku (scaler) **dan** koefisien (model) dari data yang diberikan; `predict` menerapkan keduanya. |
| `cross_val_score(simple, X_train, y_train, cv=5, scoring="f1")` | Membagi data latih menjadi 5 lipatan; tiap lipatan bergantian menjadi data validasi sementara pipeline **dilatih ulang** pada 4 lipatan lain. Hasilnya 5 skor F1. |
| `cv_f1.mean()`, `cv_f1.std()` | Rata-rata dan sebaran 5 skor: rata-rata menunjukkan kualitas, sebaran menunjukkan kestabilan. |

```text
--8<-- "expected/classification/preprocessing_baseline.txt:5:5"
```

**Interpretasi.** F1 rata-rata 0,984 (± 0,010): jauh di atas *baseline* mayoritas. Karena scaler berada **di dalam** pipeline, ia dilatih ulang hanya pada 4 lipatan latih di tiap putaran, sehingga tidak ada kebocoran dari lipatan validasi.

### Langkah 4: seberapa penting *scaling*?

```python title="preprocessing_baseline.py" linenums="31"
--8<-- "classification/preprocessing_baseline.py:31:34"
```

Sama seperti Langkah 3, tetapi **tanpa** scaler (`max_iter=10000` memberi batas iterasi lebih longgar karena fitur tak terskala lebih sulit dioptimasi).

```text
--8<-- "expected/classification/preprocessing_baseline.txt:6:6"
```

**Interpretasi.** Tanpa *scaling*, F1 turun ke 0,962 (± 0,009), dibanding 0,984 dengan *scaling*. Pada regresi logistik, perbedaan skala mempersulit optimisasi dan membuat regularisasi memperlakukan fitur secara tidak setara. Selisih ini nyata tetapi tidak dramatis; pada KNN dan SVM dampaknya lebih besar.

??? example "Program lengkap (`preprocessing_baseline.py`)"
    ```python title="preprocessing_baseline.py" linenums="1"
    --8<-- "classification/preprocessing_baseline.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Membagi data setelah *scaling* atau imputasi.
    - Tidak memakai `stratify` pada data berlabel.
    - Mengubah `random_state` sampai skor uji "bagus".
    - Melewatkan *baseline* lalu tidak tahu apakah model sebenarnya berguna.

## Latihan

1. Ubah `test_size` menjadi 0,3 dan lihat proporsi kelas pada data uji.
2. Tambahkan *baseline* `DummyClassifier(strategy="stratified")`. Bagaimana F1-nya dibanding strategi mayoritas?
3. Hitung F1 untuk **kelas 0** (`pos_label=0`) pada *baseline* mayoritas. Mengapa hasilnya 0?

## Ringkasan

Keputusan praproses bersumber dari EDA. Pisahkan data uji dengan *stratified split* lebih dulu, masukkan *scaling* ke `Pipeline`, dan selalu bandingkan model dengan *baseline*.

Berikutnya: [Logistic Regression](logistic-regression.md).
