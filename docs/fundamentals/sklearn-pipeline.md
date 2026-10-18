# Workflow scikit-learn dan Pipeline

## Tujuan pembelajaran

- [x] Memahami API scikit-learn: `fit`, `transform`, `predict`, `score`.
- [x] Membangun `Pipeline` dan `ColumnTransformer` untuk data campuran numerik dan kategorik.
- [x] Menjelaskan bagaimana `Pipeline` mencegah kebocoran data pada validasi silang.

## Konsep dasar

Objek scikit-learn berbagi antarmuka yang konsisten:

| Jenis | Metode utama | Contoh |
|---|---|---|
| *Transformer* | `fit(X)`, `transform(X)` | `StandardScaler`, `SimpleImputer`, `OneHotEncoder` |
| *Estimator/model* | `fit(X, y)`, `predict(X)`, `score(X, y)` | `LogisticRegression`, `RandomForestClassifier` |

- `fit` **mempelajari** parameter dari data (mis. rata-rata untuk *scaler*, koefisien untuk model).
- `transform`/`predict` **menerapkan** hasil belajar itu.

`Pipeline` merangkai langkah berurutan menjadi satu objek. `ColumnTransformer` menerapkan praproses berbeda pada kelompok kolom berbeda. Keuntungannya:

1. **Mencegah kebocoran data**: pada validasi silang, seluruh pipeline di-*fit* ulang hanya pada data latih tiap lipatan.
2. **Reproduksibel dan rapi**: satu objek untuk praproses dan model, dapat disimpan dan dipakai ulang.
3. **Memudahkan penyetelan** hyperparameter praproses dan model sekaligus.

```mermaid
flowchart LR
    A[Data mentah] --> B[ColumnTransformer<br/>numerik: impute + scale<br/>kategorik: impute + one-hot]
    B --> C[Model]
    C --> D[Prediksi]
```

## Implementasi langkah demi langkah

Memakai data pelanggan sintetis ([penjelasan](pandas.md#dataset-yang-dipakai)) untuk memprediksi `churn`. Simpan `customer_data.py` pada folder yang sama. Program lengkap ada di akhir bagian ini.

### Langkah 1: memisahkan `X`, `y`, dan data uji

**Tujuan.** Menyiapkan fitur dan target, lalu menyisihkan data uji **sebelum** apa pun dipelajari dari data.

```python title="sklearn_pipeline.py" linenums="1"
--8<-- "fundamentals/sklearn_pipeline.py:1:16"
```

| Kode | Penjelasan |
|---|---|
| `X, y = df.drop(columns="churn"), df["churn"]` | `X` = semua kolom **kecuali** target (tabel fitur); `y` = kolom target (`Series`). |
| `train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)` | Mengembalikan empat bagian dalam urutan `X_train, X_test, y_train, y_test`. 20% menjadi data uji; `stratify=y` menjaga proporsi churn; `random_state` membuat hasilnya dapat diulang. |
| `numeric`, `categorical` | Dua daftar nama kolom yang akan diperlakukan berbeda. |

### Langkah 2: praproses berbeda untuk kolom berbeda (`ColumnTransformer`)

**Konsep.** Kolom angka perlu imputasi dan *scaling*; kolom teks perlu imputasi dan *one-hot*. `ColumnTransformer` menerapkan rangkaian langkah berbeda pada kelompok kolom berbeda, lalu menyatukan hasilnya.

```python title="sklearn_pipeline.py" linenums="18"
--8<-- "fundamentals/sklearn_pipeline.py:18:24"
```

Setiap entri `ColumnTransformer` berbentuk `(nama, transformer, kolom)`:

- `("num", Pipeline([imputasi median, standardisasi]), numeric)`: untuk kolom numerik, jalankan imputasi median, lalu `StandardScaler`.
- `("cat", Pipeline([imputasi modus, one-hot]), categorical)`: untuk kolom kategorik, jalankan imputasi modus, lalu `OneHotEncoder(handle_unknown="ignore")` (kategori baru saat prediksi tidak menimbulkan galat).

**Istilah.** *Transformer* adalah objek dengan `fit` dan `transform`; *estimator* adalah objek yang dapat `fit` (belajar). `Pipeline` sendiri adalah estimator yang merangkai beberapa langkah berurutan.

### Langkah 3: merangkai praproses dan model menjadi satu `Pipeline`

```python title="sklearn_pipeline.py" linenums="26"
--8<-- "fundamentals/sklearn_pipeline.py:26:27"
```

`Pipeline([("preprocess", preprocess), ("clf", LogisticRegression(max_iter=1000))])` merangkai praproses dan model. Tiap langkah berupa pasangan `(nama, objek)`; **langkah terakhir adalah model**. Pada titik ini belum ada yang dilatih: kita baru **menginisialisasi** objek. `max_iter=1000` memberi optimisasi regresi logistik batas 1000 iterasi agar sempat konvergen.

**Mengapa pipeline?** Dengan satu objek, `fit` mempelajari imputer, scaler, encoder, dan model **hanya dari data yang diberikan** (data latih), dan `predict` menerapkan semuanya secara konsisten pada data baru. Inilah cara mencegah kebocoran data.

### Langkah 4: validasi silang pada seluruh pipeline

```python title="sklearn_pipeline.py" linenums="29"
--8<-- "fundamentals/sklearn_pipeline.py:29:31"
```

`cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")` membagi **data latih** menjadi 5 lipatan. Untuk tiap lipatan ia **melatih ulang seluruh pipeline** (imputer, scaler, encoder, model) pada 4 lipatan lalu menilai pada lipatan sisanya. Hasilnya 5 skor ROC-AUC; `scores.mean()` dan `scores.std()` merangkumnya. Karena praproses ikut dilatih ulang tiap lipatan, tidak ada informasi lipatan validasi yang bocor.

```text
--8<-- "expected/fundamentals/sklearn_pipeline.txt:1:1"
```

**Interpretasi.** ROC-AUC rata-rata 0,709 dengan simpangan baku 0,040: kemampuan memisahkan churn dan non-churn lumayan tetapi tidak tinggi, dan skor bervariasi antar lipatan.

### Langkah 5: melatih dan menguji

```python title="sklearn_pipeline.py" linenums="33"
--8<-- "fundamentals/sklearn_pipeline.py:33:34"
```

`model.fit(X_train, y_train)` melatih seluruh pipeline pada data latih. `model.score(X_test, y_test)` memprediksi `X_test` dan membandingkannya dengan `y_test`; untuk klasifikasi, `score` bawaan adalah **akurasi**.

```text
--8<-- "expected/fundamentals/sklearn_pipeline.txt:2:2"
```

**Interpretasi.** Akurasi uji 0,742. Menebak "bertahan" untuk semua orang memberi sekitar 0,66 (proporsi kelas 0 = 0,661), sehingga model lebih baik dari tebakan sederhana, tetapi tidak jauh.

### Langkah 6: melihat isi pipeline

```python title="sklearn_pipeline.py" linenums="36"
--8<-- "fundamentals/sklearn_pipeline.py:36:39"
```

`model.named_steps["preprocess"]` mengambil satu langkah berdasarkan namanya; `get_feature_names_out()` menamai kolom hasil praproses (awalan `num__` dan `cat__` menunjukkan cabang asalnya). `list(model.named_steps)` menampilkan nama langkah.

```text
--8<-- "expected/fundamentals/sklearn_pipeline.txt:3:5"
```

**Hasil.** Praproses menghasilkan 11 fitur: 5 numerik + 4 kota + 2 paket. Satu kolom `city` menjadi empat kolom *one-hot*, dan `plan` menjadi dua.

??? example "Program lengkap (`sklearn_pipeline.py`)"
    ```python title="sklearn_pipeline.py" linenums="1"
    --8<-- "fundamentals/sklearn_pipeline.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memanggil `fit_transform` pada seluruh data sebelum `cross_val_score` (kebocoran).
    - Memanggil `fit` pada data uji.
    - Lupa bahwa urutan kolom saat prediksi harus sama dengan saat pelatihan (pipeline dengan `DataFrame` dan nama kolom menjaga ini).
    - Menggunakan `score` default (*accuracy*) tanpa memikirkan metrik yang sesuai.

## Latihan

1. Ganti `LogisticRegression` dengan `RandomForestClassifier` dan hapus `StandardScaler`. Apakah ROC-AUC CV berubah?
2. Tambahkan `DummyClassifier(strategy="most_frequent")` sebagai *baseline* dengan pipeline yang sama.
3. Cetak `model.named_steps["clf"].coef_`. Fitur mana yang koefisiennya terbesar (dalam nilai mutlak)?
4. Mengapa `SimpleImputer` ada di dalam pipeline dan bukan dijalankan sebelum split?

## Ringkasan

`Pipeline` + `ColumnTransformer` menyatukan praproses dan model dalam satu objek yang di-*fit* hanya pada data latih, sehingga mencegah kebocoran dan membuat alur kerja rapi, dapat diulang, dan mudah disimpan.

Berikutnya: [Overfitting, Underfitting, dan Kebocoran Data](overfitting-leakage.md).
