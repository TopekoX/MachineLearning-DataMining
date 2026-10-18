# Pembersihan dan Praproses Data

## Tujuan pembelajaran

- [x] Menangani duplikat, data hilang, dan *outlier* dengan alasan yang jelas.
- [x] Menerapkan *one-hot encoding* pada variabel kategorik.
- [x] Memahami kapan dan bagaimana melakukan *scaling*.
- [x] Mengetahui bahwa praproses harus "belajar" hanya dari data latih.

## Konsep dasar

Model hanya sebaik data yang dimasukkan. Praproses membuat data bersih dan sesuai kebutuhan algoritma. Ada dua jenis langkah:

- **Tanpa parameter yang dipelajari** (misalnya menghapus duplikat): aman dilakukan sebelum split.
- **Dengan parameter yang dipelajari dari data** (median untuk imputasi, rata-rata dan simpangan baku untuk *scaling*, kategori untuk *encoding*): harus di-*fit* **hanya pada data latih**, lalu diterapkan ke data validasi/uji. Jika tidak, terjadi kebocoran data (lihat [Overfitting, Underfitting, dan Kebocoran Data](overfitting-leakage.md)).

## Langkah-langkah

### Data hilang

Strategi: hapus baris (bila sedikit dan acak), atau **imputasi**: median (numerik, tahan *outlier*), rata-rata, modus (kategorik), atau konstanta. Pertimbangkan *mengapa* data hilang: bila ketiadaan itu sendiri informatif, tambahkan kolom indikator.

### *Outlier*

Tentukan dulu apakah itu kesalahan input, kasus langka namun valid, atau bagian ekor distribusi. Pilihan: biarkan (untuk model tahan *outlier* seperti pohon), *capping* (*winsorizing*), transformasi log, atau hapus bila jelas salah.

### *Encoding* kategorik

*One-hot encoding* mengubah satu kolom kategorik menjadi kolom 0/1 per kategori. Hindari mengubah kategori nominal (kota) menjadi 1, 2, 3, karena model akan menganggap ada urutan dan jarak.

### *Scaling*

$$
z = \frac{x - \mu}{\sigma} \quad\text{(standardisasi)}, \qquad
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \quad\text{(min-max)}
$$

\(\mu\) dan \(\sigma\) adalah rata-rata dan simpangan baku fitur **pada data latih**; \(x_{\min}\), \(x_{\max}\) adalah nilai minimum dan maksimum fitur pada data latih. Standardisasi menghasilkan rata-rata 0 dan simpangan baku 1; min-max menghasilkan rentang 0–1 dan sangat dipengaruhi *outlier*.

| Algoritma | Perlu *scaling*? |
|---|---|
| KNN, SVM, k-Means, DBSCAN, regresi berregularisasi, regresi logistik | Ya |
| Decision Tree, Random Forest | Tidak |

## Implementasi langkah demi langkah

Memakai data pelanggan sintetis ([penjelasan](pandas.md#dataset-yang-dipakai)). Program lengkap ada di akhir bagian ini.

!!! warning "Perhatikan urutan pada proyek nyata"
    Contoh ini menjalankan imputasi dan *scaling* pada **seluruh data** hanya agar efek tiap langkah terlihat jelas. Pada proyek nyata, langkah yang belajar dari data (imputasi, *encoding*, *scaling*) dilakukan di dalam `Pipeline` **setelah** membagi data (lihat [Workflow scikit-learn dan Pipeline](sklearn-pipeline.md)).

### Langkah 1: menghapus duplikat

**Tujuan.** Baris yang sama persis memberi bobot ganda pada satu pengamatan dan dapat membuat satu baris muncul di data latih sekaligus data uji.

```python title="preprocessing.py" linenums="1"
--8<-- "fundamentals/preprocessing.py:1:12"
```

`drop_duplicates()` membuang baris yang identik dengan baris sebelumnya; `reset_index(drop=True)` menomori ulang indeks dari 0 (tanpa menyimpan indeks lama sebagai kolom).

```text
--8<-- "expected/fundamentals/preprocessing.txt:1:2"
```

**Hasil.** 5 baris duplikat dibuang: 605 menjadi 600 baris.

### Langkah 2: mengisi data hilang (imputasi)

**Konsep.** *Imputasi* mengisi sel kosong dengan nilai pengganti yang dihitung dari data. Pilihan: **median** untuk angka (tahan terhadap *outlier*), **modus** (nilai paling sering) untuk kategori.

```python title="preprocessing.py" linenums="14"
--8<-- "fundamentals/preprocessing.py:14:21"
```

| Kode | Penjelasan |
|---|---|
| `SimpleImputer(strategy="median")` | **Menginisialisasi** objek imputer (belum belajar apa pun). `most_frequent` = modus. |
| `fit_transform(df[["income"]])` | `fit` menghitung median `income`; `transform` mengisi sel kosong dengannya. `fit_transform` melakukan keduanya sekaligus. Perhatikan **dua kurung siku**: scikit-learn menuntut masukan 2 dimensi. |
| `.ravel()` | Meratakan hasil 2 dimensi `(n, 1)` kembali menjadi 1 dimensi `(n,)` agar bisa disimpan sebagai satu kolom. |
| `income_imputer.statistics_` | Nilai yang **dipelajari** imputer (di sini median). Atribut berakhiran `_` selalu berisi hasil pelatihan. |

```text
--8<-- "expected/fundamentals/preprocessing.txt:3:3"
```

**Hasil dan interpretasi.** Tidak ada `NaN` tersisa; median `income` = 6.004.000. Median dipilih karena rata-rata terdistorsi *outlier* (lihat [Statistik](statistics.md)). Median ini dihitung dari **data yang diberikan**: jika data uji ikut dihitung, informasi uji bocor ke praproses.

### Langkah 3: menangani *outlier*

**Konsep.** Batas atas IQR dipakai untuk **membatasi** (*capping*, atau *winsorizing*) nilai ekstrem, bukan menghapus barisnya, sehingga jumlah data tetap.

```python title="preprocessing.py" linenums="23"
--8<-- "fundamentals/preprocessing.py:23:27"
```

`df["income"].clip(upper=upper)` mengganti semua nilai di atas `upper` dengan `upper`. Hasilnya disimpan ke kolom **baru** `income_capped` sehingga kolom asli tetap ada untuk dibandingkan.

```text
--8<-- "expected/fundamentals/preprocessing.txt:4:4"
```

**Hasil.** 29 nilai `income` melewati batas atas dan dibatasi. Apakah nilai ekstrem itu kesalahan atau pelanggan sah yang kaya perlu dijawab dengan pemahaman bisnis; `clip` hanya satu pilihan.

### Langkah 4: *one-hot encoding* untuk kategori

**Konsep.** Model hanya menerima angka. Memberi kota nomor 1, 2, 3, 4 menyiratkan urutan dan jarak yang tidak ada. *One-hot* membuat satu kolom 0/1 per kategori.

```python title="preprocessing.py" linenums="29"
--8<-- "fundamentals/preprocessing.py:29:33"
```

| Parameter | Arti |
|---|---|
| `handle_unknown="ignore"` | Bila saat prediksi muncul kota yang belum pernah dilihat, kolomnya diisi 0 semua alih-alih galat. |
| `sparse_output=False` | Menghasilkan array biasa (bukan format *sparse*) agar mudah dibaca. |
| `encoder.categories_` | Daftar kategori yang dipelajari, menentukan urutan kolom. |

```text
--8<-- "expected/fundamentals/preprocessing.txt:5:9"
```

**Hasil.** Empat kolom untuk urutan `Bandung, Jakarta, Medan, Surabaya`. Baris pertama `[0, 0, 0, 1]` berarti Surabaya; baris kedua `[0, 1, 0, 0]` berarti Jakarta. Tiap baris hanya memiliki satu angka 1.

### Langkah 5: *scaling*

**Konsep.** Fitur dengan rentang berbeda jauh (`age` puluhan, `income` jutaan) diubah ke skala sebanding. `StandardScaler` membuat rata-rata 0 dan simpangan baku 1; `MinMaxScaler` memampatkan ke rentang 0–1.

```python title="preprocessing.py" linenums="35"
--8<-- "fundamentals/preprocessing.py:35:40"
```

`z.mean(axis=0)` adalah rata-rata **per kolom** (lihat `axis` di [NumPy](numpy.md)); `.round(2)` membulatkan. Rata-rata `-0.` berarti nilai sangat kecil negatif yang dibulatkan, yaitu praktis nol.

```text
--8<-- "expected/fundamentals/preprocessing.txt:10:11"
```

**Hasil.** Setelah `StandardScaler`, kedua kolom berrata-rata 0 dan simpangan baku 1; setelah `MinMaxScaler`, minimum 0 dan maksimum 1. Pilih `StandardScaler` secara umum; `MinMaxScaler` bila model memerlukan rentang tertentu, tetapi ia sangat dipengaruhi *outlier*.

??? example "Program lengkap (`preprocessing.py`)"
    ```python title="preprocessing.py" linenums="1"
    --8<-- "fundamentals/preprocessing.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - *Fit* imputer atau *scaler* pada seluruh data sebelum split.
    - Mengisi data hilang dengan rata-rata pada data yang sangat miring.
    - *Label encoding* (1, 2, 3) pada kategori nominal untuk model berbasis jarak atau linear.
    - Menghapus *outlier* hanya karena "jauh", tanpa memahami penyebabnya.
    - Menggunakan *scaler* berbeda untuk data latih dan data uji.

## Latihan

1. Ubah strategi imputasi `income` menjadi `mean`. Apa perbedaan nilai pengisinya? Mengapa itu bermasalah di sini?
2. Terapkan `OneHotEncoder` pada kolom `plan`. Berapa kolom yang terbentuk?
3. Bandingkan nilai maksimum `income` sebelum dan sesudah *capping*.
4. Mengapa `StandardScaler` tidak diperlukan oleh Random Forest?

## Ringkasan

Praproses meliputi penghapusan duplikat, imputasi, penanganan *outlier*, *encoding*, dan *scaling*. Langkah yang belajar dari data harus di-*fit* hanya pada data latih; `Pipeline` memastikan hal ini.

Berikutnya: [Feature Engineering](feature-engineering.md).
