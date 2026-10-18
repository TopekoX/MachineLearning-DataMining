# Exploratory Data Analysis (EDA)

## Tujuan pembelajaran

- [x] Menjalankan alur EDA sistematis pada dataset baru.
- [x] Mengenali masalah kualitas data: nilai hilang, duplikat, *outlier*.
- [x] Memeriksa distribusi target dan hubungan fitur dengan target.
- [x] Menerjemahkan temuan EDA menjadi keputusan praproses.

## Konsep dasar

EDA adalah tahap mengenal data **sebelum** memodelkan. Tujuannya bukan membuat model, tetapi menjawab: Data apa ini? Seberapa bersih? Bagaimana targetnya? Fitur mana yang tampak informatif? Keputusan praproses dan pemilihan metrik bergantung pada jawabannya.

## Daftar periksa EDA

1. **Ukuran dan tipe**: jumlah baris/kolom, tipe tiap kolom sesuai maknanya?
2. **Data hilang**: kolom mana, berapa persen?
3. **Duplikat**.
4. **Ringkasan statistik** numerik dan kategorik: rentang masuk akal?
5. **Distribusi target**: seimbang atau tidak? (menentukan metrik dan strategi split)
6. **Hubungan fitur–target** dan antar fitur.
7. **Catat keputusan** untuk tahap praproses.

!!! tip "Hati-hati dengan data uji"
    Lakukan EDA yang memengaruhi keputusan modelling pada **data latih**. Jika Anda mengintip data uji lalu memilih fitur atau transformasi berdasarkan itu, informasi uji ikut "bocor" ke model.

## Implementasi langkah demi langkah

Memakai data pelanggan sintetis ([penjelasan kolom](pandas.md#dataset-yang-dipakai)). Demi ringkas, contoh ini menganalisis seluruh data; pada proyek nyata, lakukan setelah memisahkan data uji. Program lengkap ada di akhir bagian ini.

### Langkah 1: ukuran dan tipe data

**Tujuan.** Memastikan data terbaca benar dan tiap kolom bertipe sesuai maknanya (angka sebagai angka, teks sebagai teks).

```python title="eda.py" linenums="1"
--8<-- "fundamentals/eda.py:1:7"
```

`df.shape` memberi `(baris, kolom)`; `df.dtypes` memberi tipe tiap kolom, dan `.to_string()` mencetaknya penuh tanpa dipotong.

```text
--8<-- "expected/fundamentals/eda.txt:1:9"
```

**Interpretasi.** 605 baris dan 8 kolom. `age`, `tenure_months`, `n_orders`, dan `churn` bertipe `int64`; `income` dan `avg_spend` `float64`; `city` dan `plan` teks (`str`). Tipe sudah wajar. Waspadai kolom seperti ID atau kode kota yang tersimpan sebagai angka padahal sebenarnya kategori.

### Langkah 2: data hilang

```python title="eda.py" linenums="9"
--8<-- "fundamentals/eda.py:9:13"
```

| Kode | Penjelasan |
|---|---|
| `df.isna().sum()` | Jumlah sel kosong per kolom. |
| `missing[missing > 0]` | Menyaring: hanya kolom yang punya nilai hilang. |
| `.to_frame("n_missing")` | Mengubah `Series` hasil saringan menjadi `DataFrame` satu kolom bernama `n_missing`. |
| `summary["pct"] = ... / len(df) * 100` | Kolom baru: persentase hilang terhadap jumlah baris. |

```text
--8<-- "expected/fundamentals/eda.txt:10:12"
```

**Interpretasi.** `city` hilang 18 baris (3,0%) dan `income` 48 baris (7,9%). Persentase membantu menentukan penanganan: sedikit hilang dapat diimputasi; jika hilang sangat banyak, pertimbangkan membuang kolom.

### Langkah 3: duplikat dan ringkasan statistik

```python title="eda.py" linenums="15"
--8<-- "fundamentals/eda.py:15:20"
```

`df.duplicated().sum()` menghitung baris yang persis sama dengan baris sebelumnya. `df[numeric].describe()` meringkas kolom numerik (jumlah, rata-rata, simpangan baku, min, kuartil, maks); `.T` memutar tabel (*transpose*) agar tiap kolom data menjadi satu baris, lalu `[["count", "mean", "50%", "max"]]` memilih empat statistik. `50%` adalah median.

```text
--8<-- "expected/fundamentals/eda.txt:13:19"
```

**Interpretasi.** Ada 5 baris duplikat. Pada `income`, `count` 557 (bukan 605) mengonfirmasi nilai hilang; rata-rata 7.445.776 jauh di atas median 6.050.000 dan maksimum 229.140.000 sangat jauh dari keduanya: tanda *outlier* dan kemiringan ke kanan. `n_orders` berkisar sampai 23 dengan median 8, wajar.

### Langkah 4: distribusi target

```python title="eda.py" linenums="22"
--8<-- "fundamentals/eda.py:22:23"
```

`value_counts(normalize=True)` menghitung **proporsi** tiap nilai (bukan jumlah).

```text
--8<-- "expected/fundamentals/eda.txt:20:23"
```

**Interpretasi.** 66,1% pelanggan bertahan (0) dan 33,9% churn (1): kelas tidak terlalu timpang, tetapi cukup tidak seimbang sehingga *stratified split* dan metrik selain akurasi tetap dianjurkan. Proporsi ini juga menjadi **baseline**: model yang menebak "bertahan" untuk semua orang sudah akurat 66,1%.

### Langkah 5: hubungan fitur dengan target

```python title="eda.py" linenums="25"
--8<-- "fundamentals/eda.py:25:28"
```

| Kode | Penjelasan |
|---|---|
| `df.groupby("plan")["churn"].mean()` | Tingkat churn per paket. |
| `df["tenure_months"] // 12 * 12` | `//` pembagian bulat: mengelompokkan lama berlangganan per 12 bulan (0–11 → 0, 12–23 → 12, dst.). Hasilnya dipakai sebagai kunci `groupby`. |
| `.rename("churn per 12 bulan")` | Memberi nama pada hasil agar keluarannya terbaca. |
| `df[[...]].corr()["churn"]` | Korelasi tiap kolom numerik dengan `churn`. |

```text
--8<-- "expected/fundamentals/eda.txt:24:41"
```

**Interpretasi.** Paket `basic` churn 39,1% vs `premium` 21,9%. Churn turun dari 55,9% (lama berlangganan 0–11 bulan) menjadi 14,9% (48–59 bulan); kelompok `60` hanya berisi pelanggan dengan tepat 60 bulan (sampel kecil), sehingga angka 21,4% pada baris itu tidak dapat diandalkan. Korelasi dengan churn: `tenure_months` -0,30 dan `n_orders` -0,17 (makin lama dan makin sering bertransaksi, makin kecil kemungkinan churn); `age` -0,06 dan `avg_spend` -0,03 hampir nol. Korelasi bukan sebab-akibat.

??? example "Program lengkap (`eda.py`)"
    ```python title="eda.py" linenums="1"
    --8<-- "fundamentals/eda.py"
    ```

## Interpretasi dan keputusan

| Temuan | Keputusan praproses |
|---|---|
| `city` hilang 3,0%, `income` hilang 7,9% | Imputasi: median untuk `income`, modus untuk `city` |
| 5 baris duplikat | Hapus duplikat |
| Rata-rata `income` (7,45 juta) jauh di atas median (6,05 juta); maksimum 229 juta | Selidiki *outlier*; pertimbangkan *capping* atau transformasi log |
| `churn`: 66,1% kelas 0 vs 33,9% kelas 1 | Tidak terlalu timpang; tetap pakai *stratified split* dan jangan hanya melihat *accuracy* |
| Churn turun dari 55,9% (lama berlangganan 0–11 bulan) ke 14,9% (48–59 bulan) | `tenure_months` kemungkinan prediktor penting |
| Korelasi dengan churn: `tenure_months` -0,30, `n_orders` -0,17, `age` -0,06, `avg_spend` -0,03 | Dua fitur pertama tampak lebih informatif secara linear; korelasi ≠ kausal |

Catatan: kelompok `tenure_months` = 60 hanya berisi pelanggan dengan tepat 60 bulan (sampel kecil), sehingga angka 21,4% pada baris itu tidak dapat diandalkan. Selalu cek ukuran kelompok.

## Kesalahan umum

!!! warning "Hindari"
    - Melompat ke pemodelan tanpa memeriksa kualitas data.
    - Menafsirkan tipe `int` sebagai numerik padahal sebenarnya kode kategori (mis. ID kota).
    - Memeriksa distribusi target setelah *resampling*, bukan sebelumnya.
    - Memilih fitur berdasarkan seluruh data termasuk data uji.
    - Menyimpulkan dari kelompok yang sangat kecil.

## Latihan

1. Tambahkan pengecekan nilai unik pada kolom kategorik `city` dan `plan`.
2. Hitung persentase churn per `city`. Apakah ada kota yang menonjol?
3. Buat daftar keputusan praproses untuk dataset `load_wine` dari hasil `describe()`.
4. Mengapa EDA yang memengaruhi keputusan sebaiknya memakai data latih saja?

## Ringkasan

EDA mengenali ukuran, kualitas, target, dan hubungan data. Hasilnya adalah daftar keputusan praproses dan pemilihan metrik. Lakukan dengan disiplin agar data uji tidak ikut memengaruhi keputusan.

Berikutnya: [Pembersihan dan Praproses Data](data-preprocessing.md).
