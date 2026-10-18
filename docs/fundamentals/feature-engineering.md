# Feature Engineering

## Tujuan pembelajaran

- [x] Membuat fitur baru dari pengetahuan domain.
- [x] Memakai transformasi log, *binning*, dan fitur polinomial.
- [x] Menguji secara jujur apakah fitur baru benar-benar membantu.

## Konsep dasar

*Feature engineering* adalah membuat atau mengubah fitur agar pola lebih mudah dipelajari model. Sumber ide terbaik adalah pengetahuan domain: rasio, selisih, hitungan, atau kategori yang bermakna. Namun fitur baru **tidak otomatis lebih baik**; ia harus dibuktikan lewat validasi.

| Teknik | Contoh | Manfaat |
|---|---|---|
| Fitur turunan (rasio, total) | `n_orders / tenure_months` | Menangkap intensitas, bukan hanya jumlah |
| Transformasi log | `log1p(income)` | Mengurangi kemiringan ekstrem |
| *Binning* | usia → kelompok usia | Menangkap pola tidak linear sederhana |
| Fitur polinomial/interaksi | \(x_1^2, x_1 x_2\) | Memungkinkan model linear menangkap kelengkungan |
| *Encoding* kategorik | *one-hot* | Mengubah kategori menjadi angka |

## Implementasi langkah demi langkah

Memakai data pelanggan sintetis ([penjelasan](pandas.md#dataset-yang-dipakai)). Bagian akhir membandingkan ROC-AUC (5-fold CV) dengan dan tanpa fitur turunan. Program lengkap ada di akhir bagian ini.

### Langkah 1: fitur turunan dari pengetahuan domain

**Tujuan.** Membuat fitur yang menyatakan konsep bermakna dari kolom yang sudah ada.

```python title="feature_engineering.py" linenums="1"
--8<-- "fundamentals/feature_engineering.py:1:15"
```

| Kode | Penjelasan |
|---|---|
| `make_customer_data().drop_duplicates().reset_index(drop=True)` | Membuat data lalu langsung membuang duplikat (rangkaian pemanggilan metode, dibaca dari kiri ke kanan). |
| `df["spend_total"] = n_orders * avg_spend` | Total belanja = jumlah transaksi × rata-rata belanja. |
| `df["orders_per_month"] = n_orders / tenure_months` | **Rasio**: intensitas transaksi per bulan; pelanggan baru dan lama menjadi sebanding. |
| `(tenure_months <= 6).astype(int)` | Syarat boolean menghasilkan `True`/`False`; `astype(int)` mengubahnya menjadi 1/0, yaitu fitur biner "pelanggan baru". |

Fitur turunan di atas hanya memakai kolom pada baris yang sama sehingga aman dari kebocoran.

### Langkah 2: transformasi log untuk data miring

**Konsep.** Pada data sangat miring ke kanan, logaritma menarik nilai besar mendekat sehingga distribusi lebih simetris. `np.log1p(x)` menghitung \(\ln(1+x)\): aman bila ada nilai 0.

```python title="feature_engineering.py" linenums="17"
--8<-- "fundamentals/feature_engineering.py:17:20"
```

```text
--8<-- "expected/fundamentals/feature_engineering.txt:1:2"
```

**Interpretasi.** *Skewness* `income` turun dari 13,73 menjadi 1,04. Masih di atas 0 karena beberapa *outlier* ekstrem tetap ada. Nilai `NaN` tetap `NaN` setelah log (imputasi dilakukan terpisah).

### Langkah 3: *binning* (mengelompokkan nilai numerik)

**Konsep.** `pd.cut` membagi angka menjadi kelompok menurut batas yang kita tentukan. Berguna bila hubungan dengan target tidak linear atau agar mudah dijelaskan.

```python title="feature_engineering.py" linenums="22"
--8<-- "fundamentals/feature_engineering.py:22:25"
```

`bins=[17, 30, 45, 60, 100]` mendefinisikan interval (17,30], (30,45], (45,60], (60,100]; `labels` memberi nama tiap kelompok. `.agg(["size", "mean"])` menghitung jumlah dan rata-rata churn tiap kelompok; `observed=True` hanya menampilkan kelompok yang benar-benar ada.

```text
--8<-- "expected/fundamentals/feature_engineering.txt:3:8"
```

**Interpretasi.** Tingkat churn: 35,5% (18–30), 31,4% (31–45), 40,7% (46–60), 23,1% (61+). Polanya tidak monoton, jadi pengelompokan membantu melihatnya. Perbedaan antar kelompok tidak besar dan kelompok berukuran 104–172 sampel, jadi jangan ditafsirkan berlebihan.

### Langkah 4: fitur polinomial dan interaksi

**Konsep.** `PolynomialFeatures(degree=2)` menambahkan kuadrat tiap fitur dan hasil kali antar-fitur (interaksi), sehingga model linear dapat menangkap kelengkungan dan pengaruh gabungan.

```python title="feature_engineering.py" linenums="27"
--8<-- "fundamentals/feature_engineering.py:27:31"
```

`include_bias=False` tidak menambah kolom konstan 1. `poly.fit(X)` mempelajari jumlah dan nama fitur; `get_feature_names_out()` menamai kolom keluaran.

```text
--8<-- "expected/fundamentals/feature_engineering.txt:9:10"
```

**Hasil.** Dua fitur asli menjadi lima kolom: dua fitur asli, kuadratnya, dan hasil kali `tenure_months n_orders` (interaksi). Jumlah kolom tumbuh cepat pada derajat dan fitur yang lebih banyak.

### Langkah 5: apakah fitur baru benar-benar membantu?

**Tujuan.** Fitur baru **harus dibuktikan** dengan validasi, bukan diasumsikan lebih baik.

```python title="feature_engineering.py" linenums="33"
--8<-- "fundamentals/feature_engineering.py:33:41"
```

| Kode | Penjelasan |
|---|---|
| `def cv_auc(columns): ...` | Fungsi yang membangun `Pipeline` (scaler + regresi logistik), lalu mengembalikan rata-rata ROC-AUC 5-fold untuk kolom fitur tertentu. Pemakaian fungsi menghindari menulis ulang kode yang sama dua kali. |
| `make_pipeline(StandardScaler(), LogisticRegression(...))` | Standardisasi dilakukan di dalam validasi silang sehingga tidak bocor. |
| `cross_val_score(..., cv=5, scoring="roc_auc")` | Melatih dan menguji 5 kali pada pembagian berbeda; hasilnya lima skor. `.mean()` merata-ratakannya. |
| `base + ["orders_per_month", "is_new"]` | Menggabungkan dua *list* untuk membentuk daftar fitur yang diperluas. |

```text
--8<-- "expected/fundamentals/feature_engineering.txt:11:12"
```

**Interpretasi.** ROC-AUC 0,703 (fitur asli) vs 0,702 (ditambah `orders_per_month` dan `is_new`): **tidak ada peningkatan**. Pada data sintetis ini fitur asli sudah memuat informasi yang dipakai pembangkit data, sehingga fitur turunan tidak menambah apa-apa. Menambah fitur tanpa bukti hanya menambah kompleksitas.

??? example "Program lengkap (`feature_engineering.py`)"
    ```python title="feature_engineering.py" linenums="1"
    --8<-- "fundamentals/feature_engineering.py"
    ```

## Prinsip penting

- Fitur dibuat dari informasi yang **tersedia saat prediksi**. Fitur yang memakai informasi masa depan atau target menyebabkan kebocoran data.
- Transformasi yang belajar dari data (misalnya *scaling*, *binning* berbasis kuantil) harus di-*fit* pada data latih, di dalam `Pipeline`.
- Banyak model (Random Forest, *gradient boosting*) dapat menangkap non-linearitas sendiri; rekayasa fitur paling penting untuk model linear dan berbasis jarak.

## Kesalahan umum

!!! warning "Hindari"
    - Membuat fitur dari kolom target atau informasi yang baru diketahui setelah kejadian.
    - Menghitung statistik kelompok (misalnya rata-rata per kota) pada seluruh data sebelum split.
    - Menambah banyak fitur tanpa validasi sehingga risiko *overfitting* naik.
    - Memakai *log* pada nilai nol atau negatif tanpa `log1p`/penyesuaian.

## Latihan

1. Buat fitur `spend_per_order` dan uji pengaruhnya pada ROC-AUC CV.
2. Coba `PolynomialFeatures(degree=3)`; berapa kolom yang terbentuk dari dua fitur?
3. Tambahkan `plan` (dengan *one-hot*) ke model. Apakah ROC-AUC naik?
4. Berikan satu contoh fitur yang menyebabkan kebocoran pada kasus churn.

## Ringkasan

Fitur baru berguna bila berlandaskan domain dan terbukti memperbaiki validasi. Buat dari informasi yang tersedia saat prediksi, dan lakukan transformasi terlatih di dalam `Pipeline`.

Berikutnya: [Data Latih, Validasi, dan Uji](data-splitting.md).
