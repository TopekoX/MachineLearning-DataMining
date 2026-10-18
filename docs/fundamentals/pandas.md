# pandas

## Tujuan pembelajaran

- [x] Memahami `DataFrame` dan `Series`.
- [x] Memilih kolom dan baris, termasuk dengan syarat.
- [x] Meringkas data per kelompok dengan `groupby`.
- [x] Mendeteksi data hilang dan duplikat, serta membuat kolom baru.

## Konsep dasar

pandas mengelola data tabular. `DataFrame` adalah tabel (baris = sampel, kolom = variabel); satu kolom adalah `Series`. Hampir semua pekerjaan data mining diawali dengan memuat tabel ke `DataFrame`, lalu memeriksa dan merapikannya.

## Dataset yang dipakai

Banyak lesson di bab Fundamentals memakai **data pelanggan sintetis** (dibangkitkan program dengan seed tetap, bukan data nyata). Tiap baris adalah pelanggan sebuah layanan berlangganan. Targetnya `churn` (1 = berhenti berlangganan).

| Kolom | Arti | Tipe |
|---|---|---|
| `age` | usia (tahun) | numerik |
| `city` | kota | kategorik |
| `plan` | paket (`basic`/`premium`) | kategorik |
| `tenure_months` | lama berlangganan (bulan) | numerik |
| `income` | pendapatan (Rp/bulan) | numerik |
| `n_orders` | jumlah transaksi | numerik (cacah) |
| `avg_spend` | rata-rata belanja per transaksi (Rp) | numerik |
| `churn` | target: 1 = berhenti | biner |

Data sengaja dibuat "kotor": ada nilai hilang, *outlier*, dan baris duplikat, agar bisa dilatih menanganinya. Karena sintetis, hubungan di dalamnya hanya benar untuk data ini dan tidak boleh dianggap temuan tentang pelanggan nyata.

??? example "Kode pembangkit data (`customer_data.py`)"
    ```python title="customer_data.py"
    --8<-- "fundamentals/customer_data.py"
    ```

## Implementasi langkah demi langkah

Simpan `customer_data.py` dan `pandas_basics.py` dalam folder yang sama. Program dipecah menjadi enam langkah; program lengkap ada di akhir bagian ini.

### Langkah 1: memuat dan mengenal data

**Tujuan.** Sebelum melakukan apa pun, lihat dulu ukuran, isi, dan tipe data.

```python title="pandas_basics.py" linenums="1"
--8<-- "fundamentals/pandas_basics.py:1:9"
```

| Kode | Penjelasan |
|---|---|
| `from customer_data import make_customer_data` | Mengimpor fungsi pembangkit data sintetis dari berkas `customer_data.py` di folder yang sama. |
| `df = make_customer_data()` | Menghasilkan sebuah `DataFrame` dan menyimpannya di variabel `df` (nama umum untuk DataFrame). Pada data nyata, baris ini diganti `pd.read_csv("data.csv")`, yang membaca berkas CSV menjadi DataFrame. |
| `df.shape` | Tuple `(jumlah_baris, jumlah_kolom)`; tanpa tanda kurung karena ia atribut, bukan fungsi. |
| `df.head(3)` | Menampilkan 3 baris pertama untuk memastikan data terbaca benar. |
| `df.info()` | Ringkasan: jumlah baris, nama kolom, **jumlah nilai terisi** (*non-null*) dan tipe tiap kolom. |

```text
--8<-- "expected/fundamentals/pandas_basics.txt:1:20"
```

**Interpretasi.** Data memiliki 605 baris dan 8 kolom. Pada `info()`, `city` memiliki 587 nilai terisi dan `income` 557, sedangkan kolom lain 605: berarti **ada data hilang** pada dua kolom itu. `NaN` pada baris pertama kolom `income` adalah penanda nilai hilang. Tipe `int64` dan `float64` adalah angka; `str` adalah teks.

!!! note "Catatan versi"
    Pada pandas 3, kolom teks bertipe `str`; pada pandas lama biasanya tampil sebagai `object`. Tampilan `df.info()` dapat berbeda antar versi.

### Langkah 2: Series versus DataFrame

**Konsep.** `DataFrame` adalah tabel (baris = sampel, kolom = variabel). Satu kolom dari tabel itu disebut `Series`: daftar nilai berindeks dengan satu dimensi. Perbedaan kecil dalam sintaks menghasilkan tipe berbeda, dan banyak fungsi (termasuk scikit-learn) membedakannya.

```python title="pandas_basics.py" linenums="11"
--8<-- "fundamentals/pandas_basics.py:11:14"
```

```text
--8<-- "expected/fundamentals/pandas_basics.txt:21:21"
```

`df["age"]` (satu kurung siku) menghasilkan `Series` berbentuk `(605,)`; `df[["age"]]` (dua kurung siku, yaitu *list* berisi nama kolom) menghasilkan `DataFrame` berbentuk `(605, 1)`. Ingat ini nanti: **fitur `X` harus 2 dimensi** (DataFrame), sedangkan **target `y` boleh 1 dimensi** (Series).

### Langkah 3: memilih baris dengan syarat, `loc`, dan `iloc`

```python title="pandas_basics.py" linenums="16"
--8<-- "fundamentals/pandas_basics.py:16:21"
```

| Kode | Penjelasan |
|---|---|
| `df["plan"].value_counts()` | Menghitung berapa kali tiap nilai muncul pada kolom (frekuensi). |
| `(df["plan"] == "premium") & (df["age"] >= 50)` | Dua syarat menghasilkan deret `True`/`False` per baris; `&` berarti "dan". **Tiap syarat wajib dalam kurung.** Gunakan `\|` untuk "atau". |
| `df[syarat]` | Menyaring: hanya baris bernilai `True` yang diambil. `len()` menghitung jumlah barisnya. |
| `df.loc[0:2, ["age", "city"]]` | Memilih menurut **label** baris dan nama kolom. Irisan label **termasuk** ujung akhirnya (0, 1, **2**). |
| `df.iloc[0:2, [0, 1]]` | Memilih menurut **posisi** (nomor urut mulai 0). Irisan posisi **tidak** termasuk ujung akhirnya (0, 1). |

```text
--8<-- "expected/fundamentals/pandas_basics.txt:22:33"
```

**Interpretasi.** Terdapat 422 pelanggan `basic` dan 183 `premium`; 70 pelanggan `premium` berusia ≥ 50. `loc[0:2]` mengembalikan **tiga** baris (0, 1, 2) sedangkan `iloc[0:2]` hanya **dua** (0, 1): kebiasaan `loc` vs `iloc` ini sering menjadi sumber kesalahan satu baris.

### Langkah 4: agregasi per kelompok dengan `groupby`

**Konsep.** `groupby` membagi data menjadi kelompok menurut nilai sebuah kolom, lalu menghitung ringkasan untuk tiap kelompok ("pecah, hitung, gabungkan").

```python title="pandas_basics.py" linenums="23"
--8<-- "fundamentals/pandas_basics.py:23:25"
```

`df.groupby("plan")["churn"].mean()` membagi menurut paket, mengambil kolom `churn`, dan merata-ratakan. Karena `churn` bernilai 0/1, rata-ratanya adalah **proporsi churn**. Pada baris kedua, `.agg(n=("churn", "size"), churn_rate=("churn", "mean"))` menghitung dua ringkasan sekaligus dan memberi nama kolom hasilnya; `.round(3)` membulatkan ke tiga desimal.

```text
--8<-- "expected/fundamentals/pandas_basics.txt:34:43"
```

**Interpretasi.** Tingkat churn paket `basic` 39,1% dan `premium` 21,9%. Per kota berkisar 30,4% (Medan) sampai 36,9% (Surabaya). Perhatikan juga jumlah per kelompok (`n`): rata-rata dari kelompok kecil kurang dapat diandalkan. Perbedaan yang tampak ini hanyalah gambaran; menyimpulkan sebab-akibat memerlukan analisis lain.

### Langkah 5: data hilang dan duplikat

```python title="pandas_basics.py" linenums="27"
--8<-- "fundamentals/pandas_basics.py:27:29"
```

`df.isna()` menghasilkan tabel `True`/`False` (apakah sel kosong); `.sum()` menjumlahkan `True` per kolom (`True` dihitung 1). `df.duplicated()` menandai baris yang identik dengan baris sebelumnya, dan `.sum()` menghitungnya.

```text
--8<-- "expected/fundamentals/pandas_basics.txt:44:53"
```

**Interpretasi.** `city` kehilangan 18 nilai dan `income` 48; ada 5 baris duplikat. Cara menanganinya dibahas di [Praproses Data](data-preprocessing.md); langkah pertama selalu **mendeteksi** masalahnya.

### Langkah 6: kolom baru dan pengurutan

```python title="pandas_basics.py" linenums="31"
--8<-- "fundamentals/pandas_basics.py:31:33"
```

`df["spend_total"] = ...` membuat kolom baru dari perkalian dua kolom, dihitung per baris. `sort_values("spend_total", ascending=False)` mengurutkan menurun, `[[...]]` memilih tiga kolom, dan `.head(3)` mengambil tiga teratas.

```text
--8<-- "expected/fundamentals/pandas_basics.txt:54:57"
```

Angka paling kiri (284, 75, 5) adalah **indeks** baris asli; indeks ikut terbawa saat diurutkan. Pelanggan pada indeks 284 memiliki 11 transaksi dengan rata-rata 902.000 sehingga total 9.922.000.

??? example "Program lengkap (`pandas_basics.py`)"
    ```python title="pandas_basics.py" linenums="1"
    --8<-- "fundamentals/pandas_basics.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai `and`/`or` pada syarat; gunakan `&`/`|` dengan tanda kurung: `(a > 1) & (b < 2)`.
    - Mengubah potongan `DataFrame` lalu heran data asli tidak berubah (atau sebaliknya); gunakan `.copy()` dan `.loc` secara sadar.
    - Mengabaikan `isna()` dan `duplicated()` sebelum memodelkan.
    - Menganggap rata-rata `groupby` pada kelompok kecil sebagai fakta; periksa jumlah sampel tiap kelompok.

## Latihan

1. Hitung rata-rata `avg_spend` per `city`.
2. Tampilkan pelanggan `premium` dengan `tenure_months` ≥ 24 dan `churn` = 1.
3. Hitung persentase data hilang tiap kolom.
4. Buat kolom `orders_per_month = n_orders / tenure_months`, lalu urutkan dan lihat 5 teratas.

??? success "Petunjuk"
    1. `df.groupby("city")["avg_spend"].mean()`
    3. `df.isna().mean() * 100`

## Ringkasan

`DataFrame` adalah struktur utama data tabular. Alur dasarnya: periksa (`shape`, `info`), pilih (`loc`, syarat), ringkas (`groupby`), dan periksa kualitas (`isna`, `duplicated`).

Berikutnya: [Visualisasi Data](data-visualization.md).
