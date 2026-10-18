# Itemset Sering dan Apriori

## Tujuan pembelajaran

- [x] Menjelaskan itemset sering (*frequent itemset*) dan `min_support`.
- [x] Menjelaskan prinsip Apriori (sifat anti-monoton) dan mengapa ia memangkas pencarian.
- [x] Menjalankan Apriori dengan `mlxtend` dan memahami ledakan kombinatorial.

## Konsep

Itemset **sering** adalah itemset dengan *support* di atas ambang `min_support`. Mencari semuanya secara naif berarti memeriksa seluruh kombinasi barang, yang berjumlah \(2^{p}-1\) untuk \(p\) barang, sehingga tidak praktis.

**Sifat anti-monoton**: jika sebuah itemset sering, semua himpunan bagiannya juga sering. Sebaliknya, jika {A, B} tidak sering, maka {A, B, C} pasti tidak sering. Apriori memanfaatkannya:

1. Hitung *support* barang tunggal; buang yang di bawah `min_support`.
2. Bentuk kandidat pasangan **hanya** dari barang yang sering; hitung *support*; buang yang tidak lolos.
3. Bentuk kandidat ukuran \(k+1\) dari itemset ukuran \(k\) yang sering (dan pastikan semua himpunan bagiannya sering); ulangi sampai tidak ada kandidat baru.

| Aspek | Keterangan |
|---|---|
| Kelebihan | Sederhana, mudah dipahami, hasil lengkap untuk ambang yang diberikan |
| Keterbatasan | Memindai data berulang kali dan menghasilkan banyak kandidat bila `min_support` rendah |
| Parameter | `min_support`, `max_len` |

## Implementasi langkah demi langkah

Memakai 2000 transaksi sintetis ([penjelasan](transactions-encoding.md)) dan `mlxtend.frequent_patterns.apriori`. Program `examples/association/apriori.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: menyiapkan matriks transaksi

```python title="apriori.py" linenums="1"
--8<-- "association/apriori.py:1:10"
```

`make_transactions()` menghasilkan 2000 keranjang; `TransactionEncoder` mengubahnya menjadi tabel `True`/`False` berukuran 2000 × 20 ([lesson sebelumnya](transactions-encoding.md#langkah-1-contoh-kecil-dan-matriks-one-hot)). `df.shape[1]` adalah jumlah kolom = jumlah barang.

### Langkah 2: mencari itemset sering

**Tujuan.** Menemukan semua itemset yang muncul pada minimal 5% transaksi.

```python title="apriori.py" linenums="12"
--8<-- "association/apriori.py:12:15"
```

| Kode | Penjelasan |
|---|---|
| `apriori(df, min_support=0.05, use_colnames=True)` | Menjalankan Apriori. `min_support=0.05` berarti itemset harus ada di ≥ 5% transaksi (≥ 100 dari 2000). `use_colnames=True` menampilkan nama barang (bukan nomor kolom). Hasilnya tabel berkolom `support` dan `itemsets`. |
| `itemsets["itemsets"].apply(len)` | Jumlah barang tiap itemset; disimpan di kolom `size`. |
| `value_counts().sort_index().to_dict()` | Berapa itemset sering untuk tiap ukuran (1, 2, 3). |

```text
--8<-- "expected/association/apriori.txt:1:3"
```

**Hasil.** Ada **54 itemset sering**: 20 barang tunggal (seluruh barang masih lolos), 29 pasangan, dan 5 itemset tiga barang.

### Langkah 3: itemset dengan support tertinggi

```python title="apriori.py" linenums="17"
--8<-- "association/apriori.py:17:20"
```

Hanya itemset berukuran ≥ 2 diambil dan diurutkan menurut *support*; `" + ".join(sorted(s))` mengubah himpunan barang menjadi teks seperti `mentega + roti`; `.head(6)` mengambil enam teratas.

```text
--8<-- "expected/association/apriori.txt:5:11"
```

**Interpretasi.** Pasangan teratas: `mentega + roti` (0,220), `gula + kopi` (0,126), `air_mineral + kantong_plastik` (0,119), `kantong_plastik + roti` (0,118). Pasangan dengan `kantong_plastik` muncul sering **karena kantong itu sendiri populer** (44,8% keranjang), bukan karena ada hubungan khusus; *support* tinggi belum berarti asosiasi (dibahas di [lesson aturan](rules-metrics.md)).

### Langkah 4: sifat anti-monoton

**Konsep.** Support sebuah itemset tidak pernah melebihi support himpunan bagiannya. Sifat inilah yang memungkinkan Apriori memangkas pencarian.

```python title="apriori.py" linenums="22"
--8<-- "association/apriori.py:22:26"
```

`sup` adalah kamus `{itemset (frozenset): support}`; `frozenset` adalah himpunan yang tidak dapat diubah sehingga boleh dipakai sebagai kunci kamus. Baris cetak membandingkan support {roti}, {mentega}, dan {roti, mentega}.

```text
--8<-- "expected/association/apriori.txt:13:13"
```

**Hasil.** {roti} = 0,264 dan {mentega} = 0,246, sedangkan {roti, mentega} = 0,220: tidak pernah melebihi himpunan bagiannya.

### Langkah 5: ledakan kombinatorial dan `max_len`

```python title="apriori.py" linenums="28"
--8<-- "association/apriori.py:28:34"
```

Perulangan menjalankan Apriori dengan lima nilai `min_support` dan mencetak jumlah itemset yang dihasilkan. Baris terakhir memakai `max_len=2` untuk membatasi ukuran itemset maksimal dua barang.

```text
--8<-- "expected/association/apriori.txt:15:22"
```

**Interpretasi.** Jumlah itemset sering naik cepat saat `min_support` turun: 7 (0,2), 21 (0,1), 54 (0,05), 170 (0,02), dan 394 (0,01). Dengan 20 barang saja sudah terlihat; dengan ribuan barang jumlahnya dapat melampaui kemampuan memori. Membatasi itemset maksimal dua barang pada `min_support=0,01` memangkas hasil dari 394 menjadi 177.

??? example "Program lengkap (`apriori.py`)"
    ```python title="apriori.py" linenums="1"
    --8<-- "association/apriori.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai `min_support` sangat rendah pada banyak barang tanpa `max_len`.
    - Menyimpulkan bahwa itemset dengan *support* tinggi pasti punya asosiasi.
    - Lupa bahwa `min_support` harus sesuai ukuran data (5% dari 100 transaksi hanya 5 keranjang).
    - Menggunakan barang yang hampir ada di semua keranjang.

## Latihan

1. Naikkan `min_support` ke 0,15. Itemset apa saja yang tersisa?
2. Hitung secara manual jumlah kandidat pasangan dari 20 barang dan bandingkan dengan 29 yang lolos.
3. Hapus `kantong_plastik` dan `air_mineral` dari data. Bagaimana pasangan teratas berubah?

## Ringkasan

Apriori mencari itemset sering dengan memangkas kandidat berdasarkan sifat anti-monoton. Pilih `min_support` dan `max_len` dengan hati-hati karena jumlah itemset dapat meledak.

Berikutnya: [FP-Growth](fpgrowth.md).
