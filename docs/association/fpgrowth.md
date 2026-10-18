# FP-Growth

## Tujuan pembelajaran

- [x] Menjelaskan ide FP-Growth dan perbedaannya dari Apriori.
- [x] Memahami urutan barang berdasarkan frekuensi pada FP-tree.
- [x] Memastikan bahwa Apriori dan FP-Growth menghasilkan itemset yang sama.

## Konsep

FP-Growth (*Frequent Pattern Growth*) menghindari pembentukan kandidat dalam jumlah besar. Prosesnya:

1. Pindai data sekali untuk menghitung frekuensi tiap barang; buang yang di bawah `min_support`.
2. Urutkan barang pada tiap transaksi menurut frekuensi menurun.
3. Bangun **FP-tree**, yaitu struktur pohon yang menggabungkan awalan (*prefix*) transaksi yang sama sehingga data dimampatkan.
4. Gali itemset sering dengan menelusuri pohon secara rekursif, tanpa menghasilkan kandidat secara eksplisit.

| | Apriori | FP-Growth |
|---|---|---|
| Strategi | Bangkitkan kandidat, lalu uji | Mampatkan data ke pohon, lalu gali |
| Pemindaian data | Berkali-kali | Dua kali |
| Hasil | Itemset sering yang sama | Itemset sering yang sama |
| Kecenderungan | Lambat dan boros memori pada `min_support` rendah | Umumnya lebih cepat pada data besar dan padat |

Perbedaan keduanya hanya pada **cara** menemukan itemset, bukan pada **hasilnya**.

## Implementasi langkah demi langkah

Bagian 1 memperlihatkan langkah 1–2 FP-Growth pada contoh lima transaksi. Bagian 2 membandingkan hasil Apriori dan FP-Growth pada 2000 transaksi sintetis. Program `examples/association/fpgrowth.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: urutan barang menurut frekuensi (persiapan FP-tree)

**Tujuan.** Memahami langkah awal FP-Growth: menghitung frekuensi barang, membuang yang di bawah ambang, dan mengurutkan barang tiap transaksi menurut frekuensi menurun.

```python title="fpgrowth.py" linenums="7"
--8<-- "association/fpgrowth.py:7:15"
```

| Kode | Penjelasan |
|---|---|
| `pd.Series([i for t in tiny for i in t]).value_counts()` | Meratakan semua transaksi menjadi satu daftar barang lalu menghitung frekuensi tiap barang. |
| `counts[counts >= 2].index.tolist()` | Barang dengan frekuensi ≥ 2 (ambang minimal 2 dari 5 transaksi), berurutan menurun; ini **urutan header**. |
| `sorted(t, key=order.index)` | Mengurutkan barang dalam satu transaksi menurut posisinya di `order` (frekuensi tertinggi dulu). |

```text
--8<-- "expected/association/fpgrowth.txt:1:7"
```

**Interpretasi.** Frekuensi `roti` 4, `mentega` 3, `susu` 2, `telur` 2; dengan ambang minimal 2 transaksi semua barang lolos, dan urutan header menurun adalah roti, mentega, susu, telur. Tiap transaksi diurutkan mengikuti urutan itu (misalnya `roti, mentega, telur`). Transaksi yang berawalan sama (`roti, mentega`) akan berbagi cabang pada FP-tree, itulah sebabnya data termampatkan.

### Langkah 2: Apriori dan FP-Growth menghasilkan itemset yang sama

```python title="fpgrowth.py" linenums="17"
--8<-- "association/fpgrowth.py:17:27"
```

| Kode | Penjelasan |
|---|---|
| `fpgrowth(df, min_support=ms, use_colnames=True)` | Antarmuka sama dengan `apriori`; hanya algoritmanya yang berbeda. |
| `{frozenset(x): round(s, 10) for x, s in zip(...)}` | Mengubah hasil menjadi kamus `{itemset: support}` (support dibulatkan agar perbandingan tidak terganggu pembulatan komputer). |
| `sa == sf` | Membandingkan dua kamus sebagai himpunan pasangan: `True` bila isinya persis sama. |

```text
--8<-- "expected/association/fpgrowth.txt:9:11"
```

**Hasil.** Pada `min_support` 0,05 dan 0,01, Apriori dan FP-Growth sama-sama menghasilkan 54 dan 394 itemset dengan *support* yang sama persis (`identik: True`).

### Langkah 3: urutan baris tidak dijamin sama

```python title="fpgrowth.py" linenums="29"
--8<-- "association/fpgrowth.py:29:30"
```

Baris ini membandingkan **urutan** itemset pada kedua hasil (bukan isinya).

```text
--8<-- "expected/association/fpgrowth.txt:13:13"
```

**Hasil.** `False`: kedua fungsi tidak menjamin urutan baris yang sama, sehingga bandingkan sebagai himpunan (seperti di Langkah 2), bukan sebagai urutan.

!!! note "Soal kecepatan"
    FP-Growth biasanya lebih cepat daripada Apriori ketika `min_support` rendah dan data besar. Materi ini **tidak** mengukur waktu (waktu bergantung perangkat dan data); untuk memeriksanya pada data Anda, ukur dengan `time.perf_counter()` untuk kedua fungsi pada `min_support` yang sama.

??? example "Program lengkap (`fpgrowth.py`)"
    ```python title="fpgrowth.py" linenums="1"
    --8<-- "association/fpgrowth.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Mengira FP-Growth menghasilkan aturan yang lebih baik; hasilnya sama, hanya caranya berbeda.
    - Membandingkan hasil dengan urutan baris.
    - Mengabaikan batas memori pada `min_support` sangat rendah, bahkan dengan FP-Growth.
    - Mengukur waktu dengan sekali jalan pada data kecil.

## Latihan

1. Ukur waktu `apriori` dan `fpgrowth` pada `min_support` 0,05; 0,02; 0,01 menggunakan `time.perf_counter()`.
2. Gambar FP-tree untuk lima transaksi kecil dengan tangan.
3. Bangkitkan 20.000 transaksi (`make_transactions(n=20000)`) dan ulangi pengukuran waktu.

## Ringkasan

FP-Growth menemukan itemset sering yang sama dengan Apriori memakai FP-tree sehingga menghindari kandidat dalam jumlah besar. Pilih berdasarkan skala data, dan jangan membandingkan hasil dengan urutan baris.

Berikutnya: [Aturan Asosiasi: Support, Confidence, Lift](rules-metrics.md).
