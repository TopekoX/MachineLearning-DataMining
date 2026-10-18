# Penyaringan Aturan, Ambang Batas, dan Keterbatasan

## Tujuan pembelajaran

- [x] Memilih `min_support` dan `min_confidence` serta memahami dampaknya pada jumlah aturan.
- [x] Menyaring aturan dengan lift, confidence, dan ukuran antecedent.
- [x] Menilai kestabilan aturan tanpa pembagian latih/uji klasik.
- [x] Mengenali keterbatasan aturan asosiasi.

## Memilih ambang

- **`min_support`** menentukan pola yang cukup sering untuk berguna dan sekaligus mengendalikan ledakan jumlah itemset. Nilainya bergantung pada ukuran data: terlalu tinggi menghilangkan pola penting tetapi langka; terlalu rendah menghasilkan banyak pola kebetulan dan beban komputasi besar.
- **`min_confidence`** menyaring kekuatan aturan, tetapi jangan dipakai sendirian.
- **Lift** menyaring aturan yang sekadar mengikuti popularitas barang (misalnya lift ≥ 1,2).
- **Ukuran antecedent**: aturan dengan banyak barang di sisi kiri sulit dipakai dan sering redundan.
- Pilihan akhir sebaiknya mempertimbangkan **kegunaan bisnis** (berapa aturan yang dapat ditinjau manusia) dan **biaya komputasi**.

## Mengapa tidak ada split latih/uji klasik?

Tidak ada label yang diprediksi, sehingga tidak ada galat prediksi untuk dihitung pada data uji. Alternatifnya adalah **uji kestabilan**: tambang aturan dari satu bagian transaksi (misalnya paruh pertama menurut waktu) lalu periksa apakah aturan itu masih berlaku, misalnya lift-nya, pada bagian lain. Aturan yang hanya muncul di satu bagian kemungkinan kebetulan atau musiman.

## Implementasi langkah demi langkah

Data: 2000 transaksi sintetis ([penjelasan](transactions-encoding.md)). Program `examples/association/filtering_thresholds.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: fungsi `mine` dan pengaruh ambang

**Tujuan.** Membungkus "itemset sering → aturan" dalam satu fungsi agar dapat diulang dengan ambang berbeda.

```python title="filtering_thresholds.py" linenums="7"
--8<-- "association/filtering_thresholds.py:7:23"
```

Fungsi `mine(data, min_support, min_confidence)` menjalankan FP-Growth lalu `association_rules`; bila tidak ada itemset (`len(its) == 0`) ia mengembalikan tabel kosong. Perulangan menghitung jumlah aturan untuk tiap kombinasi `min_support` (baris) dan `min_confidence` (kolom): `counts` adalah tiga angka (satu per `min_confidence`).

```text
--8<-- "expected/association/filtering_thresholds.txt:1:4"
```

**Interpretasi.** Dengan `min_confidence=0,3`, jumlah aturan 568 (`min_support=0,01`), 182 (0,02), dan 45 (0,05). Menurunkan `min_support` menaikkan jumlah aturan dengan cepat; pada 0,01 sudah terlalu banyak untuk ditinjau manual.

### Langkah 2: penyaringan bertahap

```python title="filtering_thresholds.py" linenums="25"
--8<-- "association/filtering_thresholds.py:25:31"
```

| Kode | Penjelasan |
|---|---|
| `rules[rules["lift"] >= 1.2]` | Hanya aturan dengan lift ≥ 1,2 (membuang yang sekadar mengikuti popularitas). |
| `step1[step1["confidence"] >= 0.5]` | Dari sisanya, hanya confidence ≥ 0,5. |
| `step2["antecedents"].apply(len) == 1` | Panjang himpunan antecedent sama dengan 1; digabung dengan `&` untuk consequent: aturan satu barang di tiap sisi. |

```text
--8<-- "expected/association/filtering_thresholds.txt:6:7"
```

**Hasil.** 182 aturan awal → 131 setelah lift ≥ 1,2 → 91 setelah confidence ≥ 0,5. Dengan hanya satu barang di tiap sisi tersisa 10 aturan yang ringkas.

### Langkah 3: confidence lolos tetapi lift gagal

```python title="filtering_thresholds.py" linenums="33"
--8<-- "association/filtering_thresholds.py:33:36"
```

Saringan: confidence ≥ 0,4 **tetapi** lift < 1,2. `popular["consequents"].apply(lambda s: ",".join(sorted(s))).value_counts()` menghitung konsekuen mana yang paling sering muncul di antara aturan itu.

```text
--8<-- "expected/association/filtering_thresholds.txt:8:8"
```

**Interpretasi.** 42 aturan, **seluruhnya berkonsekuen `kantong_plastik`**: barang populer yang muncul di hampir separuh keranjang. Saringan lift menyingkirkannya.

### Langkah 4: uji kestabilan sebagai pengganti pembagian latih/uji

**Konsep.** Tidak ada label untuk diprediksi, sehingga alternatifnya: tambang aturan dari paruh pertama transaksi, lalu periksa apakah aturan itu (misalnya liftnya) masih berlaku pada paruh kedua.

```python title="filtering_thresholds.py" linenums="38"
--8<-- "association/filtering_thresholds.py:38:57"
```

| Kode | Penjelasan |
|---|---|
| `df.iloc[:half]`, `df.iloc[half:]` | Paruh pertama dan kedua transaksi menurut urutan. |
| `mine(first, 0.02, 0.3)` lalu saringan lift dan confidence | Aturan hanya ditambang dari paruh pertama. |
| `def lift_in(data, ant, con)` | Menghitung lift sebuah aturan pada data lain: `data[list(ant)].all(axis=1)` bernilai `True` untuk keranjang yang memuat **semua** barang antecedent; rata-ratanya adalah peluangnya. Lift = \(P(A \cap B)/(P(A)P(B))\). |
| `(rules_a["lift_paruh2"] >= 1.2).mean()` | Proporsi aturan yang masih berlift ≥ 1,2 pada paruh kedua. |
| `diff = lift paruh 1 - lift paruh 2` | Seberapa besar lift bergeser. |

```text
--8<-- "expected/association/filtering_thresholds.txt:10:12"
```

**Interpretasi.** Dari 98 aturan yang lolos saringan pada paruh pertama, **99%** masih memiliki lift ≥ 1,2 pada paruh kedua. Median selisih lift hampir nol (-0,005), tetapi rentangnya lebar (-7,96 sampai +1,23): aturan dengan support terkecil (20 keranjang pada paruh 1) berayun besar karena sampel kecil. Aturan yang muncul pada sedikit keranjang tidak boleh dipercaya meski lift-nya tinggi.

??? example "Program lengkap (`filtering_thresholds.py`)"
    ```python title="filtering_thresholds.py" linenums="1"
    --8<-- "association/filtering_thresholds.py"
    ```

## Keterbatasan

| Keterbatasan | Penjelasan |
|---|---|
| Asosiasi bukan sebab-akibat | Pola belanja dapat dipengaruhi musim, promosi, atau penempatan rak |
| Banyak aturan, sedikit yang berguna | Perlu penyaringan dan penilaian manusia |
| Aturan sangat langka tidak stabil | Support kecil membuat lift berayun |
| Barang populer mendistorsi confidence | Gunakan lift dan ukuran lain |
| Mengabaikan kuantitas, harga, dan urutan waktu | Aturan klasik hanya melihat ada atau tidaknya barang |
| Hasil bergantung ambang | Laporkan ambang yang dipakai |

## Kesalahan umum

!!! warning "Hindari"
    - Menetapkan ambang agar menghasilkan "cukup banyak aturan" tanpa alasan.
    - Menafsirkan aturan langka berlift tinggi sebagai temuan.
    - Mengabaikan aturan redundan (A+B→C dan A→C yang hampir sama).
    - Mengira lolos saringan statistik berarti layak ditindaklanjuti.

## Latihan

1. Tambahkan saringan `leverage ≥ 0,02` dan lihat berapa aturan tersisa.
2. Ubah pembagian menjadi 30% pertama dan 70% sisanya. Apakah kestabilan berubah?
3. Susun daftar kriteria "aturan layak ditindaklanjuti" untuk tim merchandising.

## Ringkasan

Pilih ambang dengan mempertimbangkan ukuran data, lift, dan kegunaan; saring dengan lift dan ukuran antecedent; uji kestabilan pada bagian transaksi lain sebagai pengganti split latih/uji; dan ingat bahwa aturan asosiasi adalah pola, bukan sebab-akibat.

Berikutnya: [Proyek Analisis Keranjang Belanja](project.md).
