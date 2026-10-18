# Proyek: Analisis Keranjang Belanja Minimarket

## Tujuan proyek

Menerapkan alur aturan asosiasi lengkap: membersihkan transaksi, mengodekan, menambang itemset sering dengan FP-Growth, membentuk dan menyaring aturan, mengurangi redundansi, menguji kestabilan, dan menyiapkan tabel aturan untuk keputusan merchandising.

## Masalah dan data

**Masalah:** menemukan barang yang sering dibeli bersamaan agar tim merchandising dapat menyusun paket (bundling), penempatan rak, dan rekomendasi.

**Data:** **sintetis**, 3000 transaksi dari 20 barang minimarket (seed berbeda dari lesson sebelumnya), dengan beberapa pola yang sengaja ditanam dan satu barang sangat populer (`kantong_plastik`). Hasil tidak mencerminkan perilaku belanja nyata.

## Alur kerja

1. Normalisasi nama barang, hapus duplikat, dan buang keranjang kosong.
2. `TransactionEncoder` lalu FP-Growth dengan `min_support=0,02`.
3. Bentuk aturan; saring dengan lift ≥ 1,2, confidence ≥ 0,5, dan antecedent ≤ 2 barang.
4. Kurangi redundansi: satu aturan terbaik per itemset gabungan.
5. Uji kestabilan: tambang dari paruh pertama, ukur lift pada paruh kedua.
6. Simpan tabel aturan.

Tidak ada prediksi, pembagian latih/uji klasik, atau penyajian model lewat API di proyek ini: keluaran akhirnya adalah **tabel aturan** yang ditinjau manusia.

## Implementasi langkah demi langkah

Program `examples/association/project.py`. Konsep tiap langkah sudah dibahas di pelajaran sebelumnya; di sini fokusnya pada penyatuan langkah. Program lengkap ada di akhir bagian ini.

### Langkah 1: pembersihan dan pengodean

```python title="project.py" linenums="8"
--8<-- "association/project.py:8:15"
```

`make_transactions(n=3000, seed=11)` membuat 3000 keranjang sintetis (seed berbeda dari lesson sebelumnya). Pembersihan: `{item.strip().lower() for item in basket}` menormalkan nama dan membuang duplikat; `if basket` membuang keranjang kosong. `TransactionEncoder` lalu membuat matriks `True`/`False` ([penjelasan](transactions-encoding.md)).

```text
--8<-- "expected/association/project.txt:1:1"
```

**Hasil.** 3000 transaksi dengan rata-rata 3,27 barang per keranjang.

### Langkah 2: penambangan dan penyaringan aturan

```python title="project.py" linenums="17"
--8<-- "association/project.py:17:22"
```

| Kode | Penjelasan |
|---|---|
| `fpgrowth(df, min_support=0.02, ...)` | Itemset sering pada seluruh data (tidak ada pembagian latih/uji pada tugas ini). |
| `association_rules(..., metric="lift", min_threshold=1.2)` | Hanya aturan dengan lift ≥ 1,2. |
| `rules["confidence"] >= 0.5` dan `rules["antecedents"].apply(len) <= 2` | Tambahan saringan: confidence ≥ 0,5 dan antecedent paling banyak dua barang. |

```text
--8<-- "expected/association/project.txt:2:2"
```

**Hasil.** FP-Growth menemukan 163 itemset sering; saringan meninggalkan 71 aturan.

### Langkah 3: mengurangi redundansi

**Konsep.** Satu itemset gabungan dapat menghasilkan banyak aturan yang pada dasarnya sama (misalnya `A+B → C` dan `A → B+C`). Kita mempertahankan **satu aturan terbaik per itemset gabungan**.

```python title="project.py" linenums="25"
--8<-- "association/project.py:25:34"
```

| Kode | Penjelasan |
|---|---|
| `frozenset(a \| c)` | Gabungan antecedent `a` dan consequent `c` sebagai satu itemset (`\|` = gabungan himpunan). |
| `rules.assign(items=[...])` | Menambahkan kolom `items` dengan itemset gabungan tiap aturan. |
| `sort_values(["lift", "confidence"], ascending=False).drop_duplicates("items")` | Mengurutkan dari lift terbesar, lalu membuang baris berikutnya yang itemset gabungannya sudah muncul: tersisa aturan terbaik per itemset. |

```text
--8<-- "expected/association/project.txt:3:13"
```

**Interpretasi.** 71 aturan dirampingkan menjadi **42** itemset gabungan unik. Pasangan yang jelas dan dapat ditindaklanjuti: `sampo → sabun` (support 0,075; confidence 0,685; lift 4,70), `kopi + roti → gula` (lift 4,55), dan `air_mineral + gula → kopi` (confidence 0,856; lift 4,90). Pola ini cocok untuk uji bundling "kopi + gula", penempatan rak berdekatan, atau rekomendasi.

!!! warning "Hati-hati dengan lift ekstrem"
    Aturan teratas `gula + mentega → kopi + roti` memiliki lift 16,5, tetapi aturan ini menggabungkan **dua pola yang sebenarnya terpisah**, (kopi, gula) dan (roti, mentega), yang kebetulan muncul bersamaan. Karena masing-masing pola cukup langka, peluang bersama yang kecil dibagi dengan hasil kali peluang yang lebih kecil lagi menghasilkan lift raksasa. Informasi yang bermakna ada pada dua pasangan itu, bukan pada aturan empat barang ini. Lift tinggi pada itemset besar sering menunjukkan struktur gabungan, bukan pola baru.

### Langkah 4: uji kestabilan

**Konsep.** Sebagai pengganti validasi, aturan ditambang hanya dari paruh pertama transaksi lalu diperiksa pada paruh kedua ([penjelasan](filtering-thresholds.md#langkah-4-uji-kestabilan-sebagai-pengganti-pembagian-latihuji)).

```python title="project.py" linenums="36"
--8<-- "association/project.py:36:53"
```

`r1` adalah aturan dari paruh pertama (dengan saringan yang sama); `top` adalah lima dengan lift tertinggi. `lift_in(second, ...)` menghitung lift tiap aturan pada paruh kedua. Perulangan mencetak lift paruh 1 dan paruh 2 berdampingan.

```text
--8<-- "expected/association/project.txt:15:20"
```

**Interpretasi.** Aturan teratas paruh pertama bertahan pada paruh kedua (misalnya lift 16,44 menjadi 16,58; 15,87 menjadi 16,13), sedangkan `mentega + telur → mie_instan + roti` turun dari 9,63 menjadi 8,24, yaitu masih tinggi tetapi lebih berayun. Dengan data sintetis yang pola tanamnya stabil, hasil ini wajar; pada data nyata, musim dan promosi sering menurunkan kestabilan.

### Langkah 5: tabel keluaran

```python title="project.py" linenums="55"
--8<-- "association/project.py:55:57"
```

```text
--8<-- "expected/association/project.txt:22:22"
```

**Hasil.** Tabel 42 baris (aturan, support, confidence, lift) siap disimpan, misalnya dengan `out.to_csv("aturan.csv", index=False)`, untuk ditinjau tim merchandising. Sama seperti hasil lain di bab ini, tabel ini tidak memerlukan "model tersimpan" atau API.

??? example "Program lengkap (`project.py`)"
    ```python title="project.py" linenums="1"
    --8<-- "association/project.py"
    ```

## Refleksi

- Gunakan lift dan *support* bersama; periksa aturan besar dengan membaca aturan pasangannya.
- Terapkan penilaian manusia sebelum tindakan; uji tindakan lewat eksperimen kecil, misalnya paket promosi pada beberapa toko.
- Ulangi analisis berkala dan bandingkan dengan periode sebelumnya.

## Latihan

1. Turunkan `min_support` menjadi 0,01. Apa yang berubah pada jumlah dan kualitas aturan?
2. Buat saringan yang menghapus aturan dengan konsekuen `kantong_plastik` dan lihat dampaknya.
3. Hitung kestabilan memakai pembagian tiga bagian (sepertiga pertama, tengah, akhir).
4. Tulis rekomendasi untuk tim merchandising berdasarkan tiga aturan pilihan Anda, beserta alasan dan keterbatasan.

## Ringkasan

Proyek ini memperlihatkan alur aturan asosiasi yang benar: pembersihan dan pengodean, penambangan dengan ambang yang disengaja, penyaringan berbasis lift, pengurangan redundansi, uji kestabilan sebagai pengganti validasi, dan keluaran berupa tabel untuk keputusan manusia.

Selesai dengan bab Aturan Asosiasi dan seluruh kurikulum. Kembali ke [Peta Belajar](../roadmap/index.md) untuk meninjau seluruh alur kerja.
