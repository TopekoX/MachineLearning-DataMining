# Proyek: Segmentasi Pelanggan Toko Online

## Tujuan proyek

Menerapkan alur klastering lengkap: EDA, praproses (log dan *scaling*), pemilihan \(k\) dengan metrik internal dan kestabilan, perbandingan dengan algoritma lain, profil dan nama segmen, tindakan bisnis, penempatan pelanggan baru, dan penyimpanan pipeline.

## Masalah dan data

**Masalah:** mengelompokkan pelanggan toko online berdasarkan pola belanja agar tim pemasaran dapat menyusun tindakan yang berbeda untuk tiap kelompok.

**Data:** **sintetis** bergaya RFM, 900 pelanggan dengan tiga fitur: *recency* (hari sejak pembelian terakhir), *frequency* (jumlah pesanan), dan *monetary* (total belanja, ribu Rp). Di balik data ada tiga kelompok tersembunyi (VIP, reguler, berisiko pergi) yang hanya dipakai untuk validasi eksternal, ditambah 8 pelanggan ekstrem (misalnya reseller) dengan belanja 15 kali lipat. Hasil tidak menjelaskan pelanggan nyata.

## Alur kerja

1. EDA: skewness dan *outlier*.
2. Praproses: `log1p` dan standardisasi di dalam pipeline.
3. Pilih \(k\): silhouette, Davies-Bouldin, kestabilan bootstrap.
4. Latih k-Means, bandingkan dengan Ward, profilkan, beri nama, rumuskan tindakan.
5. Simpan pipeline dan tempatkan pelanggan baru.

## Implementasi langkah demi langkah

Program `examples/clustering/project.py` berdiri sendiri. Konsep tiap langkah sudah dibahas di pelajaran sebelumnya; di sini fokusnya pada penyatuan langkah dan keputusan yang diambil. Program lengkap ada di akhir bagian ini.

### Langkah 1: membangkitkan data RFM sintetis

```python title="project.py" linenums="15"
--8<-- "clustering/project.py:15:30"
```

| Bagian | Penjelasan |
|---|---|
| `groups = {"VIP": (0.12, 10, 30, 14.0), ...}` | Tiga kelompok tersembunyi: (proporsi, recency tipikal, frekuensi tipikal, monetary tipikal). |
| `rng.gamma(4, rec / 4, n)` | *Recency*: distribusi gamma yang miring ke kanan dengan rata-rata ≈ `rec` hari. |
| `rng.poisson(freq, n) + 1` | *Frequency*: cacahan pesanan (minimal 1). |
| `rng.lognormal(np.log(mon * 100), 0.5, n)` | *Monetary*: distribusi lognormal (miring ke kanan, sesuai belanja nyata). |
| `df.loc[rng.choice(len(df), 8, replace=False), "monetary"] *= 15` | Delapan pelanggan ekstrem (misalnya reseller) dengan belanja 15 kali lipat. |
| `df[features].skew()` | Kemiringan tiap fitur. |

```text
--8<-- "expected/clustering/project.txt:1:1"
```

**Hasil.** 900 pelanggan; `monetary` sangat miring (skewness 20,33) karena pelanggan ekstrem, `frequency` 2,15, `recency` 1,44.

### Langkah 2: praproses (log dan standardisasi)

```python title="project.py" linenums="32"
--8<-- "clustering/project.py:32:35"
```

`Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])` menerapkan log lalu standardisasi pada **semua tiga kolom**. `prep.fit_transform(df[features])` mempelajari parameter standardisasi dan menghasilkan `Z`. Tanpa transformasi, pelanggan ekstrem akan menarik pusat klaster.

```text
--8<-- "expected/clustering/project.txt:2:2"
```

**Hasil.** Skewness setelah log: -0,03 (`recency`), 0,71 (`frequency`), 0,98 (`monetary`).

### Langkah 3: memilih \(k\)

**Konsep.** Beberapa bukti digabung: silhouette, Davies-Bouldin, dan kestabilan bootstrap ([penjelasan](choose-k-validation.md)).

```python title="project.py" linenums="37"
--8<-- "clustering/project.py:37:47"
```

Untuk tiap `k` dari 2 sampai 6: melatih k-Means, lalu 20 kali menghitung kestabilan (sampel bootstrap `Z[idx]` diklaster ulang dan dibandingkan dengan klaster penuh pada titik terambil dengan ARI).

```text
--8<-- "expected/clustering/project.txt:4:9"
```

**Interpretasi.** \(k=3\) terbaik pada ketiga ukuran: silhouette 0,446 (tertinggi), Davies-Bouldin 0,815 (terendah), dan kestabilan ARI 0,990 (tertinggi); \(k=2\) (0,924), \(k=4\) (0,907), \(k=5\) (0,871), \(k=6\) (0,917). Kesepakatan ini membuat \(k=3\) pilihan yang kuat, dan sejalan dengan kebutuhan bisnis (tiga kelompok mudah dikelola).

### Langkah 4: model akhir dan pembanding

```python title="project.py" linenums="49"
--8<-- "clustering/project.py:49:53"
```

k-Means dengan \(k=3\) dilatih, lalu dibandingkan dengan Agglomerative Ward (kesepakatan antar-algoritma) dan dengan kelompok sebenarnya (validasi eksternal, hanya tersedia karena data sintetis).

```text
--8<-- "expected/clustering/project.txt:11:11"
```

**Hasil.** ARI antara k-Means dan Ward 0,849 (mirip tetapi tidak identik); ARI k-Means terhadap kelompok sebenarnya 0,901.

### Langkah 5: profil, nama, dan tindakan

```python title="project.py" linenums="55"
--8<-- "clustering/project.py:55:67"
```

`df.groupby("cluster")[features].median()` memakai **median** (bukan rata-rata) agar profil tidak terpengaruh pelanggan ekstrem. Penamaan berbasis aturan: `monetary` terbesar = "VIP", `recency` terbesar = "berisiko pergi", sisanya "reguler". Kamus `actions` memetakan tiap segmen ke tindakan yang diusulkan.

```text
--8<-- "expected/clustering/project.txt:13:20"
```

**Interpretasi.** "VIP": 111 pelanggan, recency 10 hari, frekuensi 32, belanja median 1.353 (ribu Rp); "reguler": 452 pelanggan (recency 38, frekuensi 7, belanja 502); "berisiko pergi": 337 pelanggan (recency 205, frekuensi 3, belanja 279). Tindakan yang diusulkan (program loyalitas; promo bundel; kampanye reaktivasi) adalah hipotesis yang harus diuji, misalnya dengan eksperimen A/B.

### Langkah 6: menyimpan pipeline dan menempatkan pelanggan baru

```python title="project.py" linenums="69"
--8<-- "clustering/project.py:69:76"
```

`final` adalah pipeline akhir (praproses + k-Means) yang dilatih pada seluruh data; bersama kamus `names` disimpan dengan `joblib`. Dua pelanggan baru ditempatkan dengan `predict`, dan baris terakhir memeriksa bahwa penempatan seluruh data identik setelah model dimuat ulang.

```text
--8<-- "expected/clustering/project.txt:22:23"
```

**Hasil.** Pelanggan (recency 5, frekuensi 25, belanja 4.200) masuk "VIP"; pelanggan (recency 180, frekuensi 2, belanja 250) masuk "berisiko pergi". Pipeline yang dimuat ulang memberi hasil identik.

??? example "Program lengkap (`project.py`)"
    ```python title="project.py" linenums="1"
    --8<-- "clustering/project.py"
    ```

## Refleksi

- Segmentasi adalah alat bantu keputusan, bukan kebenaran; tinjau ulang secara berkala dengan [pemantauan jarak dan proporsi](saving-pipeline.md).
- Data sintetis memudahkan validasi eksternal; pada data nyata, andalkan kestabilan, interpretasi, dan uji tindakan.
- Pertimbangkan fitur tambahan (kategori produk, kanal), dengan memperhatikan *scaling* dan pengaruh pada jarak.

## Latihan

1. Tambahkan fitur `tenure_days` (lama menjadi pelanggan) dan ulangi pemilihan \(k\).
2. Hapus 8 pelanggan ekstrem dan bandingkan profilnya. Apakah segmennya berubah?
3. Gunakan `RobustScaler` alih-alih `StandardScaler` setelah log. Bagaimana kestabilan berubah?
4. Tulis ringkasan untuk tim pemasaran: tiga segmen, ciri, dan tindakan masing-masing, beserta keterbatasannya.

## Ringkasan

Proyek ini memperlihatkan alur klastering yang benar: transformasi dan *scaling* di dalam pipeline, pemilihan \(k\) dengan beberapa bukti, profil pada satuan asli, nama segmen sebagai hipotesis, dan penyimpanan pipeline untuk penempatan pelanggan baru.

Selesai dengan bab Klaster. Lanjut ke bab [Aturan Asosiasi](../association/index.md).
