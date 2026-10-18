# Konsep Klastering dan Persiapan Data

## Tujuan pembelajaran

- [x] Menjelaskan *unsupervised learning* dan tujuan klastering.
- [x] Menjelaskan mengapa prosedur train/test terawasi tidak berlaku langsung pada klastering, dan apa gantinya.
- [x] Melakukan EDA dan menentukan praproses untuk data klastering.

## Konsep

Klastering mengelompokkan data berdasarkan **kemiripan tanpa label**. Tidak ada jawaban benar yang tersedia sebagai acuan; "kelompok" adalah struktur yang kita temukan, bukan kebenaran yang diprediksi.

| | Klasifikasi | Klastering |
|---|---|---|
| Label | Ada | Tidak ada |
| Tujuan | Memprediksi kelas | Menemukan struktur dan menjelaskannya |
| Evaluasi | Skor pada data uji berlabel | Indeks internal, kestabilan, interpretasi (validasi eksternal bila ada acuan) |
| Hasil | Satu jawaban benar per sampel | Bergantung pada algoritma, parameter, dan skala |

Penggunaan umum: segmentasi pelanggan, pengelompokan dokumen, deteksi pola, kompresi data, dan eksplorasi awal.

## Mengapa tidak ada "data uji berlabel"?

Pada klasifikasi, data uji memiliki label sehingga kita bisa menghitung kesalahan. Pada klastering tidak ada label, jadi tidak ada yang bisa "benar" atau "salah" secara langsung. Yang dapat diuji:

1. **Kestabilan**: apakah struktur yang sama ditemukan pada data lain atau sampel berbeda? Praproses tetap dipelajari dari data latih, lalu data baru ditempatkan ke klaster yang ada.
2. **Kualitas internal**: seberapa padat dan terpisah klasternya (silhouette, Davies-Bouldin).
3. **Kegunaan**: apakah klaster dapat ditafsirkan dan dimanfaatkan?
4. **Validasi eksternal** (hanya jika ada label acuan): kecocokan dengan label yang diketahui (ARI, NMI).

## Dataset: pelanggan sintetis

Data **sintetis** (dibangkitkan program dengan seed tetap): 600 pelanggan dengan lima fitur numerik `annual_spend` (juta Rp/tahun), `visits_per_month`, `avg_basket` (ribu Rp), `recency_days`, dan `age`. Di balik data ada empat kelompok tersembunyi (`segment_true`: pemburu diskon, loyal, musiman, tidak aktif) yang **hanya dipakai untuk validasi eksternal dalam materi ini**; pada masalah nyata kelompok seperti itu tidak diketahui. Karena sintetis, hasil tidak menjelaskan pelanggan nyata.

??? example "Kode pembangkit data (`segment_data.py`)"
    ```python title="segment_data.py"
    --8<-- "clustering/segment_data.py"
    ```

## Implementasi langkah demi langkah

Program `examples/clustering/concepts_data.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: EDA data pelanggan

**Tujuan.** Mengenal ukuran, kualitas, dan skala fitur sebelum mengklaster.

```python title="concepts_data.py" linenums="1"
--8<-- "clustering/concepts_data.py:1:17"
```

| Kode | Penjelasan |
|---|---|
| `from segment_data import make_customer_segments` | Mengimpor pembangkit data sintetis dari berkas `segment_data.py` di folder yang sama. |
| `X = df[features]` | Hanya lima kolom fitur untuk klastering. Kolom `segment_true` **sengaja tidak** dimasukkan ke `X`. |
| `X.describe().T[["mean", "std", "min", "max"]]` | Ringkasan statistik per fitur; `.T` memutar tabel agar satu fitur per baris. |
| `X["annual_spend"].skew()` | Kemiringan distribusi belanja. |

```text
--8<-- "expected/clustering/concepts_data.txt:1:8"
```

**Interpretasi.** 600 baris, tanpa data hilang atau duplikat. **Skala sangat berbeda**: `avg_basket` rata-rata 207 (maksimum 834) sedangkan `visits_per_month` rata-rata 4,9; `recency_days` bersimpangan baku 99, `age` 11,8. Algoritma berbasis jarak akan didominasi fitur berskala besar, sehingga *scaling* wajib (lihat [lesson berikutnya](scaling-distance.md)). `annual_spend` miring ke kanan (skewness 2,08, maksimum 101 vs rata-rata 13); transformasi log membantu (dipakai pada [profil klaster](profiling.md)).

### Langkah 2: kelompok sebenarnya (hanya untuk validasi eksternal)

```python title="concepts_data.py" linenums="19"
--8<-- "clustering/concepts_data.py:19:20"
```

```text
--8<-- "expected/clustering/concepts_data.txt:10:10"
```

Kolom `segment_true` menyimpan kelompok yang dipakai pembangkit data: pemburu diskon (181), loyal (119), musiman (150), dan tidak aktif (150). Ini **tidak tersedia pada masalah nyata**; di sini dipakai hanya untuk memeriksa hasil klastering.

### Langkah 3: apa pengganti "data uji" pada klastering?

**Konsep.** Tanpa label tidak ada galat yang bisa dihitung pada data uji. Yang dapat diuji adalah **kestabilan**: apakah struktur yang sama muncul pada data lain? Caranya: pelajari praproses dan klaster dari data latih (70%), lalu bandingkan dua cara mengelompokkan 30% data sisanya.

```python title="concepts_data.py" linenums="22"
--8<-- "clustering/concepts_data.py:22:31"
```

| Kode | Penjelasan |
|---|---|
| `train_test_split(np.arange(len(df)), test_size=0.3, ...)` | Membagi **nomor baris** (bukan data) menjadi indeks latih dan uji; indeks dipakai untuk memilih baris dengan `iloc`. |
| `StandardScaler().fit(X.iloc[train_idx])` | Scaler **belajar hanya dari data latih** (rata-rata dan simpangan baku). |
| `KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(...)` | k-Means dengan 4 klaster dilatih pada data latih terskala; `n_init=10` mengulang inisialisasi 10 kali dan memilih yang terbaik. |
| `km.predict(scaler.transform(X.iloc[test_idx]))` | **Penempatan**: data uji diubah dengan scaler **latih** lalu ditempatkan ke pusat klaster terdekat (tanpa melatih ulang). |
| `fit_predict(StandardScaler().fit_transform(X.iloc[test_idx]))` | Cara kedua: data uji **diklaster sendiri** dengan scaler dan k-Means barunya. |
| `adjusted_rand_score(a, b)` | ARI: seberapa mirip dua pengelompokan (1 = identik, ≈ 0 = acak); nomor klasternya tidak perlu sama. |

```text
--8<-- "expected/clustering/concepts_data.txt:12:12"
```

**Interpretasi.** ARI kestabilan 0,957: model dari data latih dan klastering mandiri pada data uji hampir sama, sehingga struktur yang ditemukan bukan kebetulan satu sampel.

### Langkah 4: validasi eksternal

```python title="concepts_data.py" linenums="33"
--8<-- "clustering/concepts_data.py:33:35"
```

```text
--8<-- "expected/clustering/concepts_data.txt:13:13"
```

**Interpretasi.** ARI penempatan data uji terhadap kelompok sebenarnya 0,849 (hanya mungkin karena data sintetis). Struktur cukup jelas tetapi tidak sempurna; tiap kelompok sengaja dibuat saling tumpang tindih.

??? example "Program lengkap (`concepts_data.py`)"
    ```python title="concepts_data.py" linenums="1"
    --8<-- "clustering/concepts_data.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Mengklaster tanpa *scaling* pada fitur berskala berbeda.
    - Menganggap klaster sebagai "kebenaran" dan bukan sebagai gambaran data.
    - Memilih jumlah klaster hanya dari satu metrik.
    - Meng-*fit* praproses pada seluruh data sebelum memeriksa kestabilan.
    - Mengklaster data berisi banyak fitur tidak relevan.

## Latihan

1. Ubah `n_clusters` menjadi 3 dan 5 pada bagian kestabilan. Bagaimana ARI kestabilannya?
2. Hapus fitur `age` dan ulangi. Apakah ARI terhadap kelompok sebenarnya berubah?
3. Mengapa pada klastering kita tetap boleh membagi data? Jelaskan apa tujuannya dibanding pada klasifikasi.

## Ringkasan

Klastering tanpa label tidak memiliki data uji berlabel; kualitasnya dinilai dengan indeks internal, kestabilan, interpretasi, dan (bila ada) validasi eksternal. Siapkan data dengan EDA dan *scaling*, dan simpan praproses di dalam pipeline.

Berikutnya: [Scaling dan Ukuran Jarak](scaling-distance.md).
