# Menyimpan Pipeline Klastering dan Memantau Segmen

## Tujuan pembelajaran

- [x] Menyimpan pipeline klastering (praproses + k-Means) beserta nama segmen.
- [x] Menempatkan pelanggan baru ke segmen tanpa melatih ulang.
- [x] Mendeteksi pelanggan yang tidak cocok dengan segmen mana pun dan memantau pergeseran segmen.

!!! info "Prasyarat"
    Dasar `joblib` dan keamanan berkas model dibahas di [Klasifikasi: Simpan, Deploy, Monitor](../classification/saving-deployment.md).

## Yang khas untuk klastering

- **Tidak ada "prediksi benar"**, tetapi ada **penempatan** (`predict`): titik baru masuk ke pusat terdekat memakai praproses dan pusat dari data latih.
- **Nomor klaster tidak stabil antar-pelatihan.** Bila model dilatih ulang, klaster "0" bisa menjadi segmen berbeda. Simpan **nama segmen** bersama pipeline dan petakan ulang dari profil.
- **Pelanggan di luar pola.** Titik jauh dari semua pusat tetap ditempatkan ke pusat terdekat; periksa jaraknya.
- **Pemantauan** menggunakan proporsi segmen dan jarak ke pusat, bukan akurasi.

## Implementasi langkah demi langkah

Memakai data pelanggan sintetis ([penjelasan](concepts-data.md)). Batch "bergeser" adalah **simulasi sintetis**: nilai `recency_days` dikalikan 3 dan `visits_per_month` dikalikan 0,5, sebagai tiruan pelanggan yang makin jarang kembali. Program `examples/clustering/clustering_saving.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: melatih pipeline

```python title="clustering_saving.py" linenums="14"
--8<-- "clustering/clustering_saving.py:14:23"
```

Pipeline sama dengan [pelajaran profil](profiling.md): transformasi log pada fitur miring, standardisasi, lalu k-Means (4 klaster), dilatih pada 600 pelanggan.

### Langkah 2: menamai segmen dan menyimpan bundel

**Konsep.** Nomor klaster tidak stabil antar-pelatihan (klaster "0" hari ini bisa menjadi segmen lain setelah dilatih ulang). Karena itu **nama segmen disimpan bersama pipeline** sebagai satu bundel.

```python title="clustering_saving.py" linenums="25"
--8<-- "clustering/clustering_saving.py:25:41"
```

| Kode | Penjelasan |
|---|---|
| `pipe["km"].cluster_centers_` | Pusat klaster (pada skala terstandardisasi); `pipe["km"]` mengambil langkah `km` menurut namanya. |
| `pipe[:-1].named_steps["scale"].inverse_transform(...)` | Mengembalikan pusat ke skala sebelum standardisasi (tetapi **masih dalam log** untuk dua kolom miring). Kolom diberi nama dengan urutan yang sama seperti keluaran `ColumnTransformer` (kolom log dulu). |
| `max(left, key=rule)` | Memilih nomor klaster dengan nilai terbesar menurut aturan `rule` (fungsi `lambda`), dari klaster yang tersisa. |
| `joblib.dump({"pipeline": pipe, "names": names, "features": features}, path)` | Menyimpan **kamus**: pipeline, nama segmen, dan daftar fitur. |
| `joblib.load(path)` | Memuat bundel; `bundle["names"]` kembali berisi nama segmen. |

```text
--8<-- "expected/clustering/clustering_saving.txt:1:1"
```

**Hasil.** Nama segmen tersimpan: 0 musiman, 1 loyal, 2 tidak aktif, 3 pemburu diskon.

### Langkah 3: menempatkan pelanggan baru

```python title="clustering_saving.py" linenums="43"
--8<-- "clustering/clustering_saving.py:43:47"
```

`new` adalah tabel tiga pelanggan baru dengan kolom yang sama seperti data latih. `bundle["pipeline"].predict(new[features])` menerapkan log dan standardisasi **dengan parameter dari data latih**, lalu menempatkan tiap pelanggan ke pusat klaster terdekat; **tidak ada pelatihan ulang**. `bundle["names"][int(c)]` mengubah nomor klaster menjadi nama.

```text
--8<-- "expected/clustering/clustering_saving.txt:2:2"
```

**Hasil.** Tiga pelanggan ditempatkan ke "loyal", "tidak aktif", dan "musiman".

### Langkah 4: mendeteksi pelanggan yang tidak cocok dengan segmen mana pun

**Konsep.** Pelanggan yang sangat berbeda tetap ditempatkan ke pusat terdekat. Jarak ke pusat terdekat menunjukkan seberapa "asing" ia bagi semua segmen.

```python title="clustering_saving.py" linenums="49"
--8<-- "clustering/clustering_saving.py:49:55"
```

| Kode | Penjelasan |
|---|---|
| `bundle["pipeline"].transform(...)` | Untuk pipeline yang berakhir pada k-Means, `transform` mengembalikan **jarak ke tiap pusat klaster** (satu kolom per klaster). |
| `.min(axis=1)` | Jarak ke pusat **terdekat** untuk tiap pelanggan (per baris). |
| `np.percentile(train_dist, 99)` | Batas: persentil ke-99 jarak pada data latih. Pelanggan dengan jarak lebih besar dianggap tidak lazim. |
| `odd = pd.DataFrame([[400.0, 20.0, ...]])` | Satu pelanggan ekstrem **buatan** untuk menguji deteksi. |

```text
--8<-- "expected/clustering/clustering_saving.txt:4:5"
```

**Interpretasi.** Batas jarak 2,37. Pelanggan baru normal berjarak 0,27; 0,80; dan 0,22 dari pusat terdekat, sedangkan pelanggan ekstrem buatan berjarak 29,99 dan ditandai melebihi batas: tetap masuk ke segmen terdekat tetapi perlu ditinjau.

### Langkah 5: memantau proporsi segmen dan jarak

```python title="clustering_saving.py" linenums="57"
--8<-- "clustering/clustering_saving.py:57:76"
```

| Kode | Penjelasan |
|---|---|
| `make_customer_segments(n=600, seed=7)` | Batch baru dengan distribusi yang sama (seed berbeda). |
| `shifted[...] * 3`, `* 0.5` | Batch dengan perilaku bergeser (simulasi). |
| `def share(df): ...` | Menempatkan batch ke segmen, memberi nama, lalu menghitung **proporsi** tiap segmen (`value_counts(normalize=True) * 100`). |
| `pd.DataFrame({...}).fillna(0)` | Menggabungkan proporsi tiga kelompok data; segmen yang tidak muncul diisi 0. |

```text
--8<-- "expected/clustering/clustering_saving.txt:7:17"
```

**Interpretasi.**

- **Proporsi segmen**: batch dengan distribusi sama memiliki selisih proporsi terbesar 3 poin persen terhadap data latih; batch bergeser 7 poin persen. Sinyal ini **lemah**: pergeseran besar pada fitur hanya sedikit mengubah proporsi karena titik tetap jatuh ke segmen terdekat.
- **Jarak ke pusat terdekat** lebih sensitif: rata-rata 1,13 (latih), 1,12 (batch sama), dan **1,48** (batch bergeser). Kenaikan itu menandakan data makin tidak cocok dengan segmen yang dikenal, sinyal untuk meninjau dan melatih ulang model.

??? example "Program lengkap (`clustering_saving.py`)"
    ```python title="clustering_saving.py" linenums="1"
    --8<-- "clustering/clustering_saving.py"
    ```

## Kapan melatih ulang?

- Jarak rata-rata ke pusat naik jelas dibanding periode latih.
- Proporsi segmen berubah besar atau muncul kelompok baru.
- Kebutuhan bisnis berubah (fitur baru, definisi segmen baru).
- Setelah melatih ulang, petakan ulang nama segmen dan bandingkan dengan segmentasi lama (ARI) sebelum menggantinya.

## Kesalahan umum

!!! warning "Hindari"
    - Mengandalkan nomor klaster antar-versi model.
    - Melatih ulang praproses pada data baru saat hanya ingin menempatkan.
    - Mengabaikan pelanggan yang jauh dari semua pusat.
    - Memantau hanya proporsi segmen.
    - Memuat berkas model dari sumber tidak tepercaya.

## Latihan

1. Hitung ARI antara penempatan model lama dan model yang dilatih ulang pada batch bergeser.
2. Tentukan ambang jarak rata-rata yang memicu pelatihan ulang dan alasannya.
3. Tambahkan label segmen dan jarak ke keluaran penempatan pelanggan baru.

## Ringkasan

Simpan pipeline lengkap dengan nama segmen, tempatkan pelanggan baru tanpa *fit* ulang, tandai pelanggan yang jauh dari segmen, dan pantau jarak ke pusat selain proporsi segmen.

Berikutnya: [Proyek Segmentasi Pelanggan](project.md).
