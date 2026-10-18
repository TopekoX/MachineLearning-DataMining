# Proyek: Meramalkan Penjualan Bulanan End-to-End

## Tujuan proyek

Menerapkan alur forecasting lengkap: menyiapkan deret, membagi data kronologis, membandingkan metode dengan *rolling origin*, mengevaluasi sekali pada data uji, melatih ulang pada seluruh data, memprediksi 12 bulan ke depan dengan interval, dan menyimpan model.

## Masalah dan data

**Masalah:** merencanakan stok dengan meramalkan penjualan bulanan produk untuk 12 bulan ke depan.

**Data:** **sintetis**, dibangkitkan program dengan seed tetap: 84 bulan (Januari 2017–Desember 2023) berisi tren naik 2,2 per bulan, pola musiman dengan puncak Desember (+45) dan November (+22), dan derau normal dengan simpangan baku 8. Karena kita yang membuat datanya, kita tahu derau acaknya tidak dapat diprediksi: galat absolut rata-rata minimal sekitar 6,4 (≈ 0,8 × 8) bahkan bagi model yang sempurna. Hasil tidak menggambarkan penjualan nyata.

## Alur kerja

1. Periksa data dan pisahkan 12 bulan terakhir sebagai uji.
2. Evaluasi tiga metode (seasonal naive, Holt-Winters, SARIMA) pada 4 titik asal (akhir 2019–2022), horizon 12 bulan.
3. Pilih metode; evaluasi satu kali pada 12 bulan uji.
4. Latih ulang pada seluruh data, prediksi 2024 dengan interval 95%, simpan model.

## Implementasi langkah demi langkah

Program `examples/forecasting/project.py` berdiri sendiri (fungsi metrik ditulis ulang di dalamnya). Konsep tiap langkah sudah dibahas di pelajaran sebelumnya; di sini fokusnya pada penyatuan langkah. Program lengkap ada di akhir bagian ini.

### Langkah 1: membangkitkan data penjualan sintetis

```python title="project.py" linenums="28"
--8<-- "forecasting/project.py:28:36"
```

| Kode | Penjelasan |
|---|---|
| `pd.date_range("2017-01-01", "2023-12-01", freq="MS")` | 84 bulan dari Januari 2017 sampai Desember 2023. |
| `200 + 2.2 * t` | Tren: dasar 200 naik 2,2 per bulan. |
| `np.array([-10, ..., 45])[idx.month - 1]` | Pola musiman: array 12 nilai (satu per bulan, Desember +45); `[idx.month - 1]` mengambil nilai sesuai bulan tiap tanggal (bulan 1 → indeks 0). |
| `rng.normal(0, 8, len(idx))` | Derau acak, simpangan baku 8 (seed 7). |

```text
--8<-- "expected/forecasting/project.txt:1:3"
```

**Hasil.** 84 bulan; rata-rata tahunan naik dari 216 (2017) ke 376 (2023); Desember adalah bulan dengan rata-rata tertinggi.

### Langkah 2: pembagian dan daftar metode

**Konsep.** 12 bulan terakhir menjadi data uji dan tidak disentuh selama pemilihan. Pemilihan memakai 4 titik asal (akhir 2019, 2020, 2021, 2022) dengan horizon 12 bulan.

```python title="project.py" linenums="38"
--8<-- "forecasting/project.py:38:49"
```

`dev_origins = [len(train_all) - H * k for k in (3, 2, 1, 0)]` menghasilkan empat jumlah data latih yang masing-masing berselisih 12 bulan. `methods` adalah kamus `nama → fungsi(riwayat, horizon)` seperti di [lesson perbandingan](comparison-optimization.md): seasonal naive, Holt-Winters aditif, dan SARIMA.

### Langkah 3: evaluasi pengembangan

```python title="project.py" linenums="50"
--8<-- "forecasting/project.py:50:56"
```

Untuk tiap metode, *list comprehension* menghitung MAE di tiap titik asal (`sales.iloc[:e]` sebagai riwayat, `sales.iloc[e:e + H]` sebagai nilai sebenarnya); rata-rata dan simpangan bakunya disimpan dalam tabel.

```text
--8<-- "expected/forecasting/project.txt:5:10"
```

**Interpretasi.** Holt-Winters aditif MAE 6,7 (± 1,5), SARIMA 6,8 (± 1,3), seasonal naive 27,7 (± 3,5). **Seasonal naive begitu buruk** karena tren 2,2 per bulan menghasilkan selisih sekitar 26 per tahun yang tidak ditangkap, ditambah derau; galat 27,7 hampir seluruhnya akibat tren.

### Langkah 4: keputusan model dan evaluasi akhir

```python title="project.py" linenums="58"
--8<-- "forecasting/project.py:58:68"
```

Dua metode teratas hanya berbeda 0,1, jauh lebih kecil daripada simpangan bakunya (1,3–1,5), jadi dianggap setara. **SARIMA dipilih karena menghasilkan interval prediksi**; keputusan ini ditetapkan **sebelum** melihat data uji. Kemudian model dilatih pada seluruh data sebelum 12 bulan terakhir (`train_all`) dan dinilai **sekali** pada `test`, dibandingkan dengan *seasonal naive*.

```text
--8<-- "expected/forecasting/project.txt:12:14"
```

**Interpretasi.** SARIMA MAE 4,8; MASE 0,18, dibanding seasonal naive MAE 29,7; MASE 1,14. Galat 4,8 lebih rendah dari perkiraan batas 6,4 (derau acak saja sudah ≈ 0,8 × 8) karena hanya 12 titik (variasi sampel); galat pengembangan 6,7 lebih mencerminkan kinerja yang dapat diharapkan, dan sangat dekat dengan batas derau. Batas itu hanya kita ketahui karena datanya sintetis.

### Langkah 5: melatih ulang pada seluruh data dan memprediksi 2024

```python title="project.py" linenums="70"
--8<-- "forecasting/project.py:70:75"
```

| Kode | Penjelasan |
|---|---|
| `SARIMAX(sales, ...).fit(disp=False)` | Latih ulang pada **seluruh** 84 bulan: setelah keputusan dibuat, semua data tersedia dipakai. |
| `final.get_forecast(H)` | Prediksi 12 bulan ke depan (2024) dengan interval. |
| `fc.conf_int(alpha=0.05).set_axis(["bawah_95", "atas_95"], axis=1)` | Interval 95% dengan nama kolom yang jelas; `join` menggabungkannya dengan prediksi titik. |
| `.round(0).astype(int).iloc[[0, 5, 10, 11]]` | Pembulatan, lalu menampilkan empat bulan (Januari, Juni, November, Desember). |

```text
--8<-- "expected/forecasting/project.txt:16:21"
```

**Interpretasi.** Prediksi 2024: Januari 376 (interval 362–391), Juni 399 (384–413), November 435 (420–450), Desember 453 (438–468). Lebar interval sekitar ±15, konsisten dengan derau 8 (1,96 × ~7,5). Interval berlaku bila pola masa lalu berlanjut.

### Langkah 6: menyimpan model

```python title="project.py" linenums="77"
--8<-- "forecasting/project.py:77:79"
```

```text
--8<-- "expected/forecasting/project.txt:23:23"
```

**Hasil.** Prediksi identik setelah model dimuat ulang.

??? example "Program lengkap (`project.py`)"
    ```python title="project.py" linenums="1"
    --8<-- "forecasting/project.py"
    ```



## Refleksi

- Karena derau data dikendalikan, kita dapat melihat bahwa model sudah mendekati batas; pada data nyata batas ini tidak diketahui, sehingga *baseline* dan sebaran antar-asal menjadi acuan.
- Interval harus diuji cakupannya sebelum dipakai untuk keputusan stok (lihat [Horizon dan Interval](horizon-intervals.md)).
- Pantau galat setelah deployment dan rencanakan pelatihan ulang ([Monitoring](saving-monitoring.md)).

## Latihan

1. Ubah simpangan baku derau dari 8 menjadi 20. Bagaimana MAE dan lebar interval berubah?
2. Tambahkan model Ridge berbasis lag ([lesson regresi](regression-forecasting.md)) ke perbandingan.
3. Hitung cakupan empiris interval 95% pada titik asal pengembangan.
4. Tulis ringkasan satu paragraf untuk manajer stok: apa yang dapat dan tidak dapat dijamin dari ramalan ini?

## Ringkasan

Proyek ini menunjukkan alur forecasting yang benar: evaluasi kronologis dan multi-asal, *baseline* musiman, keputusan yang dibuat sebelum membuka data uji, pelatihan ulang pada seluruh data, serta prediksi beserta interval. Pada data sintetis, kita dapat membandingkan hasil dengan batas derau yang diketahui.

Selesai dengan bab Forecasting. Lanjut ke bab [Klaster](../clustering/index.md).
