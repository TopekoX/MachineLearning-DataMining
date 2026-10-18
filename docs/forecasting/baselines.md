# Baseline, Moving Average, dan Exponential Smoothing

## Tujuan pembelajaran

- [x] Membuat *baseline* forecasting: naive, seasonal naive, drift, dan rata-rata bergerak.
- [x] Menjelaskan *exponential smoothing* (SES, Holt, Holt-Winters).
- [x] Memilih metode pada data validasi lalu menguji sekali pada data uji.
- [x] Menafsirkan hasil dari satu titik asal dengan hati-hati.

## Metode sederhana (baseline)

Dengan data sampai waktu \(T\) dan horizon \(h\):

| Metode | Rumus prediksi \(\hat{y}_{T+h}\) | Cocok bila |
|---|---|---|
| Naive | \(y_T\) | Deret acak tanpa pola |
| Seasonal naive | \(y_{T+h-m}\) (nilai musim yang sama tahun lalu) | Musiman kuat |
| Drift | \(y_T + h\cdot\dfrac{y_T - y_1}{T-1}\) | Tren hampir lurus |
| Rata-rata bergerak | rata-rata \(k\) nilai terakhir | Deret berfluktuasi di sekitar tingkat tetap |

Metode ini wajib dicoba lebih dulu: model rumit hanya bermanfaat bila mengalahkannya.

## Exponential smoothing

Alih-alih rata-rata biasa, bobot data lama mengecil secara eksponensial. *Simple exponential smoothing* (SES) memperbarui tingkat (*level*) \(\ell_t\):

$$
\ell_t = \alpha\, y_t + (1-\alpha)\,\ell_{t-1}, \qquad 0 \le \alpha \le 1
$$

\(\alpha\) besar = mengikuti data terbaru; \(\alpha\) kecil = lebih halus. Perluasannya:

- **Holt** menambah komponen tren (\(\beta\)).
- **Holt-Winters** menambah komponen musiman (\(\gamma\)), aditif atau multiplikatif.
- **Tren teredam** (*damped*) mengecilkan tren seiring horizon agar ekstrapolasi tidak lepas kendali.

Parameter diestimasi otomatis oleh `statsmodels`.

## Implementasi langkah demi langkah

Dataset CO2 bulanan. Semua metode dilatih sampai 1997-12 dan dinilai pada 24 bulan validasi (1998–1999) dari **satu titik asal**. Metode terbaik menurut MASE validasi dilatih ulang pada latih+validasi dan diuji sekali pada 2000–2001. Program `examples/forecasting/baselines.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: data dan pembagian

```python title="baselines.py" linenums="1"
--8<-- "forecasting/baselines.py:1:13"
```

`warnings.filterwarnings("ignore")` menyembunyikan peringatan konvergensi statsmodels agar keluaran ringkas (pada pekerjaan nyata, baca peringatan itu). Pembagian kronologis sama dengan [lesson validasi](splitting-evaluation.md): `H = len(valid)` = 24 = horizon prediksi.

### Langkah 2: satu fungsi untuk semua metode

**Konsep.** Fungsi `forecasts(history, horizon)` menerima data masa lalu dan menghasilkan prediksi sejumlah `horizon` langkah dengan **semua** metode. Karena hanya memakai `history`, tidak ada kebocoran dari masa depan, dan fungsi yang sama dapat dipakai untuk validasi maupun pengujian akhir.

```python title="baselines.py" linenums="16"
--8<-- "forecasting/baselines.py:16:33"
```

| Kode | Penjelasan |
|---|---|
| `last = history.iloc[-1]` | Nilai terakhir yang diketahui. |
| `np.repeat(last, horizon)` | **Naive**: mengulang nilai terakhir sebanyak `horizon` kali. |
| `history.iloc[-12:].mean()` | **Rata-rata 12 bulan terakhir**, diulang. |
| `last + drift * steps` | **Drift**: garis lurus dari nilai terakhir dengan kemiringan rata-rata historis (`(akhir - awal) / jumlah selang`). `steps = 1, 2, ..., horizon`. |
| `seasonal_naive(history, horizon)` | Menyalin 12 nilai terakhir (musim yang sama tahun lalu). |
| `SimpleExpSmoothing(history, initialization_method="estimated").fit().forecast(horizon)` | **SES**: memodelkan hanya *level*. `initialization_method="estimated"` berarti nilai awal ikut diestimasi; `fit()` mengestimasi parameter pemulusan (\(\alpha\)); `forecast(h)` memprediksi `h` langkah. |
| `Holt(...)` | Menambah **tren** (\(\beta\)). Dengan `damped_trend=True` tren **diredam** seiring horizon. |
| `ExponentialSmoothing(history, trend="add", seasonal="add", seasonal_periods=12)` | **Holt-Winters**: level + tren aditif + musiman aditif periode 12. |
| `.to_numpy()` | Mengubah hasil menjadi array NumPy agar semua metode berbentuk sama. |

Fungsi mengembalikan **kamus** `{nama metode: prediksi}`.

### Langkah 3: evaluasi pada data validasi

```python title="baselines.py" linenums="36"
--8<-- "forecasting/baselines.py:36:41"
```

`forecasts(train, H)` memberi prediksi 24 bulan dari akhir data latih. Perulangan (*list comprehension*) menghitung MAE, RMSE, dan MASE tiap metode terhadap `valid`; MASE memakai `train` untuk menghitung skala *seasonal naive* ([rumus](splitting-evaluation.md#metrik)). Tabel diurutkan menurut MASE.

```text
--8<-- "expected/forecasting/baselines.txt:1:10"
```

**Interpretasi.**

- **Holt-Winters unggul jauh**: MAE 1,37; MASE 1,10 pada validasi, karena ia memodelkan tren dan musiman sekaligus. Drift (MAE 2,34) menangkap tren tetapi bukan musiman.
- **SES identik dengan Naive** (MAE 3,20; RMSE 3,75): parameter \(\alpha\) yang diestimasi mendekati 1, sehingga SES hanya menyalin nilai terakhir. Rata-rata 12 bulan dan seasonal naive sama-sama MAE 3,72: keduanya menghapus siklus musiman tetapi tertinggal oleh tren.
- **Holt (tren) gagal total**: MAE 20,21. Tanpa komponen musiman, parameter yang diestimasi (\(\alpha\) dan \(\beta\) mendekati 1) membuat tren sama dengan perubahan bulan terakhir, yang sangat dipengaruhi musim, lalu diekstrapolasi selama 24 bulan. **Tren teredam** meredam hal ini (MAE 3,07). Pelajarannya: ekstrapolasi tren tanpa batas berbahaya.

### Langkah 4: memilih dan menguji sekali

```python title="baselines.py" linenums="43"
--8<-- "forecasting/baselines.py:43:50"
```

| Kode | Penjelasan |
|---|---|
| `table["MASE"].idxmin()` | Nama metode dengan MASE validasi terkecil. |
| `pd.concat([train, valid])` | Menggabungkan latih dan validasi: setelah memilih, semua data yang tersedia sebelum periode uji dipakai untuk melatih ulang. |
| `forecasts(trainval, len(test))[best]` | Prediksi 24 bulan uji dari metode terpilih; `[best]` mengambil dari kamus. |

```text
--8<-- "expected/forecasting/baselines.txt:12:14"
```

**Interpretasi.** Holt-Winters dipilih, dilatih ulang pada latih+validasi, dan diuji sekali pada 2000–2001: MAE 0,24; RMSE 0,31; MASE 0,18, jauh lebih baik daripada seasonal naive (MAE 1,88; MASE 1,45).

!!! warning "Hati-hati dengan satu titik asal"
    Galat Holt-Winters pada validasi 1,37 tetapi pada uji hanya 0,24. Perbedaan 5 kali lipat ini menunjukkan hasil dari satu titik asal sangat bergantung pada kebetulan periode; itu sebabnya [rolling origin](splitting-evaluation.md) dan [perbandingan multi-asal](comparison-optimization.md) diperlukan sebelum memercayai satu angka.

??? example "Program lengkap (`baselines.py`)"
    ```python title="baselines.py" linenums="1"
    --8<-- "forecasting/baselines.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Melewatkan *baseline* musiman.
    - Menganggap tren selalu boleh diekstrapolasi; gunakan tren teredam atau uji dahulu.
    - Menyimpulkan dari satu titik asal.
    - Memilih metode dengan data uji.

## Latihan

1. Ganti `seasonal="add"` menjadi `"mul"` pada Holt-Winters. Apakah MAE validasi berubah?
2. Ubah jendela rata-rata bergerak menjadi 3 dan 24 bulan.
3. Cetak `smoothing_level`, `smoothing_trend`, `smoothing_seasonal` dari model Holt-Winters.

## Ringkasan

Mulai dari naive, seasonal naive, dan drift; exponential smoothing menambahkan tren dan musiman secara bertahap. Pilih dengan data validasi, uji sekali, dan waspadai hasil satu titik asal serta ekstrapolasi tren.

Berikutnya: [Forecasting dengan Regresi dan Fitur Lag](regression-forecasting.md).
