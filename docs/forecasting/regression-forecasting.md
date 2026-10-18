# Forecasting dengan Regresi dan Fitur Lag

## Tujuan pembelajaran

- [x] Membangun model forecasting dari tabel fitur lag dengan Ridge dan Random Forest.
- [x] Membedakan prediksi satu langkah dan prediksi rekursif multi-langkah.
- [x] Menjelaskan mengapa model berbasis pohon memerlukan target selisih pada deret bertren.

## Konsep

Setelah deret diubah menjadi tabel fitur ([lesson fitur](features.md)), model regresi apa pun dapat dipakai. Dua hal khusus forecasting:

1. **Prediksi multi-langkah.** Fitur lag untuk bulan berikutnya memerlukan nilai bulan ini, yang pada saat prediksi belum diketahui. Pada **prediksi rekursif**, hasil prediksi dimasukkan kembali sebagai lag. Galat menumpuk seiring horizon. Alternatifnya, **pendekatan langsung** melatih satu model per horizon.
2. **Tren dan model pohon.** Pohon dan hutan memprediksi rata-rata nilai target di daun, sehingga **tidak dapat memprediksi di luar rentang target latih**. Pada deret yang terus naik, cara umum adalah memprediksi **selisih** \(y_t - y_{t-1}\), lalu menambahkannya kembali ke level terakhir.

## Implementasi langkah demi langkah

CO2 bulanan, fitur: lag 1, 2, 3, 12 dan sin/cos bulan. Model dilatih pada data sampai 1997-12 dan dievaluasi pada 24 bulan validasi dengan dua cara: **1-langkah** (lag memakai data aktual) dan **24-langkah rekursif** (realistis untuk peramalan). Program `examples/forecasting/regression_forecasting.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: fungsi pembuat fitur untuk satu bulan

**Tujuan.** Membuat satu baris fitur untuk bulan `date` yang **hanya** memakai data sebelum bulan itu.

```python title="regression_forecasting.py" linenums="10"
--8<-- "forecasting/regression_forecasting.py:10:20"
```

| Kode | Penjelasan |
|---|---|
| `LAGS = [1, 2, 3, 12]` | Lag yang dipakai: 1, 2, 3, dan 12 bulan lalu. |
| `v = history.to_numpy()` | Riwayat sebagai array; `v[-k]` adalah nilai `k` bulan sebelum bulan yang diprediksi (indeks negatif dari belakang). |
| `row = [v[-k] for k in LAGS]` | Empat fitur lag. |
| `row += [np.sin(...), np.cos(...)]` | Dua fitur kalender (bulan sebagai lingkaran). `+=` menambahkan elemen ke daftar. |

Karena `make_row` hanya menerima `history`, fiturnya tidak mungkin memuat masa depan.

### Langkah 2: tabel fitur–target untuk pelatihan

**Konsep.** Untuk **target selisih** (`use_diff=True`), model memprediksi perubahan \(y_t - y_{t-1}\) lalu menambahkannya ke nilai terakhir. Untuk **target level**, model memprediksi \(y_t\) langsung.

```python title="regression_forecasting.py" linenums="23"
--8<-- "forecasting/regression_forecasting.py:23:29"
```

Perulangan berjalan untuk tiap bulan `i` (mulai dari `max(LAGS)` = 12, karena baris awal belum punya lag lengkap). `series.iloc[:i]` adalah riwayat sebelum bulan `i`, jadi fitur selalu berasal dari masa lalu. `series.iloc[i] - series.iloc[i - 1] if use_diff else series.iloc[i]` memilih target selisih atau level (ekspresi kondisional satu baris). Hasilnya diubah menjadi array NumPy `X` dan `t`.

### Langkah 3: prediksi rekursif multi-langkah

**Konsep.** Untuk meramal 24 bulan, fitur bulan berikutnya memerlukan nilai bulan ini yang belum diketahui. Solusinya: **hasil prediksi dimasukkan kembali** sebagai riwayat untuk langkah berikutnya. Galat dapat menumpuk seiring horizon.

```python title="regression_forecasting.py" linenums="32"
--8<-- "forecasting/regression_forecasting.py:32:43"
```

| Kode | Penjelasan |
|---|---|
| `hist = history.copy()` | Salinan agar riwayat asli tidak berubah. |
| `date = hist.index[-1] + pd.offsets.MonthBegin(1)` | Tanggal bulan berikutnya (satu awal-bulan setelah tanggal terakhir). |
| `p = model.predict(x)[0]` | Prediksi satu langkah (satu angka). |
| `value = hist.iloc[-1] + p if use_diff else p` | Bila target selisih, nilai = nilai terakhir + selisih terprediksi; bila level, nilai = prediksi. |
| `hist = pd.concat([hist, pd.Series([value], index=[date])])` | Menambahkan prediksi ke riwayat, sehingga menjadi lag untuk langkah berikutnya. |

### Langkah 4: mengevaluasi dua cara

```python title="regression_forecasting.py" linenums="46"
--8<-- "forecasting/regression_forecasting.py:46:58"
```

`evaluate` melatih model pada `train`, lalu menghitung dua hal pada `valid`: **prediksi 1-langkah** (untuk tiap bulan validasi, lag diambil dari data **aktual**, jadi hanya memeriksa kemampuan meramal satu bulan ke depan) dan **prediksi rekursif 24 langkah** (lag memakai prediksi sebelumnya; ini sesuai kebutuhan peramalan nyata). MAE dan MASE dicetak.

### Langkah 5: membandingkan empat konfigurasi

```python title="regression_forecasting.py" linenums="61"
--8<-- "forecasting/regression_forecasting.py:61:70"
```

Empat konfigurasi: Ridge dan Random Forest, masing-masing pada target level dan target selisih. Baris terakhir mencetak rentang nilai CO2 pada data latih dan validasi, untuk menjelaskan hasil pohon.

```text
--8<-- "expected/forecasting/regression_forecasting.txt"
```

**Interpretasi.**

- **Random Forest pada target level gagal**: MAE 1-langkah 2,17 dan rekursif 2,41 (MASE 1,93), bahkan lebih buruk dari Ridge. Rentang CO2 data latih berakhir di 366,7 ppm sedangkan validasi mencapai 371,0: hutan tidak dapat memprediksi di atas rentang latihnya.
- **Target selisih memperbaiki Random Forest secara dramatis**: MAE 1-langkah 0,22 dan rekursif **0,64** (MASE 0,51), terbaik di antara semua model pada tabel ini. Tren ditangani oleh penjumlahan selisih, sedangkan hutan hanya mempelajari pola musiman dan perubahan.
- **Ridge** (model linear) dapat mengekstrapolasi, sehingga baik pada level (rekursif MAE 1,07; MASE 0,86) dan sedikit lebih baik pada selisih (0,97; MASE 0,78).
- **Galat rekursif lebih besar dari galat 1-langkah** pada semua model (misalnya 0,64 vs 0,22): kesalahan awal terbawa ke langkah berikutnya. Menilai model hanya dengan prediksi 1-langkah terlalu optimistis untuk peramalan 24 bulan.
- **Pembanding**: seasonal naive memberi MAE 3,72 (MASE 2,98), dan Holt-Winters 1,37 pada validasi yang sama (lihat [baseline](baselines.md)). Namun hasil ini satu titik asal pada satu deret yang halus; belum cukup untuk menyatakan model pohon "terbaik" (lihat [perbandingan](comparison-optimization.md)).

??? example "Program lengkap (`regression_forecasting.py`)"
    ```python title="regression_forecasting.py" linenums="1"
    --8<-- "forecasting/regression_forecasting.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai model pohon pada level deret bertren tanpa menyadari batas ekstrapolasinya.
    - Mengevaluasi hanya dengan prediksi 1-langkah padahal kebutuhannya multi-langkah.
    - Memakai `train_test_split` acak untuk tabel fitur lag.
    - Menghitung fitur lag dari nilai prediksi yang bukan hasil rekursi sesungguhnya saat evaluasi multi-langkah.

## Latihan

1. Tambahkan `lag_24` dan lihat pengaruhnya.
2. Bandingkan `GradientBoostingRegressor` pada target selisih.
3. Buat pendekatan langsung: satu model per horizon 1..12. Bandingkan dengan rekursif.

## Ringkasan

Forecasting dengan regresi membutuhkan fitur yang hanya memakai masa lalu, strategi multi-langkah (rekursif atau langsung), dan penanganan tren (target selisih untuk model pohon). Nilai model dengan prediksi multi-langkah yang realistis.

Berikutnya: [ARIMA dan SARIMA](arima.md).
