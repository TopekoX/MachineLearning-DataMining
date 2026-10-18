# ARIMA dan SARIMA

## Tujuan pembelajaran

- [x] Menjelaskan komponen AR, I, dan MA pada ARIMA, serta bagian musiman pada SARIMA.
- [x] Membandingkan beberapa orde dengan AIC pada data latih.
- [x] Membaca interval prediksi dan mengujinya secara empiris.
- [x] Memeriksa residual dengan uji Ljung-Box.

## Konsep

**ARIMA(p, d, q)** menggabungkan tiga bagian:

- **AR(p)** (*autoregressive*): nilai sekarang dimodelkan dari \(p\) nilai sebelumnya.
- **I(d)** (*integrated*): deret didiferensiasi \(d\) kali agar tren hilang.
- **MA(q)** (*moving average*): nilai sekarang dipengaruhi \(q\) galat prediksi sebelumnya.

Untuk deret yang sudah didiferensiasi \(w_t\):

$$
w_t = c + \phi_1 w_{t-1} + \dots + \phi_p w_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}
$$

\(\phi_i\) koefisien AR, \(\theta_j\) koefisien MA, dan \(\varepsilon_t\) derau. **SARIMA(p,d,q)(P,D,Q,m)** menambahkan bagian yang sama pada lag musiman \(m\) (misalnya 12) dan diferensiasi musiman \(D\).

Orde dapat dibandingkan dengan **AIC** (makin kecil makin baik; menghukum jumlah parameter). AIC hanya sah membandingkan model dengan data dan diferensiasi yang sama, dan **bukan** ukuran akurasi prediksi.

## Interval prediksi

ARIMA menghasilkan distribusi prediksi, sehingga dapat memberi **interval prediksi** (misalnya 95%) yang melebar seiring horizon. Interval yang baik harus **terbukti** mencakup nilai sebenarnya sekitar 95% dari waktu; ini diperiksa dengan data yang tidak dipakai melatih.

## Implementasi langkah demi langkah

SARIMAX dari `statsmodels`, dengan \(d=1\), \(D=1\), \(m=12\), dilatih sampai 1997-12 dan dinilai pada 24 bulan validasi. Program `examples/forecasting/arima.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: data dan pembagian

```python title="arima.py" linenums="1"
--8<-- "forecasting/arima.py:1:13"
```

Pembagian kronologis sama dengan pelajaran sebelumnya. `warnings.filterwarnings("ignore")` menyembunyikan peringatan konvergensi `statsmodels` agar keluaran ringkas (pada pekerjaan nyata, baca peringatan itu).

### Langkah 2: memilih orde dengan AIC

**Konsep.** Orde `(p, d, q)(P, D, Q, m)` ditentukan; kita mencoba beberapa kandidat pada **data latih** dan membandingkan AIC (makin kecil makin baik).

```python title="arima.py" linenums="15"
--8<-- "forecasting/arima.py:15:29"
```

| Kode | Penjelasan |
|---|---|
| `((1, 1, 1), (0, 1, 1, 12))` | Pasangan `order=(p, d, q)` = `(1, 1, 1)` dan `seasonal_order=(P, D, Q, m)` = `(0, 1, 1, 12)`. Semua kandidat memakai \(d=1\) dan \(D=1\) dengan periode 12. |
| `for order, seasonal in candidates` | Mengulang tiap kandidat; pasangan dipecah ke dua variabel. |
| `SARIMAX(train, order=order, seasonal_order=seasonal).fit(disp=False)` | Menginisialisasi lalu mengestimasi model pada data latih; `disp=False` mematikan pesan optimisasi. Hasilnya objek `res`. |
| `res.aic` | Nilai AIC model yang dilatih. |
| `pd.DataFrame(rows).sort_values("AIC")` | Tabel hasil, diurutkan dari AIC terendah. |

```text
--8<-- "expected/forecasting/arima.txt:1:6"
```

**Interpretasi.** AIC terendah pada (1,1,1)(0,1,1,12) dengan 245,0; (2,1,1)(0,1,1,12) 246,1 dan (1,1,1)(1,1,1,12) 246,3 hampir sama; (0,1,1)(0,1,1,12) 250,1; (1,1,0)(0,1,1,12) 259,9 paling buruk (tanpa komponen MA nonmusiman). Selisih AIC di bawah sekitar 2 umumnya dianggap tidak bermakna, jadi tiga model teratas hampir setara.

### Langkah 3: prediksi, interval, dan akurasi

```python title="arima.py" linenums="31"
--8<-- "forecasting/arima.py:31:41"
```

| Kode | Penjelasan |
|---|---|
| `best = aic.iloc[0]` | Baris pertama tabel (AIC terendah). |
| `res.get_forecast(24)` | Prediksi 24 langkah beserta ketidakpastiannya. |
| `fc.predicted_mean` | Prediksi titik (nilai tengah). |
| `fc.conf_int(alpha=0.05)` | **Interval prediksi 95%** (`alpha=0.05` = 5% di luar interval): tabel dengan batas bawah dan atas tiap bulan. |
| `((valid >= batas_bawah) & (valid <= batas_atas)).mean()` | Proporsi nilai **aktual** yang jatuh di dalam interval: cakupan empiris. |

```text
--8<-- "expected/forecasting/arima.txt:8:11"
```

**Interpretasi.**

- **Akurasi** (24 bulan): MAE 1,11; RMSE 1,19; MASE 0,89. Lebih baik daripada seasonal naive (MAE 3,72) dan Holt-Winters pada validasi yang sama (MAE 1,37), tetapi dari satu titik asal.
- **Interval melebar seiring horizon**: lebar interval 95% adalah 1,20 ppm (bulan ke-1), 2,41 (bulan ke-12), dan 3,42 (bulan ke-24).
- **Interval terlalu sempit pada uji ini**: hanya **54%** (sekitar 13 dari 24) nilai aktual jatuh dalam interval 95%. Beberapa kemungkinan penyebab (tidak diuji di sini): laju tren yang berubah dari periode latih, ketidakpastian parameter yang tidak masuk interval, dan asumsi varians galat yang konstan. Ini hasil satu titik asal dengan 24 titik berkorelasi, bukan estimasi cakupan yang andal; [lesson berikutnya](horizon-intervals.md) mengujinya dengan banyak titik asal.

### Langkah 4: diagnostik residual

**Konsep.** Bila model sudah menangkap struktur linear deret, residual-nya menyerupai derau putih (tanpa autokorelasi). Uji **Ljung–Box** menguji hipotesis "tidak ada autokorelasi": **p-value kecil** berarti masih ada pola yang terlewat.

```python title="arima.py" linenums="43"
--8<-- "forecasting/arima.py:43:48"
```

`res.resid` adalah residual satu-langkah pada data latih; `.iloc[13:]` membuang 13 residual awal yang dipengaruhi inisialisasi diferensiasi (satu diferensiasi biasa ditambah satu musiman 12). `acorr_ljungbox(resid, lags=[12, 24])` menguji sampai lag 12 dan 24.

```text
--8<-- "expected/forecasting/arima.txt:13:14"
```

**Interpretasi.** Ljung-Box p-value 0,470 (lag 12) dan 0,565 (lag 24): tidak ada bukti autokorelasi tersisa; rata-rata residual 0,021 dan simpangan baku 0,317. Residual yang menyerupai derau putih adalah tanda model telah menangkap struktur linear yang ada.

??? example "Program lengkap (`arima.py`)"
    ```python title="arima.py" linenums="1"
    --8<-- "forecasting/arima.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memilih orde dengan AIC lalu menganggapnya otomatis akurat.
    - Mempercayai interval prediksi tanpa memeriksa cakupannya.
    - Lupa periode musiman `m` atau salah diferensiasi.
    - Mengabaikan peringatan konvergensi tanpa memeriksa model.
    - Memakai ARIMA pada deret dengan perubahan struktural tanpa menanganinya.

## Latihan

1. Coba `seasonal_order=(0,1,1,12)` dengan `order=(0,1,2)`. Bagaimana AIC dan MAE validasi?
2. Plot `res.plot_diagnostics()` dan jelaskan keempat panelnya.
3. Gunakan interval 80% dan hitung cakupan empirisnya.

## Ringkasan

ARIMA/SARIMA memodelkan nilai dari nilai dan galat sebelumnya pada deret yang didiferensiasi. AIC membantu memilih orde, residual harus menyerupai derau putih, dan interval prediksi harus diuji, bukan dipercaya begitu saja.

Berikutnya: [Horizon dan Interval Prediksi](horizon-intervals.md).
