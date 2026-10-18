# Forecasting

Forecasting memprediksi **nilai di masa depan** dari data berurutan waktu, misalnya penjualan bulanan atau konsentrasi CO2. Urutan waktu adalah informasi penting, sehingga cara membagi, memvalidasi, dan mengevaluasi data berbeda dari bab lain.

## Tujuan pembelajaran

- Mengenali tren, musiman (*seasonality*), dan derau pada deret waktu.
- Membuat *baseline* sederhana sebelum model yang kompleks.
- Memvalidasi secara kronologis dan dengan *rolling origin* tanpa kebocoran dari masa depan.
- Mengevaluasi dengan MAE, RMSE, sMAPE, dan MASE; membaca galat menurut horizon dan interval prediksi.
- Membandingkan exponential smoothing, SARIMA, dan regresi berbasis fitur lag.

## Prasyarat

[Dasar-dasar](../fundamentals/index.md) (khususnya [pandas](../fundamentals/pandas.md), [Statistik](../fundamentals/statistics.md), [Data Latih, Validasi, Uji](../fundamentals/data-splitting.md)) dan bab [Estimasi](../estimation/index.md), karena model berbasis lag adalah regresi.

!!! note "Pustaka tambahan"
    Bab ini memakai `statsmodels` (dekomposisi, exponential smoothing, SARIMA) selain pustaka inti. Pasang lewat `requirements-ml.txt`.

## Tahap alur kerja yang relevan

| Tahap | Status | Catatan |
|---|---|---|
| Definisi masalah | Wajib | Tentukan *horizon* dan frekuensi |
| Pengumpulan data | Wajib | Indeks waktu teratur, periksa timestamp dan nilai hilang |
| EDA | Berbeda | Plot waktu, dekomposisi, autokorelasi |
| Praproses | Berbeda | *Resampling*, pengisian yang tidak memakai masa depan |
| Rekayasa fitur | Berbeda | Lag dan kalender untuk model ML; tidak perlu untuk model klasik |
| Pembagian data | Berbeda | Kronologis, *rolling origin*; **tidak boleh diacak** |
| Evaluasi | Berbeda | Bandingkan dengan *baseline* naive dan seasonal naive |
| Optimasi | Wajib | Validasi berurutan waktu |
| Penyimpanan, deployment, monitoring | Wajib/Opsional | Pelatihan ulang terjadwal, pemantauan galat |

## Pelajaran

| # | Pelajaran | Isi utama |
|---|---|---|
| 1 | [Dasar Deret Waktu](ts-fundamentals.md) | Tren, musiman, derau, ACF, diferensiasi |
| 2 | [Data dan EDA](data-eda.md) | CO2 Mauna Loa, celah data, *resample* |
| 3 | [Fitur Waktu dan Lag](features.md) | Lag, statistik bergulir, kebocoran |
| 4 | [Validasi Kronologis dan Metrik](splitting-evaluation.md) | Split, *rolling origin*, MAE/RMSE/sMAPE/MASE |
| 5 | [Baseline dan Exponential Smoothing](baselines.md) | Naive, drift, SES, Holt, Holt-Winters |
| 6 | [Regresi dan Fitur Lag](regression-forecasting.md) | Ridge dan Random Forest, rekursif |
| 7 | [ARIMA dan SARIMA](arima.md) | Orde, AIC, residual, interval |
| 8 | [Horizon dan Interval Prediksi](horizon-intervals.md) | Galat per horizon, cakupan interval |
| 9 | [Perbandingan dan Optimasi](comparison-optimization.md) | Multi-asal, pemilihan, uji akhir |
| 10 | [Simpan, Deploy, Monitor](saving-monitoring.md) | Model menua, ambang peringatan |
| 11 | [Proyek: Penjualan Bulanan](project.md) | Alur lengkap pada data sintetis |

## Data yang dipakai

- **CO2 Mauna Loa** (data publik dibundel statsmodels, domain publik) untuk pelajaran 2–10.
- **Data sintetis** untuk pengantar komponen (pelajaran 1) dan proyek; ditandai sebagai sintetis.

Contoh kode ada di `examples/forecasting/`; keluaran yang ditampilkan berasal dari program yang dijalankan.
