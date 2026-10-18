# Perbandingan dan Optimasi Model Forecasting

## Tujuan pembelajaran

- [x] Membandingkan banyak metode forecasting dengan *rolling origin* yang sama.
- [x] Mengoptimasi varian model tanpa memakai data uji.
- [x] Memutuskan model terpilih dengan mempertimbangkan sebaran antar titik asal.

## Ringkasan metode

| Metode | Prinsip | Kekuatan | Keterbatasan |
|---|---|---|---|
| Seasonal naive | Salin musim yang sama tahun lalu | Sederhana, *baseline* wajib | Tidak menangkap tren |
| Drift | Ekstrapolasi garis lurus | Menangkap tren | Tanpa musiman |
| Holt-Winters | Tingkat, tren, musiman yang dihaluskan | Cepat, sedikit parameter | Perlu pilihan aditif/multiplikatif dan peredaman tren |
| SARIMA | Autoregresi dan rata-rata bergerak pada deret terdiferensiasi | Interval prediksi, teori matang | Penyetelan orde, asumsi linear |
| Regresi + fitur lag | Tabel lag dan kalender dengan Ridge/Random Forest | Fleksibel, fitur eksternal | Perlu strategi multi-langkah dan penanganan tren |

## Prosedur

1. Sisihkan **24 bulan terakhir sebagai data uji** dan jangan dipakai selama pemilihan.
2. Evaluasi semua kandidat pada **6 titik asal** (data latih berakhir 1993-12 sampai 1998-12), masing-masing horizon 12 bulan, yang prediksinya tidak menyentuh data uji.
3. Bandingkan MAE rata-rata **dan sebarannya antar-asal**.
4. Pilih satu metode, lalu evaluasi sekali pada data uji bersama *baseline* seasonal naive.

Optimasi dilakukan dengan membandingkan varian pada titik asal yang sama: Holt-Winters aditif vs multiplikatif vs tren teredam, dan Ridge dengan `alpha` 0,1 vs 10. (Untuk penyetelan fitur ML secara lebih luas, gunakan `TimeSeriesSplit` agar validasi tetap kronologis.)

## Implementasi langkah demi langkah

Program `examples/forecasting/comparison_optimization.py`. Sembilan metode dibandingkan pada titik asal yang sama. Program lengkap ada di akhir bagian ini.

### Langkah 1: fungsi prediksi berbasis regresi (selisih bulanan)

**Tujuan.** Membungkus model ML (Ridge, Random Forest) menjadi fungsi `ml_forecast(model, history, horizon)` yang bentuknya sama dengan metode statistik, sehingga semua metode dapat dinilai dalam satu perulangan. Idenya sama dengan [pelajaran regresi untuk forecasting](regression-forecasting.md): fitur lag dari **selisih** bulanan dan prediksi **rekursif**.

```python title="comparison_optimization.py" linenums="16"
--8<-- "forecasting/comparison_optimization.py:16:35"
```

| Kode | Penjelasan |
|---|---|
| `d = history.diff()` | Selisih bulanan; ini target. |
| `X = pd.DataFrame({f"lag_{k}": d.shift(k) for k in LAGS})` | Fitur lag dari selisih, hanya memakai masa lalu. |
| `ok = X.notna().all(axis=1) & d.notna()` | Baris yang fitur dan targetnya lengkap (membuang baris awal yang kosong). |
| `model = make_model().fit(X[ok], d[ok])` | `make_model` adalah **fungsi** pembuat model (misalnya `lambda: Ridge(alpha=10)`); dipanggil `()` untuk membuat model baru lalu dilatih. |
| `for _ in range(horizon): ...` | Prediksi rekursif: tiap langkah memakai selisih terakhir (termasuk prediksi sebelumnya) sebagai lag, lalu `level += step` mengakumulasi selisih menjadi level. |

### Langkah 2: pabrik Holt-Winters dan daftar metode

```python title="comparison_optimization.py" linenums="38"
--8<-- "forecasting/comparison_optimization.py:38:53"
```

`hw(trend, damped, seasonal)` mengembalikan sebuah **fungsi** `lambda tr, h: ...` yang melatih Holt-Winters dengan pengaturan tersebut; dengan begitu varian aditif, teredam, dan multiplikatif dibuat dari satu definisi. `methods` adalah kamus `nama → fungsi(riwayat, horizon)`: setiap metode, apa pun jenisnya, dipanggil dengan cara yang sama. Varian yang dibandingkan di sini sekaligus menjadi bentuk sederhana **optimasi model**: Holt-Winters aditif vs teredam vs musiman multiplikatif, dan Ridge `alpha` 0,1 vs 10.

### Langkah 3: titik asal pengembangan

**Konsep.** Data uji (24 bulan terakhir) **tidak boleh disentuh** selama pemilihan. Titik asal dipilih sehingga prediksi 12 bulannya berakhir paling lambat di akhir data pengembangan.

```python title="comparison_optimization.py" linenums="55"
--8<-- "forecasting/comparison_optimization.py:55:58"
```

```text
--8<-- "expected/forecasting/comparison_optimization.txt:1:1"
```

Enam titik asal dari 1993-12 sampai 1998-12; prediksinya paling jauh sampai 1999-12, sebelum data uji (2000–2001).

### Langkah 4: menilai semua metode pada semua titik asal

```python title="comparison_optimization.py" linenums="60"
--8<-- "forecasting/comparison_optimization.py:60:69"
```

Perulangan luar melewati metode, perulangan dalam melewati titik asal. Pada tiap titik asal, `fn(tr, 12)` memprediksi 12 bulan memakai **hanya** `tr` (data sampai titik asal). MAE, RMSE, dan MASE dikumpulkan; tabel meringkas rata-rata dan simpangan baku MAE antar titik asal (`Std MAE`), lalu diurutkan menurut MAE.

```text
--8<-- "expected/forecasting/comparison_optimization.txt:2:12"
```

**Interpretasi.**

- **Seasonal naive (MAE 1,87) dan drift (2,06) jauh tertinggal.** Semua metode lain berada di kisaran 0,49–0,60 (MASE 0,39–0,48).
- **Tujuh metode teratas tidak dapat dibedakan.** Selisih MAE rata-rata antar mereka (0,49 sampai 0,60) lebih kecil dari simpangan baku antar-asal (0,24–0,36). Urutan bisa berubah bila titik asal diganti.
- **Pemilihan**: Holt-Winters musiman multiplikatif memiliki MAE rata-rata terendah (0,49 ± 0,24; MASE 0,39), disusul SARIMA (0,51 ± 0,27) dan Random Forest berbasis selisih (0,52 ± 0,36). Random Forest memiliki sebaran terbesar, tanda hasilnya kurang stabil antar-asal.
- **Optimasi varian memberi perubahan kecil**: HW aditif 0,57, tren teredam 0,59, multiplikatif 0,49; Ridge `alpha=10` 0,59 vs `alpha=0,1` 0,60. Dengan hanya 6 titik asal, selisih seperti ini belum bermakna.

### Langkah 5: evaluasi akhir sekali pada data uji

```python title="comparison_optimization.py" linenums="71"
--8<-- "forecasting/comparison_optimization.py:71:77"
```

`table.index[0]` adalah nama metode dengan MAE terendah. Metode itu dan *seasonal naive* dilatih pada seluruh data sebelum 2000 (`train`) lalu dinilai pada 24 bulan uji **sekali**; `:<26` meratakan nama ke kiri selebar 26 karakter.

```text
--8<-- "expected/forecasting/comparison_optimization.txt:14:16"
```

**Interpretasi.** Holt-Winters multiplikatif: MAE 0,44; RMSE 0,54; MASE 0,34, dibanding seasonal naive MAE 1,88; MASE 1,45. Keunggulan atas *baseline* jelas; keunggulan atas model lain tidak dapat diklaim. Rekomendasi praktis: bila beberapa metode setara, pilih yang paling sederhana, paling cepat, atau yang menyediakan interval prediksi.

??? example "Program lengkap (`comparison_optimization.py`)"
    ```python title="comparison_optimization.py" linenums="1"
    --8<-- "forecasting/comparison_optimization.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memilih berdasarkan satu titik asal atau selisih yang lebih kecil dari sebaran.
    - Menyetel parameter dengan titik asal yang menyentuh data uji.
    - Membandingkan metode pada horizon dan data latih yang berbeda.
    - Melaporkan hanya rata-rata tanpa sebaran.

## Latihan

1. Tambahkan `SARIMA (0,1,1)(0,1,1,12)` dan Holt-Winters dengan `seasonal_periods=6`. Bagaimana peringkatnya?
2. Perbanyak titik asal menjadi 10. Apakah urutan kandidat berubah?
3. Tambahkan model ensambel: rata-rata prediksi HW dan SARIMA. Apakah galatnya lebih kecil?

## Ringkasan

Bandingkan metode dengan titik asal yang sama, perhatikan sebaran, dan jaga data uji tetap utuh. Banyak metode yang layak sering setara; keunggulan atas *baseline* sederhana adalah yang paling penting dibuktikan.

Berikutnya: [Menyimpan Model, Deployment, dan Monitoring](saving-monitoring.md).
