# Menyimpan Model, Deployment, dan Monitoring Forecasting

## Tujuan pembelajaran

- [x] Menyimpan dan memuat model deret waktu.
- [x] Menentukan ambang peringatan dari galat acuan.
- [x] Memantau galat setelah data baru tiba dan menentukan kapan melatih ulang.

!!! info "Prasyarat"
    Dasar `joblib`, keamanan berkas model, dan bentuk deployment dibahas di [Klasifikasi: Simpan, Deploy, Monitor](../classification/saving-deployment.md).

## Yang khas untuk forecasting

- **Model menua.** Deret waktu berubah (tren, pola musiman, perubahan struktural), sehingga model harus dilatih ulang secara terjadwal atau saat galat melonjak.
- **Prediksi punya tanggal.** Catat sampai kapan data latih dan mulai kapan prediksi.
- **Label datang terlambat tetapi pasti.** Nilai sebenarnya akan diketahui pada waktunya, sehingga galat dapat dihitung terus-menerus.
- **Pembanding pemantauan**: bandingkan galat dengan *baseline* seasonal naive dan dengan galat acuan saat pengembangan.

## Implementasi langkah demi langkah

Model SARIMA dilatih sampai 1999-12; data 2000–2001 diperlakukan sebagai "data baru yang tiba kemudian". Ambang peringatan = 2 × MAE acuan dari *rolling origin*. Bagian akhir **mensimulasikan** perubahan struktural (aktual dinaikkan 3 ppm sejak 2000-07); data ini sintetis, bukan kejadian nyata. Program `examples/forecasting/saving_monitoring.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: melatih dan menyimpan model

```python title="saving_monitoring.py" linenums="14"
--8<-- "forecasting/saving_monitoring.py:14:20"
```

`train, new_data = y.iloc[:-24], y.iloc[-24:]` memisahkan data yang diketahui saat model dipasang dari data yang "datang kemudian". `SARIMAX(...).fit(disp=False)` melatih model dan menghasilkan objek hasil `res`. `joblib.dump(res, path)` menyimpan **objek hasil statsmodels** ke berkas; objek ini sudah memuat parameter terestimasi dan keadaan model sehingga dapat memprediksi lagi.

### Langkah 2: memuat ulang dan memverifikasi

```python title="saving_monitoring.py" linenums="22"
--8<-- "forecasting/saving_monitoring.py:22:26"
```

`loaded.forecast(24)` memprediksi 24 bulan setelah akhir data latih; `np.allclose` membandingkan dengan prediksi model asli. Mencatat **tanggal akhir data latih** dan **tanggal mulai prediksi** penting agar prediksi tidak tertukar.

```text
--8<-- "expected/forecasting/saving_monitoring.txt:1:2"
```

**Hasil.** Prediksi identik; model dilatih sampai 1999-12 dan prediksi dimulai 2000-01. Seperti model lain, hanya muat berkas dari sumber tepercaya dan dengan versi pustaka yang sama.

### Langkah 3: galat acuan dan ambang peringatan

**Konsep.** Pemantauan memerlukan patokan: berapa galat yang normal? Patokan diambil dari MAE *rolling origin* (horizon 12 bulan) **sebelum** data baru, lalu ambang peringatan ditetapkan 2 kali patokan (aturan praktis yang dipilih di sini; sesuaikan dengan biaya keputusan nyata).

```python title="saving_monitoring.py" linenums="28"
--8<-- "forecasting/saving_monitoring.py:28:35"
```

Perulangan melatih ulang SARIMA pada beberapa titik asal di dalam data latih dan menghitung MAE prediksi 12 bulan; `np.mean(ref_maes)` adalah patokan `ref`, dan `threshold = 2 * ref` ambangnya.

```text
--8<-- "expected/forecasting/saving_monitoring.txt:4:4"
```

**Hasil.** MAE acuan 0,51 ppm; ambang peringatan 1,03 ppm.

### Langkah 4: memantau galat dengan jendela bergulir

```python title="saving_monitoring.py" linenums="38"
--8<-- "forecasting/saving_monitoring.py:38:47"
```

| Kode | Penjelasan |
|---|---|
| `err = (actual - forecast).abs()` | Galat mutlak tiap bulan: nilai aktual yang baru masuk dibandingkan dengan prediksi yang dibuat sebelumnya. |
| `err.rolling(6).mean().dropna()` | **MAE 6 bulan bergulir**: rata-rata galat pada 6 bulan terakhir di tiap titik waktu (6 bulan pertama tidak lengkap sehingga dibuang). |
| `flag = roll > threshold` | Penanda `True` bila MAE bergulir melewati ambang. |
| `flag.idxmax()` | Tanggal pertama bernilai `True` (`idxmax` pada deret boolean mengembalikan indeks `True` pertama); hanya dipakai bila `flag.any()`. |

```text
--8<-- "expected/forecasting/saving_monitoring.txt:5:5"
```

**Hasil.** Pada data aktual, MAE 6 bulan terakhir 0,42: di bawah ambang, tidak ada peringatan.

### Langkah 5: simulasi perubahan struktural

```python title="saving_monitoring.py" linenums="49"
--8<-- "forecasting/saving_monitoring.py:49:54"
```

`shifted.loc["2000-07-01":] += 3` menaikkan semua nilai aktual **sejak Juli 2000** sebesar 3 ppm (simulasi buatan), seolah terjadi perubahan struktural yang tidak diketahui model. Bias dihitung sebagai rata-rata (aktual − prediksi).

```text
--8<-- "expected/forecasting/saving_monitoring.txt:6:7"
```

**Interpretasi.** Pada simulasi, MAE 6 bulan terakhir naik ke 2,58 dan ambang terlampaui pertama kali pada 2000-08 (setelah dua bulan bergeser masuk jendela 6 bulan); bias +1,90, dibanding −0,35 pada data aktual (model rata-rata sedikit terlalu tinggi). Tanda ini memicu pemeriksaan data dan pelatihan ulang. Ambang "2 × acuan" adalah aturan praktis; tentukan ambang sesuai biaya keputusan nyata.

??? example "Program lengkap (`saving_monitoring.py`)"
    ```python title="saving_monitoring.py" linenums="1"
    --8<-- "forecasting/saving_monitoring.py"
    ```

## Praktik deployment

| Aspek | Saran |
|---|---|
| Bentuk | Pekerjaan terjadwal (misalnya bulanan) yang memuat model, memprediksi, dan menyimpan hasil |
| Pelatihan ulang | Terjadwal (tiap periode) dan berbasis pemicu (galat melewati ambang) |
| Pencatatan | Versi model, rentang data latih, parameter, tanggal prediksi, galat per periode |
| Keamanan | Hanya muat berkas model tepercaya |
| Pembanding | Selalu laporkan juga seasonal naive |

## Kesalahan umum

!!! warning "Hindari"
    - Memakai model bertahun-tahun tanpa pemantauan galat.
    - Melatih ulang tanpa evaluasi pembanding (model baru bisa lebih buruk).
    - Menetapkan ambang tanpa acuan dari periode pengembangan.
    - Lupa mencatat tanggal akhir data latih.

## Latihan

1. Ubah ambang menjadi 1,5 × acuan. Kapan peringatan pertama muncul pada simulasi?
2. Tambahkan pemantauan relatif terhadap seasonal naive (rasio MAE model / MAE baseline).
3. Rancang aturan pelatihan ulang untuk data bulanan yang berubah struktur (misalnya pandemi).

## Ringkasan

Model forecasting perlu disimpan bersama metadata, dipantau terhadap galat acuan, dan dilatih ulang secara terjadwal atau saat galat melampaui ambang.

Berikutnya: [Proyek Forecasting End-to-End](project.md).
