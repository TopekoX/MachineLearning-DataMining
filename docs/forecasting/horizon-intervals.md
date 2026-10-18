# Horizon dan Interval Prediksi

## Tujuan pembelajaran

- [x] Menjelaskan bagaimana galat bergantung pada *horizon* prediksi.
- [x] Mengevaluasi model dengan banyak titik asal (*rolling origin*).
- [x] Mengukur cakupan empiris interval prediksi.

## Horizon

*Horizon* \(h\) adalah berapa langkah ke depan yang diprediksi. Semakin jauh, semakin tidak pasti. Kebutuhan bisnis menentukan horizon (stok bulan depan vs rencana tahun depan), dan model dapat unggul pada satu horizon namun tidak pada yang lain. Karena itu galat harus dilaporkan **per horizon**, bukan satu angka rata-rata saja.

## Implementasi langkah demi langkah

Delapan titik asal (data latih berakhir 1993-12, 1994-12, ..., 2000-12), horizon 12 bulan, tiga metode: seasonal naive, Holt-Winters aditif, dan SARIMA(1,1,1)(0,1,1,12). Tiap prediksi hanya memakai data sampai titik asalnya. Program `examples/forecasting/horizon_intervals.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: menentukan titik asal

```python title="horizon_intervals.py" linenums="12"
--8<-- "forecasting/horizon_intervals.py:12:17"
```

`origins` adalah daftar jumlah data latih untuk tiap titik asal: `len(y) - H - 12 * k` dengan `k` dari 7 turun ke 0, sehingga titik asal pertama paling awal dan yang terakhir menyisakan tepat 12 bulan untuk dinilai. `y.index[origins[0] - 1]` adalah tanggal terakhir data latih pada titik asal pertama.

```text
--8<-- "expected/forecasting/horizon_intervals.txt:1:1"
```

**Hasil.** Titik asal berentang dari 1993-12 sampai 2000-12, sebanyak 8.

### Langkah 2: prediksi dari tiap titik asal

```python title="horizon_intervals.py" linenums="19"
--8<-- "forecasting/horizon_intervals.py:19:31"
```

| Kode | Penjelasan |
|---|---|
| `train, actual = y.iloc[:end], y.iloc[end:end + H]` | Latih sampai `end`; nilai sebenarnya 12 bulan berikutnya. |
| `abs_err[...].append(np.abs(actual - prediksi))` | Menyimpan **galat mutlak per langkah** (array 12 angka) untuk tiap metode. |
| `.get_forecast(H)` pada SARIMA | Prediksi dengan interval; `fc.predicted_mean` titik, `fc.conf_int(alpha=0.05)` interval 95%. |
| `covered.append((actual >= bawah) & (actual <= atas))` | Array `True`/`False` per langkah: apakah nilai aktual berada di dalam interval. |
| `width.append(atas - bawah)` | Lebar interval per langkah. |

### Langkah 3: galat menurut horizon

```python title="horizon_intervals.py" linenums="33"
--8<-- "forecasting/horizon_intervals.py:33:38"
```

`np.mean(v, axis=0)` merata-ratakan galat **atas titik asal** (tiap langkah horizon dirata-ratakan sendiri), menghasilkan 12 angka per metode. Tabel ditampilkan hanya pada horizon 1, 3, 6, 9, 12 (`table.loc[[...]]`).

```text
--8<-- "expected/forecasting/horizon_intervals.txt:3:11"
```

**Interpretasi.**

- **Seasonal naive**: galat hampir konstan terhadap horizon (1,69 pada bulan ke-1; 1,78 pada bulan ke-12), karena ia tidak memanfaatkan data terbaru, hanya menyalin tahun lalu; galatnya lebih mencerminkan tren setahun yang terlewat.
- **Holt-Winters dan SARIMA**: galat naik seiring horizon, dari 0,25 pada bulan ke-1 menjadi 0,68 (Holt-Winters) dan 0,57 (SARIMA) pada bulan ke-12. Rata-rata semua horizon: Holt-Winters 0,48; SARIMA 0,46; seasonal naive 1,73. Kedua metode terbaik sangat dekat; pada horizon pendek identik (0,25).

### Langkah 4: lebar dan cakupan interval

```python title="horizon_intervals.py" linenums="40"
--8<-- "forecasting/horizon_intervals.py:40:45"
```

`np.mean(covered, axis=0)` memberi, untuk tiap langkah horizon, proporsi titik asal yang nilai aktualnya masuk interval. `cov[:4].mean()` merata-ratakan horizon 1–4, dan seterusnya. Cakupan empiris dibandingkan dengan target nominal 95%.

```text
--8<-- "expected/forecasting/horizon_intervals.txt:13:14"
```

**Interpretasi.**

- **Interval 95% SARIMA** melebar dari 1,20 ppm (bulan ke-1) ke 2,47 ppm (bulan ke-12).
- **Cakupan empiris** (nominal 95%): 1,00 pada horizon 1–4, 0,91 pada 5–8, 0,88 pada 9–12, dan 0,93 keseluruhan (8 titik asal × 12 langkah = 96 prediksi). Pada horizon pendek intervalnya konservatif, pada horizon panjang agak terlalu sempit. Perhatikan bahwa 96 prediksi **tidak independen** (prediksi berurutan dari satu titik asal berkorelasi), sehingga ketidakpastian pada angka cakupan ini lebih besar daripada yang tampak.
- **Bandingkan dengan lesson sebelumnya**: cakupan 54% pada satu titik asal dengan horizon 24 bulan jauh lebih rendah daripada 93% di sini (horizon 12, delapan asal). Dua temuan itu tidak bertentangan: horizon yang lebih panjang dan satu asal saja memang lebih sulit. Intinya, cakupan perlu dievaluasi pada horizon dan titik asal yang sesuai dengan kegunaan.

??? example "Program lengkap (`horizon_intervals.py`)"
    ```python title="horizon_intervals.py" linenums="1"
    --8<-- "forecasting/horizon_intervals.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Melaporkan satu MAE tanpa menyebut horizonnya.
    - Mengira interval 95% otomatis mencakup 95% kenyataan.
    - Mengevaluasi horizon panjang dengan data yang terlalu pendek.
    - Memakai prediksi 1-langkah untuk menilai kebutuhan peramalan 12 bulan.

## Latihan

1. Ubah `H` menjadi 6 dan 24. Bagaimana tabel galat dan cakupan berubah?
2. Ganti interval menjadi 80% (`alpha=0.2`) dan hitung cakupannya.
3. Tambahkan jumlah titik asal menjadi 12. Apakah kesimpulan Holt-Winters vs SARIMA berubah?

## Ringkasan

Galat tumbuh dengan horizon, jadi laporkan per horizon. Gunakan banyak titik asal dan periksa cakupan interval prediksi secara empiris sebelum memakainya untuk keputusan.

Berikutnya: [Perbandingan dan Optimasi Model](comparison-optimization.md).
