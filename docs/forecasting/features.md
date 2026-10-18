# Fitur Waktu dan Lag

## Tujuan pembelajaran

- [x] Mengubah deret waktu menjadi tabel fitur–target untuk model regresi.
- [x] Membuat fitur lag, statistik bergulir, dan fitur kalender tanpa kebocoran.
- [x] Mengenali dan menghindari kebocoran dari masa depan.

## Konsep

Model regresi tabular (Ridge, Random Forest, dll.) dapat meramalkan deret waktu bila deret diubah menjadi tabel:

| Fitur | Contoh | Penjelasan |
|---|---|---|
| Lag | `lag_1`, `lag_12` | Nilai 1 dan 12 bulan lalu |
| Statistik bergulir | rata-rata 3 bulan terakhir | **Harus digeser** agar tidak memuat nilai saat ini |
| Kalender | `sin`/`cos` bulan, hari dalam minggu, hari libur | Menangkap musiman tanpa 12 kolom *dummy*; sin/cos menjaga kedekatan Desember–Januari |
| Selisih | `y_t - y_{t-1}` | Target yang lebih stasioner daripada level |

Aturan emas: **fitur untuk waktu \(t\) hanya boleh memakai informasi yang sudah diketahui sebelum \(t\)** (atau yang benar-benar tersedia saat prediksi dibuat, misalnya kalender yang sudah pasti).

$$
\text{contoh baris}: \quad \Delta y_t \;\leftarrow\; \bigl[\Delta y_{t-1},\; \Delta y_{t-2},\; \Delta y_{t-3},\; \Delta y_{t-12},\; \text{rata-rata}(\Delta y_{t-1..t-3}),\; \sin, \cos\bigr], \qquad \Delta y_t = y_t - y_{t-1}
$$

## Implementasi langkah demi langkah

Target adalah **perubahan CO2 bulanan** (`dy`), agar tren tidak mendominasi. Bagian kedua sengaja membuat fitur yang **bocor** untuk memperlihatkan bahayanya. Program `examples/forecasting/lag_features.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: membuat target selisih

```python title="lag_features.py" linenums="1"
--8<-- "forecasting/lag_features.py:1:9"
```

`load_co2_monthly()` (dari `forecast_utils.py`, lihat [Data dan EDA](data-eda.md)) menghasilkan CO2 bulanan. `y.diff()` menghitung selisih tiap bulan dengan bulan sebelumnya: perubahan bulanan `dy`. Prediksi selisih lebih mudah daripada prediksi level yang terus naik.

### Langkah 2: fitur yang hanya memakai masa lalu

**Tujuan.** Membuat tabel fitur untuk bulan \(t\) dari informasi yang sudah diketahui **sebelum** \(t\).

```python title="lag_features.py" linenums="11"
--8<-- "forecasting/lag_features.py:11:22"
```

| Kode | Penjelasan |
|---|---|
| `pd.DataFrame({"dy": dy})` | Tabel dengan kolom target `dy`. |
| `dy.shift(k)` | **Menggeser** deret `k` baris ke bawah: baris bulan \(t\) kini berisi nilai bulan \(t-k\). Itulah fitur **lag**. |
| `dy.shift(1).rolling(3).mean()` | Geser **dulu** 1 bulan, baru hitung rata-rata 3 bulan (`rolling(3)`). Hasilnya rata-rata tiga bulan **sebelum** \(t\). Urutan ini yang mencegah kebocoran. |
| `np.sin/cos(2*np.pi*bulan/12)` | Fitur kalender: bulan diubah menjadi dua fitur lingkaran sehingga Desember dan Januari berdekatan. |
| `feat.dropna()` | Membuang baris awal yang kosong karena belum ada nilai lag (bukan diisi dari masa depan). |

```text
--8<-- "expected/forecasting/lag_features.txt:1:5"
```

**Hasil.** Tujuh fitur jujur dibuat. Pembuatan lag menyisakan 513 dari 526 baris karena `dy` kehilangan 1 baris awal dan lag 12 kehilangan 12 baris. Baris pertama yang tersisa (1959-04) menunjukkan `lag_1` = nilai `dy` bulan sebelumnya (0,03), dan seterusnya.

### Langkah 3: fitur yang bocor

**Konsep.** Statistik bergulir tanpa `shift` ikut memakai nilai bulan \(t\) sendiri, yaitu target yang ingin diprediksi.

```python title="lag_features.py" linenums="24"
--8<-- "forecasting/lag_features.py:24:26"
```

`dy.rolling(3).mean()` tanpa `shift` menghitung rata-rata \(\Delta y_t\), \(\Delta y_{t-1}\), \(\Delta y_{t-2}\): **memuat target saat ini**. `.reindex(feat.index)` menyelaraskannya dengan baris yang tersisa. `honest` adalah daftar kolom fitur jujur.

### Langkah 4: pembagian kronologis dan membandingkan

```python title="lag_features.py" linenums="28"
--8<-- "forecasting/lag_features.py:28:43"
```

| Kode | Penjelasan |
|---|---|
| `feat.iloc[:-24]`, `feat.iloc[-24:]` | Pembagian **kronologis**: 24 baris terakhir (24 bulan terakhir) menjadi data uji; sisanya data latih. Tidak diacak. |
| `def fit_eval(columns)` | Melatih regresi linear pada kolom tertentu dan mengembalikan RMSE pada data uji. |
| `test["dy"].std()` | Simpangan baku target uji: acuan tebakan "rata-rata". |
| `.corr()` | Korelasi tiap fitur dengan target. |

```text
--8<-- "expected/forecasting/lag_features.txt:7:12"
```

**Interpretasi.** Fitur jujur memberi RMSE 0,352 (dibanding simpangan baku target uji 1,213) dengan korelasi fitur–target 0,30. Fitur bocor menghasilkan **RMSE 0,000** dan korelasi 0,78: karena rata-rata tiga nilai dikalikan 3 dikurangi dua lag sama dengan \(\Delta y_t\), regresi linear menyelesaikannya secara eksak, padahal tidak ada kemampuan meramal sama sekali. Skor yang terlalu bagus (galat ≈ 0) hampir selalu tanda kebocoran, bukan keberhasilan.

!!! warning "Fitur yang baru diketahui kemudian"
    Selain statistik bergulir, kebocoran sering muncul dari fitur eksternal yang nilainya baru diketahui setelah \(t\) (misalnya cuaca aktual, jumlah promosi yang belum direncanakan). Jika fitur eksternal dipakai, nilainya harus tersedia (atau dapat diramal) pada saat prediksi.

??? example "Program lengkap (`lag_features.py`)"
    ```python title="lag_features.py" linenums="1"
    --8<-- "forecasting/lag_features.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Menghitung statistik bergulir tanpa `shift`.
    - Menormalkan atau menstandardisasi menggunakan statistik seluruh deret (termasuk masa depan); hitung dari data latih.
    - Mengisi nilai awal yang kosong akibat lag dengan nilai dari masa depan.
    - Memakai bulan sebagai angka 1–12 pada model linear tanpa menyadari Desember dan Januari "berjauhan".

## Latihan

1. Tambahkan `lag_24` dan `roll_std_6` (digeser). Apakah RMSE jujur membaik?
2. Ganti target menjadi level `y` (bukan selisih). Bagaimana RMSE model linear berubah, dan mengapa?
3. Buat fitur "bulan Desember" biner. Bandingkan dengan sin/cos.

## Ringkasan

Fitur lag, statistik bergulir (yang digeser), dan fitur kalender mengubah deret waktu menjadi masalah regresi. Pastikan setiap fitur hanya memakai informasi yang sudah diketahui pada saat prediksi.

Berikutnya: [Validasi Kronologis dan Metrik Forecasting](splitting-evaluation.md).
