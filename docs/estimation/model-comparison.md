# Perbandingan Model Regresi

## Tujuan pembelajaran

- [x] Membandingkan model regresi dengan lipatan dan metrik yang sama.
- [x] Menafsirkan MAE, RMSE, dan R² bersama sebaran antar-lipatan.
- [x] Memilih model dengan mempertimbangkan akurasi, stabilitas, dan kebutuhan interpretasi.

## Ringkasan sifat model

| Aspek | Linear | Ridge / Lasso | KNN | Decision Tree | Random Forest |
|---|---|---|---|---|---|
| Bentuk hubungan | Linear | Linear (teregularisasi) | Lokal | Tangga | Tangga (rata-rata) |
| Perlu *scaling* | Disarankan | **Wajib** | **Wajib** | Tidak | Tidak |
| Ekstrapolasi | Ya (linear) | Ya | Tidak | Tidak | Tidak |
| Interpretabilitas | Tinggi | Tinggi | Rendah | Tinggi (kecil) | Rendah |
| Risiko *overfitting* | Naik dengan jumlah fitur | Dikendalikan `alpha` | Tinggi bila `k` kecil | Tinggi bila dalam | Sedang |
| Hyperparameter kunci | — | `alpha` | `n_neighbors` | `max_depth`, `min_samples_leaf` | `n_estimators`, `min_samples_leaf` |

Sifat ini bersifat umum; performa sebenarnya bergantung pada data.

## Prosedur

Data uji dipisahkan lebih dulu. Semua model dievaluasi pada data latih dengan `RepeatedKFold` (5 lipatan × 3 pengulangan = 15 skor per model) dan metrik yang sama (MAE, RMSE, R²). Data uji dipakai sekali untuk model terpilih.

## Implementasi langkah demi langkah

Dataset Diabetes; pemuatan dan pembagian data sama dengan [pelajaran regresi linear](linear-regression.md#langkah-1-memuat-data-dan-membagi-data). Program `examples/estimation/model_comparison.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: mendefinisikan kandidat model

**Konsep.** Semua model disimpan dalam kamus (`nama → model`) agar dapat dinilai dengan perulangan yang sama. Model yang sensitif pada skala (linear teregularisasi, KNN) dimasukkan ke `Pipeline` dengan `StandardScaler`; pohon dan hutan tidak memerlukannya.

```python title="model_comparison.py" linenums="14"
--8<-- "estimation/model_comparison.py:14:25"
```

Daftar ini mencakup *baseline* rata-rata, tiga model linear, dan tiga model non-linear. Nilai hyperparameter (`alpha`, `k`, `max_depth`) dipilih dari pelajaran sebelumnya; semua model baru **diinisialisasi**, belum dilatih.

### Langkah 2: validasi silang berulang dan beberapa metrik

```python title="model_comparison.py" linenums="27"
--8<-- "estimation/model_comparison.py:27:29"
```

| Kode | Penjelasan |
|---|---|
| `RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)` | Validasi silang 5 lipatan diulang 3 kali dengan pengacakan berbeda = 15 skor per model; **objek `cv` yang sama** dipakai untuk semua model sehingga lipatannya identik. |
| `scoring = {...}` | Kamus tiga metrik: `neg_mean_absolute_error` (MAE negatif), `neg_root_mean_squared_error` (RMSE negatif), dan `r2`. |

### Langkah 3: menjalankan perbandingan

```python title="model_comparison.py" linenums="31"
--8<-- "estimation/model_comparison.py:31:37"
```

`cross_validate` mengembalikan kamus skor per lipatan dengan kunci `test_mae`, `test_rmse`, `test_r2`. Tanda minus pada `-s["test_mae"]` mengembalikan skor negatif menjadi galat positif. `Std RMSE` adalah simpangan baku RMSE antar 15 lipatan, yaitu ukuran kestabilan.

```text
--8<-- "expected/estimation/model_comparison.txt:1:9"
```

**Interpretasi.**

- **Semua model mengalahkan *baseline*** (RMSE 78,31; R² -0,04).
- **Keluarga linear imbang**: Linear (55,23), Ridge (55,30), dan Lasso (55,29) berbeda jauh lebih kecil daripada simpangan baku antar-lipatan (3,4–3,5). Regularisasi ringan tidak mengubah apa-apa pada 10 fitur ini.
- **KNN (59,04) dan Random Forest (58,48)** sedikit di bawah model linear; **Decision Tree** kedalaman 3 paling lemah (RMSE 65,43; R² 0,26): dengan 8 daun, prediksinya hanya 8 nilai berbeda, kasar untuk target kontinu, sedangkan pohon yang lebih dalam bervarians tinggi.
- **Bacaan bersama metrik**: MAE sekitar 45 berarti galat rata-rata 45 satuan target (yang simpangan bakunya 77); RMSE selalu ≥ MAE, dan selisih besar menandakan adanya galat besar yang sesekali.

### Langkah 4: memilih model dan menguji sekali

```python title="model_comparison.py" linenums="39"
--8<-- "estimation/model_comparison.py:39:44"
```

`results["RMSE"].idxmin()` mengembalikan **nama** model dengan RMSE terendah; model itu dilatih pada seluruh data latih dan diuji pada data uji **sekali saja**.

```text
--8<-- "expected/estimation/model_comparison.txt:11:12"
```

**Interpretasi.** Linear Regression terpilih; pada data uji MAE 42,79, RMSE 53,85, R² 0,453. Karena Ridge dan Lasso nyaris sama, memilih yang paling sederhana pun masuk akal. **Mengapa model linear cukup baik di sini?** Kemungkinan hubungan fitur–target pada dataset ini mendekati linear dengan derau besar, dan 353 data latih terlalu sedikit bagi model fleksibel untuk memanfaatkan non-linearitas. Ini kesimpulan untuk dataset ini; bukan aturan umum.

??? example "Program lengkap (`model_comparison.py`)"
    ```python title="model_comparison.py" linenums="1"
    --8<-- "estimation/model_comparison.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Membandingkan model dengan pembagian data atau metrik berbeda.
    - Mengabaikan sebaran antar-lipatan (Std RMSE) saat selisih kecil.
    - Menyimpulkan model fleksibel selalu lebih baik.
    - Memilih model berdasarkan data uji berulang.

## Latihan

1. Tambahkan `RandomForestRegressor(min_samples_leaf=10)` dan `Ridge(alpha=100)` ke daftar. Apakah kesimpulan berubah?
2. Ganti `SEED`. Apakah urutan model dengan RMSE terendah tetap sama?
3. Bandingkan hasil ketika target diubah menjadi `log(target)` untuk semua model (gunakan `TransformedTargetRegressor`).

## Ringkasan

Bandingkan model dengan lipatan dan metrik yang sama; baca selisih skor relatif terhadap sebarannya. Pada data ini keluarga linear sudah memadai; model fleksibel belum tentu lebih baik.

Berikutnya: [Metrik Regresi dan Analisis Residual](metrics-residuals.md).
