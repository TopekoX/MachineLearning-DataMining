# Metrik Evaluasi Machine Learning

## Tujuan pembelajaran

- [x] Menghitung dan menafsirkan metrik dasar klasifikasi: *accuracy*, *precision*, *recall*, F1.
- [x] Menghitung metrik regresi: MAE, MSE, RMSE, R².
- [x] Menjelaskan mengapa *accuracy* menyesatkan pada data tidak seimbang.
- [x] Memilih metrik berdasarkan tujuan dan biaya kesalahan.

!!! info "Cakupan"
    Lesson ini memperkenalkan metrik dengan contoh yang dapat dihitung tangan. Pembahasan lebih dalam: kurva ROC/PR dan ambang keputusan di bab [Klasifikasi](../classification/index.md), analisis residual di bab [Estimasi](../estimation/index.md), dan metrik deret waktu di bab [Forecasting](../forecasting/index.md).

## Metrik klasifikasi

Hasil klasifikasi biner dirangkum dalam *confusion matrix*: **TP** (positif diprediksi positif), **FP** (negatif diprediksi positif), **FN** (positif diprediksi negatif), **TN** (negatif diprediksi negatif).

$$
\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad
\text{precision} = \frac{TP}{TP + FP}, \quad
\text{recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

- **Precision**: dari yang diprediksi positif, berapa yang benar-benar positif. Penting bila alarm palsu mahal.
- **Recall**: dari semua yang positif, berapa yang tertangkap. Penting bila melewatkan kasus positif mahal.
- **F1**: rata-rata harmonik precision dan recall; rendah bila salah satunya rendah.

## Metrik regresi

Dengan nilai sebenarnya \(y_i\), prediksi \(\hat{y}_i\), dan \(n\) sampel:

$$
\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|, \quad
\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2, \quad
\text{RMSE} = \sqrt{\text{MSE}}
$$

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

MAE dan RMSE berada pada satuan target; RMSE lebih menghukum galat besar. \(R^2\) membandingkan model dengan prediksi konstan berupa rata-rata \(\bar{y}\): 1 berarti sempurna, 0 sama dengan menebak rata-rata, dan dapat negatif.

## Implementasi langkah demi langkah

Program `examples/fundamentals/metrics_basics.py` memakai contoh kecil yang dapat dihitung tangan. Program lengkap ada di akhir bagian ini.

### Langkah 1: metrik klasifikasi pada contoh kecil

**Tujuan.** Memahami dari mana angka *accuracy*, *precision*, *recall*, dan F1 berasal.

```python title="metrics_basics.py" linenums="1"
--8<-- "fundamentals/metrics_basics.py:1:16"
```

| Kode | Penjelasan |
|---|---|
| `y_true` | Label **sebenarnya** untuk 10 data (4 positif = 1, 6 negatif = 0). |
| `y_pred` | Label **prediksi** model untuk 10 data yang sama. Membandingkan `y_true` dengan `y_pred` adalah dasar semua metrik klasifikasi. |
| `confusion_matrix(y_true, y_pred)` | Tabel 2×2 jumlah prediksi benar/salah per kelas. `.ravel()` meratakannya menjadi empat angka berurutan `tn, fp, fn, tp` yang langsung disimpan ke empat variabel. |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | Fungsi yang menghitung metrik dari `y_true` dan `y_pred`; **urutan argumen**: yang sebenarnya dulu, baru prediksi. |

Menelusuri `y_pred` terhadap `y_true`: 3 positif ditebak positif (TP = 3); 1 negatif ditebak positif (FP = 1); 1 positif ditebak negatif (FN = 1); 5 negatif ditebak negatif (TN = 5).

```text
--8<-- "expected/fundamentals/metrics_basics.txt:1:5"
```

**Interpretasi.** *Accuracy* = (3+5)/10 = 0,80; *precision* = 3/(3+1) = 0,75: dari 4 prediksi positif, 3 benar; *recall* = 3/(3+1) = 0,75: dari 4 data yang sebenarnya positif, 3 tertangkap; F1 = 0,75 (rata-rata harmonik keduanya).

### Langkah 2: jebakan *accuracy* pada data tidak seimbang

**Konsep.** Bila 95% data negatif, model yang selalu menjawab "negatif" tampak akurat padahal tidak berguna.

```python title="metrics_basics.py" linenums="18"
--8<-- "fundamentals/metrics_basics.py:18:23"
```

| Kode | Penjelasan |
|---|---|
| `[0] * 95 + [1] * 5` | Membuat daftar berisi 95 angka 0 diikuti 5 angka 1 (operator `*` mengulang list). |
| `DummyClassifier(strategy="most_frequent")` | *Baseline* yang selalu memprediksi kelas terbanyak. Ia tidak memakai fitur; `np.zeros((100, 1))` hanya fitur kosong sebagai syarat bentuk masukan. |
| `.fit(X, y)` lalu `.predict(X)` | `fit` = "belajar" (di sini hanya mengingat kelas terbanyak); `predict` = menghasilkan prediksi. |

```text
--8<-- "expected/fundamentals/metrics_basics.txt:7:7"
```

**Interpretasi.** *Accuracy* 0,95 tetapi *recall* 0,00: tidak satu pun dari 5 kasus positif tertangkap. Pada data tidak seimbang, jangan menilai hanya dengan *accuracy*.

### Langkah 3: metrik regresi

```python title="metrics_basics.py" linenums="25"
--8<-- "fundamentals/metrics_basics.py:25:32"
```

`y_real` adalah nilai sebenarnya dan `y_hat` prediksi. Galat (nilai sebenarnya dikurangi prediksi) untuk lima titik adalah `-10, 10, -10, 20, -30`. `np.sqrt(mse)` mengakarkan MSE untuk mendapatkan RMSE yang bersatuan sama dengan target.

```text
--8<-- "expected/fundamentals/metrics_basics.txt:9:12"
```

**Interpretasi.** Nilai absolut galat: 10, 10, 10, 20, 30, sehingga MAE = 80/5 = 16,0. MSE = (100+100+100+400+900)/5 = 320,0 dan RMSE ≈ 17,89: lebih besar dari MAE karena galat 30 dihukum kuadratik. \(R^2\) = 1 − 1600/25000 = 0,936: jumlah kuadrat galat model (1600) hanya 6,4% dari jumlah kuadrat galat bila selalu menebak rata-rata (25000).

??? example "Program lengkap (`metrics_basics.py`)"
    ```python title="metrics_basics.py" linenums="1"
    --8<-- "fundamentals/metrics_basics.py"
    ```

## Memilih metrik

| Situasi | Metrik yang disarankan |
|---|---|
| Kelas seimbang, biaya kesalahan setara | *Accuracy*, F1 |
| Data tidak seimbang | Precision, recall, F1, PR-AUC |
| Melewatkan kasus positif sangat mahal (diagnosis, fraud) | Recall (dengan memperhatikan precision) |
| Alarm palsu mahal | Precision |
| Regresi, galat dalam satuan asli | MAE, RMSE |
| Regresi, galat besar harus dihukum berat | RMSE |

## Kesalahan umum

!!! warning "Hindari"
    - Melaporkan *accuracy* saja pada data tidak seimbang.
    - Membandingkan model memakai metrik berbeda atau data uji berbeda.
    - Menganggap \(R^2\) tinggi selalu berarti model baik; periksa juga galat dan residual.
    - Memilih metrik setelah melihat hasil agar model favorit tampak menang.
    - Memakai metrik klasifikasi untuk regresi atau klastering.

## Latihan

1. Ubah `y_pred` sehingga FP=0 dan FN=2. Hitung manual precision dan recall, lalu cek dengan kode.
2. Pada data 95/5, bagaimana *precision* model yang selalu menebak 0? (Perhatikan pembagian dengan nol.)
3. Tambahkan satu prediksi yang meleset 200 pada contoh regresi. Bagaimana perubahan MAE vs RMSE?
4. Untuk deteksi penyakit serius, metrik mana yang Anda prioritaskan, dan mengapa?

## Ringkasan

Pilih metrik sesuai tujuan dan biaya kesalahan. *Accuracy* menyesatkan pada data tidak seimbang; precision, recall, dan F1 memberi gambaran lebih lengkap. Pada regresi, MAE dan RMSE dalam satuan target, dan \(R^2\) membandingkan dengan prediksi rata-rata.

Berikutnya: [Workflow scikit-learn dan Pipeline](sklearn-pipeline.md).
