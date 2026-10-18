# Ridge dan Lasso

## Tujuan pembelajaran

- [x] Menjelaskan regularisasi L2 (Ridge) dan L1 (Lasso).
- [x] Menunjukkan bagaimana regularisasi mengendalikan *overfitting* saat fitur banyak.
- [x] Menjelaskan mengapa Lasso dapat menghasilkan koefisien nol (seleksi fitur).

## Konsep

Model dengan banyak fitur mudah *overfitting*: koefisien membesar untuk mengikuti derau. **Regularisasi** menambahkan penalti pada besar koefisien:

$$
\text{Ridge: } \min \sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j w_j^2
\qquad
\text{Lasso: } \min \sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j |w_j|
$$

\(\alpha \ge 0\) mengatur kekuatan penalti: `alpha` = 0 sama dengan regresi linear biasa; `alpha` besar menekan koefisien kuat (risiko *underfitting*).

- **Ridge (L2)** mengecilkan semua koefisien ke arah nol, jarang tepat nol; bagus untuk fitur berkorelasi.
- **Lasso (L1)** dapat membuat sebagian koefisien **tepat nol**, sehingga sekaligus memilih fitur.

Karena penalti bergantung pada besar koefisien, fitur **harus berskala sama** (gunakan `StandardScaler`).

| Aspek | Ridge | Lasso |
|---|---|---|
| Efek | Menyusutkan semua koefisien | Menyusutkan dan menolkan sebagian |
| Hyperparameter | `alpha` | `alpha` |
| Cocok untuk | Fitur berkorelasi, banyak fitur kecil berpengaruh | Seleksi fitur, model ringkas |
| Keterbatasan | Tidak memilih fitur | Tidak stabil pada fitur berkorelasi tinggi |

## Eksperimen langkah demi langkah

Untuk melihat dampak regularisasi, fitur diperbanyak dengan `PolynomialFeatures(degree=2)`: 10 fitur Diabetes menjadi **65** fitur (asli, kuadrat, dan interaksi) dari hanya 353 data latih. Program `examples/estimation/ridge_lasso.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: fungsi bantu untuk menilai model

**Tujuan.** Menilai banyak model dengan cara yang sama: RMSE rata-rata validasi silang pada data latih.

```python title="ridge_lasso.py" linenums="9"
--8<-- "estimation/ridge_lasso.py:9:16"
```

| Kode | Penjelasan |
|---|---|
| `KFold(5, shuffle=True, random_state=SEED)` | Validasi silang 5 lipatan dengan pengacakan tetap, sehingga semua model dinilai pada lipatan yang sama. |
| `def cv_rmse(model): ...` | Fungsi yang menerima sebuah model dan mengembalikan RMSE validasi silang. Tanda minus mengembalikan skor negatif scikit-learn menjadi RMSE positif. |

### Langkah 2: fitur polinomial membuat model terlalu rumit

**Konsep.** `PolynomialFeatures(degree=2)` menambah kuadrat tiap fitur dan hasil kali antar-fitur: dari 10 fitur menjadi 65. Fitur sebanyak itu pada 353 sampel mudah menyebabkan *overfitting* pada model linear.

```python title="ridge_lasso.py" linenums="19"
--8<-- "estimation/ridge_lasso.py:19:24"
```

| Kode | Penjelasan |
|---|---|
| `poly = lambda est: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), est)` | `lambda` adalah fungsi singkat satu baris. `poly(Ridge(alpha=10))` membangun pipeline: buat fitur polinomial, standarkan, lalu model `est`. Standardisasi penting karena penalti Ridge/Lasso bergantung pada besar koefisien. |
| `poly(LinearRegression()).fit(X_train, y_train)[0].n_output_features_` | Melatih pipeline lalu mengambil langkah pertama (`[0]` = `PolynomialFeatures`) dan membaca jumlah fitur keluarannya. |

```text
--8<-- "expected/estimation/ridge_lasso.txt:1:4"
```

**Interpretasi.** Regresi linear dengan 10 fitur asli memberi RMSE CV 55,39, sedangkan dengan 65 fitur polinomial **memburuk** ke 62,50: fitur tambahan menambah derau.

### Langkah 3: Ridge (penalti L2)

**Konsep.** Ridge menambahkan penalti \(\alpha\sum w_j^2\) pada fungsi tujuan sehingga koefisien menyusut. `alpha` = 0 sama dengan regresi linear biasa; `alpha` besar menekan koefisien kuat (risiko *underfitting*).

```python title="ridge_lasso.py" linenums="26"
--8<-- "estimation/ridge_lasso.py:26:29"
```

Perulangan mencoba enam nilai `alpha` pada skala logaritmik (0,1 sampai 1000); skala semacam ini lazim untuk `alpha` karena efeknya multiplikatif.

```text
--8<-- "expected/estimation/ridge_lasso.txt:6:12"
```

**Interpretasi.** RMSE CV turun dari 62,03 (`alpha=0,1`) ke 56,92 (`alpha=100`), lalu naik lagi ke 58,61 (`alpha=300`) dan 63,73 (`alpha=1000`): terlalu kuat menyebabkan *underfitting*. Nilai terbaik berada di tengah.

### Langkah 4: Lasso (penalti L1)

**Konsep.** Lasso memakai penalti \(\alpha\sum|w_j|\), yang dapat membuat sebagian koefisien **tepat nol**; ia sekaligus memilih fitur.

```python title="ridge_lasso.py" linenums="31"
--8<-- "estimation/ridge_lasso.py:31:36"
```

| Kode | Penjelasan |
|---|---|
| `Lasso(alpha=a, max_iter=50000)` | `max_iter` dinaikkan agar optimisasi sempat konvergen pada 65 fitur. |
| `model[-1].coef_` | Koefisien milik langkah terakhir pipeline (Lasso). |
| `(model[-1].coef_ != 0).sum()` | Menghitung berapa koefisien yang **tidak nol** (`!=` menghasilkan `True`/`False`; `.sum()` menjumlahkan `True`). |

```text
--8<-- "expected/estimation/ridge_lasso.txt:14:20"
```

**Interpretasi.** Pada `alpha=0,01` hampir semua koefisien aktif (62 dari 65) dan RMSE 62,19; makin besar `alpha`, makin sedikit koefisien tidak nol: 36 pada `alpha=1` (RMSE 57,13), 16 pada `alpha=3` (RMSE 55,99), dan 4 pada `alpha=10` (RMSE 57,75). Pada `alpha=3` model sudah jauh lebih ringkas dengan galat terbaik.

**Perspektif yang jujur.** Skor terbaik regularisasi (Lasso 55,99; Ridge 56,92) **tidak lebih baik** daripada regresi linear biasa dengan 10 fitur (55,39). Regularisasi menyelamatkan model yang terlalu rumit, tetapi di sini fitur polinomial memang tidak menambah informasi berguna.

### Langkah 5: evaluasi pada data uji

```python title="ridge_lasso.py" linenums="38"
--8<-- "estimation/ridge_lasso.py:38:45"
```

Kamus `{nama: model}` dipakai agar tiga model dilatih dan diuji dengan perulangan yang sama. `model.fit(X_train, y_train).predict(X_test)` melatih lalu memprediksi, dan RMSE dihitung dari prediksi itu.

```text
--8<-- "expected/estimation/ridge_lasso.txt:21:23"
```

**Interpretasi.** RMSE uji 55,64 (linear 65 fitur), 51,57 (Ridge `alpha=100`), dan 50,63 (Lasso `alpha=1`). Urutannya sejalan dengan validasi silang, tetapi perbedaan kecil pada 89 data uji tidak boleh dipakai untuk memilih ulang `alpha`.

??? example "Program lengkap (`ridge_lasso.py`)"
    ```python title="ridge_lasso.py" linenums="1"
    --8<-- "estimation/ridge_lasso.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai Ridge atau Lasso tanpa *scaling*.
    - Menyetel `alpha` dengan data uji.
    - Menganggap Lasso selalu memilih fitur "benar"; pada fitur berkorelasi, pilihannya bisa sembarang.
    - Memakai `alpha` bawaan tanpa menelusuri beberapa nilai (skala logaritmik).

## Latihan

1. Gunakan `RidgeCV(alphas=...)` dan `LassoCV` untuk memilih `alpha` otomatis. Samakah dengan hasil tabel?
2. Coba `ElasticNet` (gabungan L1 dan L2).
3. Pada Lasso `alpha=3`, tampilkan nama fitur yang koefisiennya tidak nol.

## Ringkasan

Ridge dan Lasso menambahkan penalti pada koefisien untuk mengurangi *overfitting*; Lasso juga memilih fitur. Wajib *scaling*, dan `alpha` harus disetel dengan validasi silang.

Berikutnya: [KNN, Decision Tree, dan Random Forest Regressor](nonlinear-regressors.md).
