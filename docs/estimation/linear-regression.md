# Baseline dan Regresi Linear

## Tujuan pembelajaran

- [x] Membuat *baseline* regresi dan membaca metriknya.
- [x] Menjelaskan model dan fungsi tujuan regresi linear (kuadrat terkecil).
- [x] Melatih regresi linear dan menafsirkan koefisiennya dengan hati-hati.

## *Baseline* regresi

*Baseline* paling sederhana selalu memprediksi **rata-rata target data latih**. Model apa pun harus mengalahkannya. Perhatikan: \(R^2\) pada *baseline* ini ≈ 0 secara konstruksi, karena \(R^2\) membandingkan model dengan prediksi rata-rata.

## Regresi linear

Model memprediksi target sebagai kombinasi linear fitur:

$$
\hat{y} = b + w_1 x_1 + w_2 x_2 + \dots + w_p x_p
$$

\(\hat{y}\) prediksi, \(x_j\) fitur, \(w_j\) bobot (koefisien), dan \(b\) intersep. Bobot dipilih untuk meminimalkan jumlah kuadrat residual (*ordinary least squares*, OLS):

$$
\min_{b, \mathbf{w}} \sum_{i=1}^{n} \bigl(y_i - \hat{y}_i\bigr)^2
$$

Solusinya tertutup (persamaan normal): \(\mathbf{\beta} = (A^\top A)^{-1} A^\top \mathbf{y}\), dengan \(A\) matriks fitur yang diberi kolom 1 untuk intersep dan \(\mathbf{\beta}\) berisi intersep dan bobot.

| Aspek | Keterangan |
|---|---|
| Kelebihan | Cepat, dapat ditafsirkan, tanpa hyperparameter, *baseline* yang kuat |
| Keterbatasan | Hanya hubungan linear; sensitif terhadap *outlier* dan multikolinearitas |
| Asumsi | Hubungan linear, residual kira-kira acak, tidak ada multikolinearitas ekstrem |
| Praproses | *Scaling* tidak mengubah prediksi, tetapi memudahkan perbandingan koefisien |

## Implementasi langkah demi langkah

Program `examples/estimation/linear_regression.py` memakai dataset Diabetes. Program lengkap ada di akhir bagian ini.

### Langkah 1: memuat data dan membagi data

**Tujuan.** Menyisihkan data uji sebelum modelling. Berbeda dari klasifikasi, **tidak ada `stratify`** karena targetnya angka kontinu.

```python title="linear_regression.py" linenums="1"
--8<-- "estimation/linear_regression.py:1:12"
```

| Kode | Penjelasan |
|---|---|
| `train_test_split(X, y, test_size=0.2, random_state=SEED)` | 80% latih (353), 20% uji (89); pembagian acak yang dapat diulang berkat `random_state`. Urutan keluaran: `X_train, X_test, y_train, y_test`. |
| `X_train.shape` | `(353, 10)`: 353 sampel × 10 fitur. |

```text
--8<-- "expected/estimation/linear_regression.txt:1:1"
```

### Langkah 2: fungsi bantu untuk melaporkan metrik

```python title="linear_regression.py" linenums="15"
--8<-- "estimation/linear_regression.py:15:18"
```

Fungsi `report(name, model)` menerima nama dan model terlatih, memprediksi data uji (`model.predict(X_test)`), lalu mencetak tiga metrik: **MAE**, **RMSE** (`np.sqrt` dari `mean_squared_error`), dan **R²**. Urutan argumen setiap metrik: nilai **sebenarnya** (`y_test`) dulu, baru **prediksi** (`pred`). Dengan fungsi ini, model-model berikutnya dilaporkan dengan cara yang persis sama. Format `{...:6.2f}` berarti lebar 6 karakter dengan dua desimal.

### Langkah 3: *baseline*, selalu memprediksi rata-rata

**Konsep.** *Baseline* regresi memprediksi **rata-rata target data latih** untuk setiap sampel. Model apa pun harus lebih baik.

```python title="linear_regression.py" linenums="21"
--8<-- "estimation/linear_regression.py:21:23"
```

`DummyRegressor(strategy="mean").fit(X_train, y_train)` menginisialisasi lalu "melatih" (mencatat rata-rata `y_train`).

```text
--8<-- "expected/estimation/linear_regression.txt:2:2"
```

**Interpretasi.** MAE 64,01, RMSE 73,22, R² -0,012. R² hampir nol memang diharapkan (R² membandingkan model dengan prediksi rata-rata); sedikit negatif karena rata-rata data latih tidak persis sama dengan rata-rata data uji.

### Langkah 4: regresi linear

**Konsep.** Model memprediksi \(\hat{y} = b + w_1x_1 + \dots + w_{10}x_{10}\) dengan bobot yang meminimalkan jumlah kuadrat galat pada data latih.

```python title="linear_regression.py" linenums="25"
--8<-- "estimation/linear_regression.py:25:29"
```

| Kode | Penjelasan |
|---|---|
| `LinearRegression().fit(X_train, y_train)` | Melatih (mencari \(b\) dan \(w_j\)). Tidak ada hyperparameter penting. |
| `-cross_val_score(..., scoring="neg_root_mean_squared_error")` | Scikit-learn mengasumsikan "skor lebih besar lebih baik", sehingga galat dinyatakan **negatif**. Tanda minus di depan mengembalikannya menjadi RMSE positif. |
| `cv=5` | Validasi silang 5 lipatan (pada regresi tanpa pengacakan: lipatan berurutan menurut baris). |

```text
--8<-- "expected/estimation/linear_regression.txt:3:4"
```

**Interpretasi.** Pada data uji MAE 42,79, RMSE 53,85, R² 0,453. Galat turun jelas dibanding *baseline*, dan model menjelaskan sekitar 45% variasi target relatif terhadap prediksi rata-rata. RMSE 53,85 masih besar dibanding simpangan baku target (77,1): prediksinya cukup berguna, bukan akurat. Validasi silang memberi RMSE 55,97 (± 3,18), sedikit di atas hasil uji; gunakan rentang ini, bukan satu angka, untuk mengira performa pada data baru.

### Langkah 5: membaca koefisien

```python title="linear_regression.py" linenums="31"
--8<-- "estimation/linear_regression.py:31:34"
```

`lin.coef_` adalah bobot \(w_j\) hasil pelatihan (berakhiran `_`), dan `lin.intercept_` adalah \(b\). `pd.Series(..., index=X.columns)` memasangkan bobot dengan nama fitur.

```text
--8<-- "expected/estimation/linear_regression.txt:6:16"
```

**Interpretasi.** Fitur satu skala sehingga besar koefisien dapat dibandingkan: `s5` (736,2) dan `bmi` (542,4) paling positif, selaras dengan korelasinya terhadap target. Namun `s1` (-931,5) dan `s2` (+518,1) besar dan berlawanan tanda: tanda **multikolinearitas** (`s1`–`s2` berkorelasi 0,90). Model membagi pengaruh keduanya secara tidak stabil, sehingga koefisien individualnya tidak boleh diartikan sebagai efek tiap fitur. `sex` berkoefisien -242 meski korelasi marginalnya hampir 0: koefisien menunjukkan efek **dengan fitur lain tetap**, bukan hubungan sederhana.

### Langkah 6: pemeriksaan dengan persamaan normal

**Konsep.** Solusi kuadrat terkecil dapat dihitung langsung: \(\mathbf{\beta} = (A^\top A)^{-1} A^\top \mathbf{y}\), dengan \(A\) matriks fitur yang diberi kolom 1 untuk intersep. Langkah ini membuktikan bahwa `LinearRegression` benar-benar menyelesaikan masalah kuadrat terkecil.

```python title="linear_regression.py" linenums="36"
--8<-- "estimation/linear_regression.py:36:39"
```

| Kode | Penjelasan |
|---|---|
| `np.column_stack([np.ones(n), X])` | Menambahkan kolom berisi 1 di kiri `X` untuk intersep. |
| `A.T @ A`, `A.T @ y` | `.T` = transpos; `@` = perkalian matriks. |
| `np.linalg.solve(M, v)` | Menyelesaikan sistem persamaan linear \(M\mathbf{\beta} = v\). |
| `beta[1:]` | Koefisien (mengabaikan elemen pertama, yaitu intersep). |

```text
--8<-- "expected/estimation/linear_regression.txt:18:18"
```

**Hasil.** Selisih maksimum hanya sekitar \(10^{-11}\) (galat pembulatan): hasilnya sama dengan scikit-learn.

??? example "Program lengkap (`linear_regression.py`)"
    ```python title="linear_regression.py" linenums="1"
    --8<-- "estimation/linear_regression.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Menafsirkan koefisien individual pada fitur yang berkorelasi tinggi.
    - Menganggap R² tinggi pasti berarti model baik, atau R² rendah berarti tidak berguna.
    - Mengekstrapolasi jauh di luar rentang data latih.
    - Melewatkan *baseline* sehingga tidak tahu seberapa jauh model lebih baik dari rata-rata.

## Latihan

1. Hapus `s2` dari fitur dan latih ulang. Bagaimana koefisien `s1` berubah?
2. Hitung RMSE CV dengan `KFold(shuffle=True)`. Apakah mirip 55,97?
3. Tambahkan fitur `bmi**2`. Apakah RMSE CV membaik?

## Ringkasan

Regresi linear meminimalkan jumlah kuadrat galat dan menjadi *baseline* yang kuat dan mudah dibaca. Bandingkan dengan *baseline* rata-rata, waspadai multikolinearitas, dan tafsirkan koefisien dengan hati-hati.

Berikutnya: [Ridge dan Lasso](ridge-lasso.md).
