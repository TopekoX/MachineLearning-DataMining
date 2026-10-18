# Logistic Regression

## Tujuan pembelajaran

- [x] Menjelaskan bagaimana Logistic Regression mengubah skor linear menjadi probabilitas.
- [x] Memahami peran regularisasi dan parameter `C`.
- [x] Melatih, memprediksi, dan menafsirkan koefisien dengan hati-hati.

## Intuisi

Logistic Regression menghitung skor linear dari fitur, lalu memetakannya ke probabilitas 0–1 dengan fungsi sigmoid. Meskipun bernama "regresi", model ini untuk **klasifikasi**.

$$
z = b + w_1 x_1 + \dots + w_p x_p, \qquad
P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

\(x_j\) adalah fitur, \(w_j\) bobot, \(b\) bias. Skor \(z\) positif berarti peluang kelas 1 di atas 0,5. Bobot dipelajari dengan meminimalkan *log loss* ditambah penalti regularisasi (L2 secara bawaan):

$$
\min_{\mathbf{w}, b}\; \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{n} \ell_i
$$

\(\ell_i\) adalah *log loss* sampel ke-\(i\). **`C` kecil = regularisasi kuat** (bobot ditekan, model lebih sederhana); `C` besar = regularisasi lemah.

## Ringkasan model

| Aspek | Keterangan |
|---|---|
| Kelebihan | Cepat, stabil, menghasilkan probabilitas, koefisien dapat ditafsirkan |
| Keterbatasan | Batas keputusan linear; koefisien tidak stabil bila fitur sangat berkorelasi |
| Hyperparameter penting | `C`, `penalty`, `class_weight`, `max_iter` |
| Praproses | *Scaling* sangat dianjurkan |
| Cocok untuk | *Baseline* pertama, kebutuhan probabilitas dan penjelasan |

## Implementasi langkah demi langkah

Program `examples/classification/logistic_regression.py` memakai data Breast Cancer. Langkah memuat data dan membagi data sama dengan [pelajaran sebelumnya](preprocessing-baseline.md#langkah-1-memuat-data-dan-memisahkan-data-uji), jadi tidak dijelaskan ulang.

### Langkah 1: melatih model dan membaca laporan klasifikasi

**Tujuan.** Melatih regresi logistik pada data latih dan menilainya pada data uji.

```python title="logistic_regression.py" linenums="1"
--8<-- "classification/logistic_regression.py:1:18"
```

| Kode | Penjelasan |
|---|---|
| `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` | Pipeline: standardisasi lalu regresi logistik (`max_iter` = batas iterasi optimisasi). |
| `model.fit(X_train, y_train)` | Melatih: scaler menghitung rata-rata/simpangan baku, regresi logistik mempelajari bobot \(w\) dan bias \(b\). Dilakukan **hanya** pada data latih. |
| `model.predict(X_test)` | Memprediksi **kelas** (0 atau 1) untuk data uji. Scaler yang sama (parameter dari data latih) otomatis dipakai dulu. |
| `classification_report(y_test, y_pred, digits=3)` | Tabel metrik per kelas: *precision*, *recall*, *f1-score*, dan *support* (jumlah sampel sebenarnya tiap kelas pada data uji). |

```text
--8<-- "expected/classification/logistic_regression.txt:1:8"
```

**Cara membaca laporan.** Baris `0` dan `1` adalah metrik tiap kelas (0 = ganas, 1 = jinak). `accuracy` adalah proporsi prediksi benar keseluruhan. `macro avg` merata-ratakan metrik antar-kelas dengan bobot sama; `weighted avg` memberi bobot menurut `support`.

**Interpretasi.** *Accuracy* 0,982: dari 114 data uji hanya 2 yang salah. *Precision* dan *recall* kelas ganas (0) 0,976 dan kelas jinak (1) 0,986. Dengan data uji sekecil ini, selisih satu atau dua sampel mengubah angka cukup besar.

### Langkah 2: probabilitas, bukan hanya kelas

**Konsep.** Regresi logistik menghitung skor linear \(z = b + w_1x_1 + \dots\), lalu mengubahnya menjadi probabilitas \(\sigma(z) = 1/(1+e^{-z})\). Contoh: \(z=0\) memberi 0,5; \(z=2\) memberi sekitar 0,88; \(z=-2\) sekitar 0,12. `predict` memilih kelas dengan ambang 0,5, sedangkan `predict_proba` menunjukkan probabilitasnya.

```python title="logistic_regression.py" linenums="20"
--8<-- "classification/logistic_regression.py:20:22"
```

`model.predict_proba(X_test.iloc[:3])` mengembalikan tabel dengan satu kolom per kelas (kolom 0 = P(kelas 0), kolom 1 = P(kelas 1)); `[:, 1]` mengambil semua baris kolom kedua, yaitu probabilitas kelas 1. `X_test.iloc[:3]` memilih tiga baris pertama berdasarkan posisi (lihat [`loc` vs `iloc`](../fundamentals/pandas.md)).

```text
--8<-- "expected/classification/logistic_regression.txt:10:10"
```

**Interpretasi.** Probabilitas jinak untuk tiga sampel adalah ≈ 0, 1, dan 0,006, sesuai label asli (0, 1, 0). Model sangat yakin pada contoh ini. Probabilitas berguna karena ambang 0,5 dapat digeser sesuai biaya kesalahan (lihat [Evaluasi Model](evaluation.md)).

### Langkah 3: membaca koefisien

```python title="logistic_regression.py" linenums="24"
--8<-- "classification/logistic_regression.py:24:29"
```

| Kode | Penjelasan |
|---|---|
| `model[-1]` | Mengambil **langkah terakhir** pipeline (regresi logistik); `model[0]` adalah scaler. |
| `.coef_[0]` | Bobot \(w\) yang **dipelajari** (atribut berakhiran `_` = hasil pelatihan); `[0]` karena klasifikasi biner hanya memiliki satu baris bobot. |
| `pd.Series(..., index=X.columns)` | Memasangkan tiap bobot dengan nama fiturnya. |
| `.sort_values()`, `.head(3)`, `.tail(3)` | Mengurutkan naik; tiga terkecil (paling negatif) dan tiga terbesar. |

```text
--8<-- "expected/classification/logistic_regression.txt:12:19"
```

**Interpretasi.** Karena fitur sudah distandardisasi, besar koefisien dapat dibandingkan. `worst texture` (-1,26), `radius error` (-1,08), dan `worst concave points` (-0,95) mendorong prediksi ke kelas **ganas** (0). Bobot positif mendorong ke kelas jinak (1).

!!! warning "Hati-hati menafsirkan koefisien"
    Koefisien `mean compactness` bertanda **positif** (+0,65), padahal tumor ganas umumnya lebih kompak. Ini efek fitur yang saling berkorelasi (*multikolinearitas*): bobot dibagi di antara fitur serupa dan tanda individual bisa menyesatkan. Tafsirkan koefisien sebagai bagian dari model keseluruhan, bukan sebagai pengaruh sebab-akibat satu fitur.

### Langkah 4: hyperparameter `C` (kekuatan regularisasi)

**Konsep.** `C` adalah **hyperparameter**: ditetapkan sebelum pelatihan, berbeda dari koefisien (parameter yang dipelajari). `C` kecil = regularisasi kuat (bobot ditekan, model sederhana); `C` besar = regularisasi lemah.

```python title="logistic_regression.py" linenums="31"
--8<-- "classification/logistic_regression.py:31:35"
```

`for C in [...]` mengulang untuk tiap nilai; di dalamnya dibuat pipeline baru dengan `C` tersebut, lalu `cross_val_score(..., scoring="f1")` menilainya pada data latih dengan 5 lipatan. Format `{C:<8}` meratakan teks ke kiri selebar 8 karakter.

```text
--8<-- "expected/classification/logistic_regression.txt:21:27"
```

**Interpretasi.** F1 validasi silang 0,912 pada `C=0,001` (terlalu teregularisasi: *underfitting*), naik ke 0,985 pada `C=0,1`, 0,984 pada `C=1`, lalu turun ke 0,965 pada `C=100` (regularisasi terlalu lemah). Nilai terbaik di tengah; `C=0,1` hingga `1` hampir setara. Pemilihan `C` dilakukan dengan **validasi silang pada data latih**, bukan dengan data uji.

??? example "Program lengkap (`logistic_regression.py`)"
    ```python title="logistic_regression.py" linenums="1"
    --8<-- "classification/logistic_regression.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Tidak melakukan *scaling* lalu membandingkan koefisien antar fitur.
    - Menafsirkan koefisien individual pada fitur yang sangat berkorelasi.
    - Menganggap probabilitas model selalu terkalibrasi.
    - Menggunakan ambang 0,5 tanpa mempertimbangkan biaya kesalahan.

## Latihan

1. Coba `penalty="l1"` dengan `solver="liblinear"`. Berapa koefisien yang menjadi nol?
2. Latih hanya dengan fitur `mean`. Bagaimana F1 validasi silangnya?
3. Hitung `predict_proba` untuk seluruh data uji dan temukan sampel dengan probabilitas paling mendekati 0,5.

## Ringkasan

Logistic Regression memodelkan peluang kelas dengan sigmoid dari skor linear. `C` mengatur kekuatan regularisasi. Ia cepat dan menjadi *baseline* yang kuat, tetapi koefisiennya perlu ditafsirkan hati-hati.

Berikutnya: [K-Nearest Neighbors (KNN)](knn.md).
