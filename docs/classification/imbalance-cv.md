# Kelas Tidak Seimbang dan Validasi Silang

## Tujuan pembelajaran

- [x] Mengenali masalah kelas tidak seimbang (*class imbalance*) dan mengapa *accuracy* menyesatkan.
- [x] Memakai validasi silang berstrata dan metrik yang sesuai.
- [x] Mencoba `class_weight` dan memahami batas efektivitasnya.
- [x] Mengetahui opsi lain: ambang keputusan dan resampling.

## Masalahnya

Pada deteksi fraud, penyakit langka, atau kerusakan mesin, kelas yang penting sering hanya beberapa persen data. Model yang selalu menebak "normal" sudah mencapai akurasi sangat tinggi tanpa berguna sama sekali. Selain itu, kelas langka sedikit contohnya sehingga model kurang belajar mengenalinya.

## Prinsip penanganan

1. **Gunakan metrik yang tepat**: precision, recall, F1, PR-AUC; bukan *accuracy* saja.
2. **Validasi berstrata** (`StratifiedKFold`) agar tiap lipatan memuat kelas langka.
3. **Coba beberapa pendekatan**:
    - *Class weight*: menaikkan bobot kesalahan pada kelas langka (`class_weight="balanced"`).
    - **Ambang keputusan**: geser dari 0,5 (lihat [Evaluasi](evaluation.md)).
    - *Resampling* (oversampling/undersampling, SMOTE): memerlukan pustaka `imbalanced-learn` yang **tidak dipakai di materi ini**. Jika dipakai, lakukan **hanya pada data latih di dalam lipatan** (bukan sebelum split), agar tidak bocor.
4. **Evaluasi pada distribusi asli**: data uji tidak boleh di-resample.

## Implementasi langkah demi langkah

Data **sintetis** dari `make_classification`: 2000 sampel, 10 fitur, sekitar 5% kelas positif; sengaja dibuat sulit. Program `examples/classification/imbalance_cv.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: membuat data tidak seimbang

```python title="imbalance_cv.py" linenums="1"
--8<-- "classification/imbalance_cv.py:1:17"
```

| Parameter `make_classification` | Penjelasan |
|---|---|
| `n_samples=2000`, `n_features=10`, `n_informative=5` | 2000 sampel dengan 10 fitur; hanya 5 yang benar-benar berisi informasi, sisanya derau/redundan. |
| `weights=[0.95, 0.05]` | Proporsi kelas 95% : 5% (tidak seimbang). |
| `flip_y=0.01` | 1% label dibalik secara acak (derau label). |
| `class_sep=0.8` | Pemisahan antar-kelas; makin kecil makin sulit. |

```text
--8<-- "expected/classification/imbalance_cv.txt:1:1"
```

Terdapat 108 sampel positif dari 2000 (5,4%).

### Langkah 2: mengapa validasi harus berstrata

**Konsep.** Pada kelas langka, pembagian acak dapat membuat satu lipatan berisi jauh lebih banyak atau lebih sedikit positif daripada lipatan lain, sehingga skor antar lipatan tidak sebanding.

```python title="imbalance_cv.py" linenums="19"
--8<-- "classification/imbalance_cv.py:19:23"
```

Perulangan memakai dua *splitter* secara bergantian. `splitter.split(X, y)` menghasilkan pasangan indeks `(latih, validasi)` untuk tiap lipatan; `y[val].sum()` menghitung jumlah positif di lipatan validasi (karena `y` bernilai 0/1). Tanda `_` berarti indeks latih tidak dipakai.

```text
--8<-- "expected/classification/imbalance_cv.txt:2:3"
```

**Interpretasi.** Dengan `KFold` biasa, jumlah positif per lipatan validasi bervariasi (17, 17, 22, 28, 24); dengan `StratifiedKFold` hampir sama (21, 21, 22, 22, 22). Lipatan yang tidak merata membuat skor antar lipatan tidak sebanding.

### Langkah 3: beberapa strategi untuk dibandingkan

```python title="imbalance_cv.py" linenums="25"
--8<-- "classification/imbalance_cv.py:25:34"
```

| Model | Arti |
|---|---|
| `DummyClassifier(strategy="most_frequent")` | *Baseline*: selalu menebak kelas mayoritas. |
| `LogisticRegression(...)` | Regresi logistik biasa. |
| `class_weight="balanced"` | Bobot kesalahan tiap kelas dibuat berbanding terbalik dengan frekuensinya, sehingga salah memprediksi kelas langka dihukum lebih berat. |
| `RandomForestClassifier(..., class_weight="balanced_subsample")` | Versi yang sama untuk hutan; bobot dihitung ulang pada tiap sampel bootstrap. |

### Langkah 4: metrik yang tepat dan perbandingan

**Konsep.** Hanya *accuracy* tidak cukup. Kita hitung *precision*, *recall*, F1, dan **PR-AUC** (`average_precision`). Bila sebuah model tidak pernah memprediksi positif, *precision* tidak terdefinisi (pembagian dengan nol). Parameter `zero_division=0` menetapkannya sebagai 0 agar perhitungan tidak berhenti dengan peringatan.

```python title="imbalance_cv.py" linenums="35"
--8<-- "classification/imbalance_cv.py:35:48"
```

| Kode | Penjelasan |
|---|---|
| `scoring = {...}` | Kamus metrik: nama → metrik. `make_scorer(fungsi, zero_division=0)` membungkus fungsi metrik agar dapat dipakai `cross_validate` dengan parameter tambahan. |
| `cross_validate(model, X, y, cv=cv, scoring=scoring)` | Hasil: kamus skor per lipatan dengan kunci `test_<nama metrik>`. |
| `s[f"test_{m}"].mean()` | Rata-rata tiap metrik atas 5 lipatan. |
| `" ".join(f"{v:.3f}" for v in vals)` | Menggabungkan angka menjadi satu baris teks dengan tiga desimal. |

```text
--8<-- "expected/classification/imbalance_cv.txt:5:10"
```

**Interpretasi.**

- **Jebakan *accuracy***: model "selalu tebak mayoritas" mencapai akurasi **0,946** dengan recall 0 dan PR-AUC 0,054 (sama dengan proporsi positif 5,4%: tebakan acak). Regresi logistik biasa juga hanya 0,946 dengan recall 0,009: ia hampir tidak pernah memprediksi positif, jadi tidak lebih berguna daripada tebakan mayoritas.
- **`class_weight="balanced"` pada regresi logistik**: recall melonjak ke 0,768, tetapi precision hanya 0,124 dan akurasi turun ke 0,692. Model menangkap banyak positif dengan membayar banyak alarm palsu; F1 naik dari 0,017 menjadi 0,213. Apakah ini baik bergantung pada biaya alarm palsu.
- **Random Forest** jauh lebih baik: precision 0,949, recall 0,286, F1 0,434, PR-AUC 0,610. Ia jarang memprediksi positif, tetapi bila ya hampir selalu benar.
- **Random Forest dengan bobot seimbang**: precision 1,000, recall 0,212, F1 0,344, PR-AUC 0,641. **F1-nya lebih rendah** daripada tanpa bobot pada ambang 0,5, sedangkan PR-AUC sedikit lebih tinggi. Jadi *class weight* tidak otomatis memperbaiki hasil; perlu diuji, dan sering lebih efektif menggeser ambang pada model yang sama.
- Semua angka berasal dari data sintetis yang dibuat sulit; tingkat kesulitan nyata di data Anda akan berbeda.

??? example "Program lengkap (`imbalance_cv.py`)"
    ```python title="imbalance_cv.py" linenums="1"
    --8<-- "classification/imbalance_cv.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Melaporkan *accuracy* pada data timpang.
    - Melakukan oversampling **sebelum** membagi data (menduplikasi sampel ke latih dan validasi sekaligus).
    - Me-*resample* data uji.
    - Memakai `class_weight` tanpa mengukur dampaknya.
    - Membandingkan model hanya pada ambang 0,5 padahal ambang optimal berbeda tiap model.

## Latihan

1. Ubah proporsi positif menjadi 20% (`weights=[0.8, 0.2]`). Bagaimana jarak antara model biasa dan berbobot?
2. Untuk Random Forest, geser ambang dari 0,5 ke 0,2 dengan `cross_val_predict`. Bagaimana precision dan recall berubah?
3. Mengapa PR-AUC dasar (tebakan acak) sama dengan proporsi kelas positif?

## Ringkasan

Pada data timpang, jangan percaya *accuracy*. Gunakan validasi berstrata, metrik precision/recall/F1/PR-AUC, dan uji pendekatan seperti `class_weight` atau pergeseran ambang. Setiap teknik harus diuji; hasilnya bergantung pada data.

Berikutnya: [Optimasi Hyperparameter](hyperparameter-tuning.md).
