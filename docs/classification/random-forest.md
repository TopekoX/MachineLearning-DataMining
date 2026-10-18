# Random Forest

## Tujuan pembelajaran

- [x] Menjelaskan *bagging* dan pemilihan fitur acak pada Random Forest.
- [x] Memahami pengaruh `n_estimators` dan `max_features`.
- [x] Memakai skor *out-of-bag* (OOB) dan membandingkan dua jenis *feature importance*.

## Intuisi

Satu pohon cenderung bervarians tinggi. Random Forest melatih **banyak pohon** yang sengaja dibuat beragam:

1. Tiap pohon dilatih pada **sampel bootstrap** (pengambilan acak dengan pengembalian) dari data latih.
2. Pada tiap pembagian, pohon hanya mempertimbangkan **subset fitur acak** (`max_features`).
3. Prediksi akhir adalah suara mayoritas (atau rata-rata probabilitas) seluruh pohon.

Kesalahan acak tiap pohon saling meredam, sehingga varians turun dibanding satu pohon tanpa menaikkan bias secara berarti. Sekitar sepertiga data tidak terpilih pada tiap bootstrap; data ini (*out-of-bag*) dapat dipakai menilai model tanpa data validasi terpisah.

## Ringkasan model

| Aspek | Keterangan |
|---|---|
| Kelebihan | Umumnya akurat dan stabil, tahan *outlier* dan skala fitur, sedikit penyetelan |
| Keterbatasan | Lebih sulit ditafsirkan, memori dan waktu prediksi lebih besar, tidak selalu mengalahkan model sederhana |
| Hyperparameter penting | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`, `class_weight` |
| Praproses | *Scaling* tidak diperlukan |
| Cocok untuk | Data tabular dengan hubungan non-linear dan fitur campuran; *baseline* kuat |

## Implementasi langkah demi langkah

Program `examples/classification/random_forest.py` memakai data Breast Cancer; pemuatan dan pembagian datanya sama dengan [pelajaran dasar klasifikasi](preprocessing-baseline.md#langkah-1-memuat-data-dan-memisahkan-data-uji).

### Langkah 1: satu pohon sebagai pembanding

**Tujuan.** Mengetahui seberapa besar perbaikan yang diberikan hutan dibanding satu pohon.

```python title="random_forest.py" linenums="15"
--8<-- "classification/random_forest.py:15:17"
```

`cross_val_score(...)` menghasilkan lima akurasi (satu per lipatan); `tree_cv.mean()` dan `tree_cv.std()` adalah rata-rata dan simpangan bakunya. `%.3f` memformat angka dengan tiga desimal.

```text
--8<-- "expected/classification/random_forest.txt:1:1"
```

**Hasil.** Satu pohon tanpa batas memberi akurasi validasi silang 0,910 (± 0,019). Angka ini patokan untuk langkah berikutnya.

### Langkah 2: pengaruh jumlah pohon (`n_estimators`)

```python title="random_forest.py" linenums="19"
--8<-- "classification/random_forest.py:19:23"
```

`RandomForestClassifier(n_estimators=n, random_state=SEED)` menginisialisasi hutan berisi `n` pohon. Tiap pohon dilatih pada sampel bootstrap yang berbeda, dan prediksi akhir adalah suara mayoritas seluruh pohon. `random_state` mengunci keacakan sampel bootstrap dan pemilihan fitur.

```text
--8<-- "expected/classification/random_forest.txt:3:8"
```

**Interpretasi.** Satu pohon memberi 0,908 (mirip pohon tunggal sebelumnya). Menambah pohon memperbaiki hasil dengan cepat lalu mendatar: 0,941 (5), 0,954 (25 dan 100), 0,958 (300). Menambah pohon tidak menyebabkan *overfitting*, hanya menambah biaya komputasi dan memori.

### Langkah 3: jumlah fitur per pembagian (`max_features`)

```python title="random_forest.py" linenums="25"
--8<-- "classification/random_forest.py:25:29"
```

`max_features` menentukan berapa fitur yang dipertimbangkan pada **setiap** pembagian: `"sqrt"` = akar kuadrat jumlah fitur (sekitar 5 dari 30), `0.5` = separuh fitur, `1.0` = semua fitur. Nilai lebih kecil membuat pohon lebih beragam satu sama lain. `str(mf)` mengubah angka atau teks menjadi teks agar dapat diratakan dengan `:<13`.

```text
--8<-- "expected/classification/random_forest.txt:10:13"
```

**Interpretasi.** `"sqrt"` (0,960), `0,5` (0,958), dan `1,0` (0,958) hampir sama pada data ini; perbedaannya jauh di bawah ketidakpastian validasi silang. Pada data lain pengaruhnya bisa lebih besar.

### Langkah 4: model akhir, skor OOB, dan evaluasi uji

**Konsep.** Setiap pohon dilatih pada sampel bootstrap sehingga sekitar sepertiga data tidak terpakai oleh pohon itu (*out-of-bag*, OOB). Data OOB dapat dipakai menilai model **tanpa data validasi terpisah**.

```python title="random_forest.py" linenums="31"
--8<-- "classification/random_forest.py:31:34"
```

`oob_score=True` meminta skor OOB dihitung saat `fit`; hasilnya di atribut `rf.oob_score_` (berakhiran `_` = hasil pelatihan). `classification_report` menilai pada data uji.

```text
--8<-- "expected/classification/random_forest.txt:15:23"
```

**Interpretasi.** Skor OOB 0,956 sangat dekat dengan akurasi validasi silang, memberi estimasi performa gratis. Pada data uji akurasi 0,947 dan *recall* kelas ganas 0,929 (3 dari 42 terlewat). Pada data ini Random Forest **tidak** lebih baik daripada regresi logistik (0,982); Random Forest tidak selalu menang.

### Langkah 5: dua cara menghitung pentingnya fitur

**Konsep.** *Impurity importance* menghitung seberapa sering fitur dipakai menurunkan ketidakmurnian saat pelatihan. *Permutation importance* mengacak satu fitur lalu melihat seberapa turun skor model; makin besar penurunan, makin penting.

```python title="random_forest.py" linenums="36"
--8<-- "classification/random_forest.py:36:41"
```

`permutation_importance(rf, X_test, y_test, n_repeats=10, ...)` mengacak tiap fitur 10 kali pada **data uji** dan merata-ratakan penurunan skor (`.importances_mean`). `list(imp.head(3).index)` mengambil nama tiga fitur teratas.

```text
--8<-- "expected/classification/random_forest.txt:25:26"
```

**Interpretasi.** Kedua metode memberi daftar berbeda: *impurity* menempatkan `worst perimeter`, `worst area`, `worst concave points` di puncak, sedangkan *permutation* menempatkan `mean concave points`, `mean radius`, `mean fractal dimension`. Fitur ukuran (radius, perimeter, area) sangat berkorelasi sehingga pentingnya saling berbagi; permutation pada 114 data uji juga cukup berisik. Jangan menyimpulkan "fitur X yang terpenting" dari satu metode.

??? example "Program lengkap (`random_forest.py`)"
    ```python title="random_forest.py" linenums="1"
    --8<-- "classification/random_forest.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Mengira lebih banyak pohon selalu menyebabkan *overfitting* (tidak; ia mendatar).
    - Menafsirkan *impurity importance* sebagai kebenaran mutlak, terutama pada fitur berkorelasi atau bervariasi tinggi.
    - Tidak membatasi `max_depth`/`min_samples_leaf` pada data berderau.
    - Memakai Random Forest tanpa *baseline* sederhana sebagai pembanding.

## Latihan

1. Tambahkan `max_depth=3` pada hutan. Apakah akurasi CV berubah?
2. Hitung *permutation importance* pada **data latih** dan bandingkan dengan data uji.
3. Gunakan `class_weight="balanced"`. Apakah recall kelas ganas berubah?

## Ringkasan

Random Forest menggabungkan banyak pohon yang dilatih pada bootstrap dan subset fitur acak. Hasilnya lebih stabil daripada satu pohon, dengan OOB sebagai estimasi bawaan; tetapi kurang mudah dijelaskan dan tidak otomatis unggul.

Berikutnya: [Support Vector Machine (SVM)](svm.md).
