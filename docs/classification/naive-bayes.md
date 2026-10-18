# Naive Bayes

## Tujuan pembelajaran

- [x] Menjelaskan teorema Bayes dan asumsi "naif" independensi fitur.
- [x] Membedakan `GaussianNB` (fitur numerik) dan `MultinomialNB` (cacah kata).
- [x] Memahami mengapa Naive Bayes tetap berguna meskipun asumsinya tidak terpenuhi.

## Intuisi

Naive Bayes menghitung peluang tiap kelas berdasarkan teorema Bayes:

$$
P(c \mid x_1, \dots, x_p) \;\propto\; P(c)\,\prod_{j=1}^{p} P(x_j \mid c)
$$

\(P(c)\) adalah peluang awal (*prior*) kelas \(c\), dan \(P(x_j \mid c)\) peluang fitur \(x_j\) pada kelas tersebut. Bagian "naif" adalah **asumsi bahwa fitur saling independen jika kelasnya diketahui**, yang membuat peluang gabungan cukup berupa perkalian. Kelas dengan hasil tertinggi dipilih.

- **`GaussianNB`**: tiap fitur numerik per kelas dimodelkan dengan distribusi normal.
- **`MultinomialNB`**: untuk cacah (misalnya frekuensi kata pada teks); parameter `alpha` (*Laplace smoothing*) mencegah peluang nol bagi kata yang tidak pernah muncul pada suatu kelas.

## Ringkasan model

| Aspek | Keterangan |
|---|---|
| Kelebihan | Sangat cepat, bekerja pada data kecil, baik untuk teks berdimensi tinggi |
| Keterbatasan | Asumsi independensi jarang benar; probabilitas cenderung terlalu ekstrem |
| Hyperparameter penting | `var_smoothing` (Gaussian), `alpha` (Multinomial) |
| Praproses | Tidak perlu *scaling* untuk `GaussianNB`; teks perlu diubah menjadi cacah kata |
| Cocok untuk | *Baseline* cepat, klasifikasi teks (spam, topik) |

## Implementasi langkah demi langkah

Program `examples/classification/naive_bayes.py` terdiri atas dua bagian: Bagian 1 memakai Breast Cancer (data publik, fitur numerik); Bagian 2 memakai **8 kalimat sintetis buatan sendiri**, hanya untuk ilustrasi mekanisme teks, bukan data nyata.

### Langkah 1: `GaussianNB` pada fitur numerik

**Tujuan.** Melatih Naive Bayes untuk fitur numerik; asumsi tiap fitur berdistribusi normal di dalam tiap kelas.

```python title="naive_bayes.py" linenums="11"
--8<-- "classification/naive_bayes.py:11:20"
```

| Kode | Penjelasan |
|---|---|
| `GaussianNB()` | Menginisialisasi model. Tidak ada hyperparameter penting yang harus disetel; tidak perlu *scaling*. |
| `cross_val_score(gnb, X_train, y_train, cv=5).mean()` | Akurasi validasi silang pada data latih. |
| `gnb.fit(X_train, y_train)` | Melatih: untuk tiap kelas, menghitung rata-rata dan varians tiap fitur, serta proporsi kelas. |
| `gnb.class_prior_` | **Prior kelas**: peluang tiap kelas sebelum melihat fitur; hasil pelatihan (berakhiran `_`), sama dengan proporsi kelas di data latih. |

```text
--8<-- "expected/classification/naive_bayes.txt:1:10"
```

**Interpretasi.** Akurasi validasi silang 0,936 dan akurasi uji 0,939, lebih rendah daripada regresi logistik dan SVM. *Prior* kelas ganas 0,374 dan jinak 0,626 hanya proporsi kelas pada data latih. Hasil uji ini kebetulan sama persis dengan pohon berkedalaman 3 (sama-sama 7 kesalahan dari 114), bukan karena kedua model sama.

### Langkah 2: memeriksa asumsi independensi

```python title="naive_bayes.py" linenums="22"
--8<-- "classification/naive_bayes.py:22:24"
```

`X_train[[...]].corr()` menghitung korelasi antar-tiga fitur ukuran tumor pada data latih; `.round(3)` membulatkannya.

```text
--8<-- "expected/classification/naive_bayes.txt:12:16"
```

**Interpretasi.** Korelasi `mean radius`–`mean perimeter` 0,998 dan `mean radius`–`mean area` 0,989. Fitur-fitur ini membawa informasi yang hampir sama, tetapi Naive Bayes memperlakukannya sebagai bukti independen sehingga bukti "dihitung berulang" dan model menjadi terlalu yakin. Akurasinya masih wajar karena keputusan kelas tidak sepenuhnya bergantung pada kalibrasi probabilitas.

### Langkah 3: `MultinomialNB` pada teks

**Konsep.** Model hanya menerima angka, jadi teks diubah menjadi **cacah kata** (*bag of words*): tiap kalimat menjadi vektor jumlah kemunculan tiap kata. `CountVectorizer` melakukannya; `MultinomialNB` lalu menghitung peluang tiap kata pada tiap kelas. Parameter `alpha` (*Laplace smoothing*) menambahkan hitungan kecil agar kata yang tidak pernah muncul pada satu kelas tidak memberi peluang nol.

```python title="naive_bayes.py" linenums="26"
--8<-- "classification/naive_bayes.py:26:39"
```

| Kode | Penjelasan |
|---|---|
| `docs = [...]` | Daftar 8 kalimat latih (4 bergaya spam, 4 bukan spam). **Sintetis**, sangat kecil. |
| `labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])` | Label tiap kalimat: 1 = spam, 0 = bukan spam, sejajar urutannya dengan `docs`. |
| `make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))` | Merangkai pengubah teks dan model. `CountVectorizer` belajar daftar kata (kosakata) dari data latih. |
| `text_model.fit(docs, labels)` | Latih: membangun kosakata dan menghitung peluang kata per kelas. |
| `predict_proba(tests)[:, 1]` | Probabilitas **spam** (kolom kelas 1) untuk tiga kalimat baru; pipeline otomatis mengubah teks baru dengan kosakata yang sama. |
| `for t, p in zip(tests, ...)` | `zip` memasangkan tiap kalimat dengan probabilitasnya; f-string `f"{p:.3f}"` menampilkannya dengan tiga desimal. |

```text
--8<-- "expected/classification/naive_bayes.txt:17:19"
```

**Interpretasi.** "hadiah gratis klik" mendapat \(P(\text{spam}) = 0{,}978\) karena ketiga katanya khas spam; "laporan rapat proyek" 0,034; "diskon laporan" 0,490 karena satu kata khas spam dan satu kata khas bukan-spam saling mengimbangi. Dengan hanya 8 kalimat, angka ini hanya ilustrasi mekanisme dan tidak menunjukkan kualitas model pada data teks nyata.

??? example "Program lengkap (`naive_bayes.py`)"
    ```python title="naive_bayes.py" linenums="1"
    --8<-- "classification/naive_bayes.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai probabilitas Naive Bayes sebagai estimasi terkalibrasi.
    - Memakai `MultinomialNB` pada fitur bernilai negatif (misalnya hasil *scaling*).
    - Mengabaikan `alpha`; `alpha=0` menyebabkan peluang nol bagi kata baru.
    - Menyimpulkan kualitas model dari contoh teks kecil.

## Latihan

1. Ubah `alpha` pada `MultinomialNB` menjadi 0,01 dan 10. Bagaimana probabilitas "diskon laporan" berubah?
2. Naikkan `var_smoothing` pada `GaussianNB` (mis. `1e-5`, `1e-2`). Apakah akurasi CV berubah?
3. Hapus dua fitur ukuran yang berkorelasi tinggi. Apakah akurasi Naive Bayes naik?

## Ringkasan

Naive Bayes mengalikan peluang fitur per kelas dengan asumsi independensi. Cepat dan efektif untuk teks, tetapi asumsinya sering dilanggar sehingga probabilitasnya tidak boleh dipercaya begitu saja.

Berikutnya: [Perbandingan dan Pemilihan Model](model-selection.md).
