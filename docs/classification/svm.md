# Support Vector Machine (SVM)

## Tujuan pembelajaran

- [x] Menjelaskan ide *margin* maksimum dan peran *support vector*.
- [x] Memahami `C`, kernel, dan `gamma` pada SVM RBF.
- [x] Menunjukkan mengapa *scaling* wajib dan bagaimana `gamma` yang salah menghancurkan model.

## Intuisi

SVM mencari batas pemisah dengan **margin** terbesar, yaitu jarak terjauh ke sampel terdekat dari tiap kelas. Sampel yang menentukan batas itu disebut ***support vector***; sampel lain tidak memengaruhi model. Pada data yang tidak terpisah sempurna, SVM memakai *soft margin* yang membolehkan sebagian pelanggaran:

$$
\min_{\mathbf{w}, b, \mathbf{\xi}}\; \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_{i} \xi_i
$$

\(\xi_i\) adalah besar pelanggaran margin sampel \(i\). **`C` besar** menghukum pelanggaran berat (margin sempit, risiko *overfitting*); **`C` kecil** menoleransi pelanggaran (margin lebar, lebih halus).

Untuk batas non-linear, SVM memakai **kernel**. Kernel RBF mengukur kemiripan dua titik:

$$
K(\mathbf{a}, \mathbf{b}) = \exp\!\left(-\gamma \lVert \mathbf{a} - \mathbf{b} \rVert^2\right)
$$

`gamma` (\(\gamma\)) mengatur jangkauan pengaruh satu titik: **`gamma` besar = pengaruh sangat lokal** (batas berliku, risiko *overfitting*), `gamma` kecil = pengaruh luas (batas halus).

## Ringkasan model

| Aspek | Keterangan |
|---|---|
| Kelebihan | Efektif pada data berdimensi tinggi dan berukuran kecil–menengah; batas non-linear lewat kernel |
| Keterbatasan | Lambat pada data sangat besar, sensitif terhadap `C`/`gamma` dan *scaling*, tidak langsung memberi probabilitas |
| Hyperparameter penting | `C`, `kernel`, `gamma` |
| Praproses | ***Scaling* wajib** |
| Cocok untuk | Data tabular kecil–menengah berdimensi tinggi; pembanding kuat |

## Implementasi langkah demi langkah

Program `examples/classification/svm.py` memakai data Breast Cancer; pemuatan dan pembagian datanya sama dengan [pelajaran dasar klasifikasi](preprocessing-baseline.md#langkah-1-memuat-data-dan-memisahkan-data-uji).

### Langkah 1: fungsi bantu dan membandingkan kernel

**Tujuan.** Membandingkan kernel linear dan RBF, serta melihat pengaruh *scaling*.

```python title="svm.py" linenums="15"
--8<-- "classification/svm.py:15:23"
```

| Kode | Penjelasan |
|---|---|
| `def cv_acc(model): return cross_val_score(...).mean()` | **Fungsi bantu**: menerima sebuah model dan mengembalikan akurasi validasi silang 5 lipatan pada data latih. Dengan fungsi ini, baris yang sama tidak perlu ditulis berulang. `model` adalah **parameter** fungsi, dan nilai setelah `return` adalah keluarannya. |
| `SVC(kernel='linear')` | SVM dengan kernel linear: batas pemisah berupa bidang lurus. |
| `SVC(kernel='rbf')` | Kernel RBF (*radial basis function*): batas dapat melengkung. Ini nilai bawaan. |
| `make_pipeline(StandardScaler(), SVC(...))` | SVM berbasis jarak sehingga *scaling* wajib; di baris terakhir sengaja tidak dipakai untuk perbandingan. |

```text
--8<-- "expected/classification/svm.txt:1:4"
```

**Interpretasi.** Kernel linear 0,967, RBF 0,971, dan **RBF tanpa *scaling* hanya 0,910**: fitur berskala besar mendominasi perhitungan jarak, sama seperti pada KNN.

### Langkah 2: pengaruh `C` dan `gamma`

**Konsep.** `C` mengatur toleransi pelanggaran margin (besar = margin sempit, risiko *overfitting*). `gamma` mengatur jangkauan pengaruh satu titik pada kernel RBF (besar = sangat lokal). Keduanya adalah **hyperparameter**: ditetapkan sebelum pelatihan.

```python title="svm.py" linenums="25"
--8<-- "classification/svm.py:25:30"
```

| Kode | Penjelasan |
|---|---|
| `for C, gamma in [(1, 0.001), ...]` | Mengulang daftar pasangan (C, gamma); tiap pasangan langsung dipecah ke dua variabel. |
| `gamma="scale"` | Nilai otomatis scikit-learn: \(1/(\text{jumlah fitur} \times \text{varians } X)\). Titik awal yang masuk akal. |
| `model.fit(...).score(X_train, y_train)` | **Akurasi latih**; dibandingkan dengan akurasi validasi silang untuk mendiagnosis *overfitting*. |

```text
--8<-- "expected/classification/svm.txt:6:12"
```

**Interpretasi.**

- `gamma=1.0`: akurasi latih 1,000 tetapi validasi silang **0,631** (nyaris proporsi kelas mayoritas 0,626): setiap titik hanya memengaruhi tetangga sangat dekat sehingga model menghafal data latih dan gagal pada data baru, contoh *overfitting* ekstrem.
- `C=1`, `gamma=0,001`: model terlalu halus (latih 0,952; CV 0,947).
- Kombinasi terbaik pada tabel kecil ini: `C=100`, `gamma=0,001` (CV 0,978); `C=1`, `gamma="scale"` (0,971) hampir setara.
- `C=100` dengan `gamma="scale"` justru turun (0,958): `C` dan `gamma` berinteraksi sehingga harus disetel bersama (lihat [Optimasi Hyperparameter](hyperparameter-tuning.md)).

### Langkah 3: evaluasi akhir dan *support vector*

```python title="svm.py" linenums="32"
--8<-- "classification/svm.py:32:35"
```

`final[-1]` adalah langkah terakhir pipeline (objek `SVC`); `n_support_` adalah jumlah **support vector** per kelas (atribut berakhiran `_` = hasil pelatihan). `.tolist()` mengubah array menjadi list agar tercetak ringkas.

```text
--8<-- "expected/classification/svm.txt:14:23"
```

**Interpretasi.** 97 dari 455 data latih (51 kelas ganas, 46 kelas jinak) menjadi *support vector*, yaitu sampel yang menentukan batas keputusan; sampel lain tidak memengaruhi model. Pada data uji akurasi 0,982, setara regresi logistik pada data ini.

??? example "Program lengkap (`svm.py`)"
    ```python title="svm.py" linenums="1"
    --8<-- "classification/svm.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai SVM tanpa *scaling*.
    - Memilih `gamma` terlalu besar sehingga model menghafal.
    - Menyetel `C` dan `gamma` terpisah padahal keduanya berinteraksi.
    - Menggunakan SVM kernel pada data sangat besar tanpa mempertimbangkan waktu pelatihan.
    - Menganggap `decision_function` sebagai probabilitas.

## Latihan

1. Coba `kernel="poly"` dengan `degree=2` dan `degree=3`.
2. Tambahkan `class_weight="balanced"`. Apakah recall kelas ganas berubah?
3. Hitung jumlah *support vector* untuk `C=0,1` dan `C=100`. Mana yang lebih banyak, dan mengapa?

## Ringkasan

SVM memaksimalkan margin; `C` mengatur toleransi pelanggaran dan `gamma` jangkauan pengaruh titik pada kernel RBF. *Scaling* wajib, dan `C` serta `gamma` perlu disetel bersama.

Berikutnya: [Naive Bayes](naive-bayes.md).
