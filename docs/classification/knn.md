# K-Nearest Neighbors (KNN)

## Tujuan pembelajaran

- [x] Menjelaskan cara KNN mengklasifikasikan berdasarkan tetangga terdekat.
- [x] Memahami pengaruh `k`, *scaling*, dan pembobotan jarak.
- [x] Mengaitkan `k` dengan *overfitting* dan *underfitting*.

## Intuisi

"Tunjukkan teman-temanmu, kuberitahu siapa dirimu." Untuk sampel baru, KNN mencari \(k\) sampel latih terdekat, lalu memilih kelas mayoritas di antara mereka. Kedekatan diukur dengan jarak, biasanya jarak Euclidean:

$$
d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{j=1}^{p} (a_j - b_j)^2}
$$

KNN tidak mempelajari parameter; ia menyimpan seluruh data latih dan menghitung jarak saat prediksi (*lazy learning*). Karena itu pelatihannya cepat, tetapi prediksinya lambat pada data besar.

## Ringkasan model

| Aspek | Keterangan |
|---|---|
| Kelebihan | Sederhana, batas keputusan fleksibel, tanpa asumsi bentuk data |
| Keterbatasan | Prediksi lambat dan memakan memori, sensitif pada fitur tidak relevan dan dimensi tinggi |
| Hyperparameter penting | `n_neighbors`, `weights`, `metric` |
| Praproses | ***Scaling* wajib**; jarak didominasi fitur berskala besar |
| Cocok untuk | Data kecil–menengah dengan sedikit fitur relevan; pembanding yang mudah dipahami |

## Implementasi langkah demi langkah

Program `examples/classification/knn.py` memakai data Breast Cancer. Pemuatan dan pembagian data sama dengan [pelajaran dasar klasifikasi](preprocessing-baseline.md#langkah-1-memuat-data-dan-memisahkan-data-uji).

### Langkah 1: pengaruh jumlah tetangga `k`

**Tujuan.** Melihat bagaimana `k` mengubah keseimbangan antara *overfitting* dan *underfitting*.

```python title="knn.py" linenums="14"
--8<-- "classification/knn.py:14:20"
```

| Kode | Penjelasan |
|---|---|
| `for k in [1, 3, 5, ...]` | Mengulang untuk tiap nilai `k` yang ingin dicoba. |
| `KNeighborsClassifier(n_neighbors=k)` | Menginisialisasi KNN dengan `k` tetangga. Ini **hyperparameter** (ditetapkan manusia, bukan dipelajari). |
| `make_pipeline(StandardScaler(), ...)` | *Scaling* wajib untuk KNN karena jarak dihitung dari nilai fitur. |
| `cross_val_score(model, X_train, y_train, cv=5).mean()` | Akurasi rata-rata validasi silang 5 lipatan pada data latih (`scoring` bawaan = akurasi). |
| `model.fit(X_train, y_train).score(X_train, y_train)` | Melatih pada seluruh data latih lalu menghitung akurasi pada data latih yang sama (**akurasi latih**). Rangkaian `.fit(...).score(...)` bekerja karena `fit` mengembalikan objek model. |

```text
--8<-- "expected/classification/knn.txt:1:8"
```

**Cara membaca.** Bandingkan kolom `akurasi_latih` dan `akurasi_CV`. Akurasi latih bisa menipu karena model diuji pada data yang sudah dilihatnya.

**Interpretasi.** `k=1` memberi akurasi latih 1,000 (menghafal; setiap titik adalah tetangga terdekat dirinya sendiri) tetapi validasi silang hanya 0,943: *overfitting*. Validasi silang terbaik pada `k=3` (0,969) dan `k=5` (0,967). Pada `k=101` akurasi latih turun ke 0,927 dan validasi silang 0,912: batas keputusan terlalu halus (*underfitting*).

### Langkah 2: pengaruh *scaling* dan pembobotan jarak

```python title="knn.py" linenums="22"
--8<-- "classification/knn.py:22:30"
```

| Kode | Penjelasan |
|---|---|
| `variants = {...}` | Kamus (*dictionary*): kunci berupa nama varian, nilai berupa model. Praktis untuk membandingkan beberapa model dengan satu perulangan. |
| `KNeighborsClassifier(n_neighbors=11)` | KNN tanpa scaler (hanya untuk perbandingan). |
| `weights="distance"` | Tetangga yang lebih dekat diberi suara lebih besar; bawaannya `"uniform"` (semua suara sama). |
| `for name, m in variants.items()` | Mengulang pasangan (nama, model) dalam kamus. |
| `{name:<29}` | Menata teks rata kiri selebar 29 karakter agar kolom rapi. |

```text
--8<-- "expected/classification/knn.txt:10:13"
```

**Interpretasi.** Pada `k=11` akurasi validasi silang 0,934 tanpa *scaling* vs 0,963 dengan *scaling*. Tanpa *scaling*, `mean area` (ratusan) mendominasi jarak dan fitur berskala kecil nyaris tak berpengaruh. Pembobotan jarak memberi 0,963, sama dengan bobot seragam pada data ini; manfaatnya bergantung pada data.

### Langkah 3: evaluasi akhir

```python title="knn.py" linenums="32"
--8<-- "classification/knn.py:32:34"
```

Model terpilih (`k=11` dengan *scaling*) dilatih pada seluruh data latih lalu diuji **sekali** pada data uji. `variants["scaling, bobot seragam"]` mengambil model itu dari kamus.

```text
--8<-- "expected/classification/knn.txt:15:22"
```

**Interpretasi.** Akurasi 0,974. *Recall* kelas ganas (0) hanya 0,929: dari 42 kasus ganas, **3 terlewat** (kelas jinak recall 1,000). Dalam konteks diagnosis, ini lebih serius daripada angka akurasi yang tampak tinggi. Karena itu metrik per kelas perlu dibaca.

??? example "Program lengkap (`knn.py`)"
    ```python title="knn.py" linenums="1"
    --8<-- "classification/knn.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Tidak melakukan *scaling*.
    - Memilih `k` berdasarkan akurasi latih (selalu terbaik pada `k=1`).
    - Memakai KNN pada data besar tanpa mempertimbangkan waktu prediksi.
    - Memasukkan banyak fitur tidak relevan; jarak menjadi tidak informatif.
    - Memakai `k` genap pada dua kelas sehingga terjadi seri.

## Latihan

1. Ganti `metric` menjadi `"manhattan"`. Apakah akurasi validasi silang berubah?
2. Uji `k` ganjil dari 1 sampai 31 dan buat grafik akurasi validasi silang terhadap `k`.
3. Hanya gunakan 5 fitur pertama. Bagaimana pengaruhnya pada KNN dibanding Logistic Regression?

## Ringkasan

KNN mengklasifikasikan dengan suara mayoritas \(k\) tetangga terdekat. `k` kecil menyebabkan *overfitting*, `k` besar *underfitting*, dan *scaling* adalah keharusan.

Berikutnya: [Decision Tree](decision-tree.md).
