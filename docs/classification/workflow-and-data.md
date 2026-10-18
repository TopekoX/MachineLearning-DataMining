# Alur Kerja dan Data Klasifikasi

## Tujuan pembelajaran

- [x] Merumuskan masalah klasifikasi dengan jelas: apa yang diprediksi, untuk siapa, dan apa biaya kesalahannya.
- [x] Membedakan fitur, label, dan jenis klasifikasi (biner, multikelas).
- [x] Mengenal dataset yang dipakai di bab ini beserta sumber, tipe data, dan batasannya.
- [x] Melakukan pemeriksaan data awal yang relevan untuk klasifikasi.

## Apa itu klasifikasi?

Klasifikasi adalah *supervised learning* dengan target berupa **kelas diskret**. Model belajar dari contoh yang sudah berlabel, lalu memprediksi label data baru.

| Istilah | Arti |
|---|---|
| Fitur (*feature*) | Variabel masukan, misalnya ukuran sel |
| Label / target | Kelas yang diprediksi, misalnya ganas atau jinak |
| Klasifikasi biner | Dua kelas |
| Klasifikasi multikelas | Lebih dari dua kelas, satu label per sampel |

## Merumuskan masalah

Sebelum menulis kode, jawab:

1. **Apa yang diprediksi** dan kapan prediksi dibuat? (menentukan fitur apa yang boleh dipakai)
2. **Apa kelas positif yang penting?** Pada deteksi tumor, kasus ganas yang terlewat (negatif palsu) lebih berbahaya daripada alarm palsu.
3. **Metrik apa yang sesuai** dengan biaya kesalahan tersebut? (lihat [Metrik Evaluasi](../fundamentals/evaluation-metrics.md))
4. **Apa *baseline* yang harus dikalahkan?**

## Tahap alur kerja klasifikasi

Alur umum mengikuti [Peta Belajar](../roadmap/index.md). Untuk klasifikasi hampir semua tahap berlaku, dengan penekanan:

| Tahap | Penekanan pada klasifikasi |
|---|---|
| Pengumpulan data dan label | Kualitas dan konsistensi label menentukan batas atas performa |
| EDA | Distribusi kelas, fitur yang membedakan kelas |
| Praproses dan rekayasa fitur | Dalam `Pipeline`; *scaling* untuk model berbasis jarak/gradien |
| Pembagian data | *Stratified* train/test dan validasi silang |
| Pemilihan model | Mulai dari *baseline*, lalu bandingkan beberapa model |
| Evaluasi | *Confusion matrix*, precision/recall/F1, ROC-AUC/PR-AUC |
| Optimasi, penyimpanan, pemantauan | Pencarian hyperparameter, `joblib`, deteksi pergeseran data |

## Dataset: Breast Cancer Wisconsin (Diagnostic)

- **Sumber dan akses**: dataset publik yang dibundel dengan scikit-learn (`load_breast_cancer`), sehingga tidak perlu mengunduh. Dataset aslinya berasal dari UCI Machine Learning Repository; periksa ketentuan lisensi pada halaman sumber aslinya bila hendak dipakai di luar pembelajaran. Dokumentasi: [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html).
- **Isi**: 569 sampel, 30 fitur numerik. Fitur dihitung dari citra digital aspirasi jarum halus massa payudara: 10 ukuran inti sel (misalnya *radius*, *texture*, *concavity*), masing-masing dalam bentuk rata-rata (*mean*), galat standar (*error*), dan nilai terburuk (*worst*).
- **Target**: `0` = *malignant* (ganas), `1` = *benign* (jinak).
- **Batasan**: ukuran kecil, bersih, dan relatif mudah dipisahkan, sehingga skor tinggi di sini **tidak** mencerminkan masalah klinis nyata. Dataset ini dipakai sebagai alat belajar, bukan alat diagnosis.

!!! warning "Perhatikan kelas positif"
    Pada scikit-learn, metrik biner seperti `precision` dan `recall` secara bawaan menganggap kelas `1` sebagai positif, yaitu **jinak** pada dataset ini. Bila yang penting adalah menangkap kasus **ganas**, hitung metrik untuk kelas `0` (`pos_label=0`) atau lihat laporan per kelas.

## Pemeriksaan data awal langkah demi langkah

Sebelum membangun model, kenali dulu datanya. Program `examples/classification/workflow_data.py` melakukannya dalam empat langkah.

### Langkah 1: memuat data dan memahami label

**Tujuan.** Mengetahui ukuran data, arti label, dan seberapa seimbang kelasnya.

```python title="workflow_data.py" linenums="1"
--8<-- "classification/workflow_data.py:1:10"
```

| Kode | Penjelasan |
|---|---|
| `load_breast_cancer(as_frame=True)` | Memuat dataset sebagai objek `data` yang berisi tabel pandas (bukan array mentah). |
| `X, y = data.data, data.target` | `X` = tabel 30 fitur; `y` = kolom target (0 atau 1). Konvensi: **huruf besar `X`** untuk tabel 2 dimensi, **huruf kecil `y`** untuk satu kolom. |
| `data.target_names` | Nama kelas untuk angka 0 dan 1: `malignant` (ganas) dan `benign` (jinak). `map(str, ...)` mengubahnya jadi teks biasa, `enumerate` memberi nomor 0, 1, dan `dict(...)` menyusun kamus angka → nama. |
| `y.value_counts()` | Jumlah sampel tiap kelas. `.rename(index=...)` mengganti angka 0/1 dengan nama kelas agar mudah dibaca. |
| `y.value_counts(normalize=True)` | Proporsi tiap kelas (jumlah dibagi total). |

```text
--8<-- "expected/classification/workflow_data.txt:1:7"
```

**Interpretasi.** Data berukuran `(569, 30)`: 569 sampel dan 30 fitur. Kelas `0` = `malignant` (212 sampel, 37,3%) dan kelas `1` = `benign` (357 sampel, 62,7%). Perhatikan: **kelas 1 adalah jinak**. Kelas tidak terlalu timpang tetapi tidak seimbang sempurna, sehingga *stratified split* dan metrik per kelas tetap dipakai.

### Langkah 2: memeriksa nama fitur dan kualitas data

```python title="workflow_data.py" linenums="12"
--8<-- "classification/workflow_data.py:12:14"
```

`X.columns` berisi nama kolom; `X.columns[:3]` mengambil tiga yang pertama dan `X.columns[-3:]` tiga yang terakhir (indeks negatif menghitung dari belakang). `X.isna().any().any()` bertanya: "adakah sel kosong di tabel mana pun?" (`any()` pertama per kolom, `any()` kedua gabungan seluruh kolom). `bool(...)` mengubah hasilnya menjadi `True`/`False` biasa.

```text
--8<-- "expected/classification/workflow_data.txt:9:10"
```

**Interpretasi.** Nama fitur memperlihatkan polanya: 10 pengukuran inti sel (misalnya `radius`, `texture`) masing-masing dalam bentuk rata-rata (`mean`), galat standar (`error`), dan nilai terburuk (`worst`) = 30 fitur. Tidak ada data hilang dan tidak ada duplikat, sehingga praproses lebih ringan daripada data pelanggan pada bab Fundamentals.

### Langkah 3: memeriksa rentang nilai fitur

```python title="workflow_data.py" linenums="16"
--8<-- "classification/workflow_data.py:16:18"
```

`describe()` meringkas fitur; `.loc[["min", "max"]]` mengambil hanya baris `min` dan `max`; `.round(3)` membulatkan.

```text
--8<-- "expected/classification/workflow_data.txt:11:14"
```

**Interpretasi.** `mean area` berkisar 143,5–2501 sedangkan `mean smoothness` hanya 0,053–0,163: skalanya berbeda ribuan kali. Model berbasis jarak atau gradien (KNN, SVM, regresi logistik) sangat dipengaruhi hal ini sehingga memerlukan *scaling* (lihat [Praproses, Split, Baseline](preprocessing-baseline.md)).

### Langkah 4: apakah fitur membedakan kelas?

```python title="workflow_data.py" linenums="20"
--8<-- "classification/workflow_data.py:20:21"
```

`y.map(dict(...))` mengganti angka 0/1 dengan nama kelas; hasilnya dipakai sebagai kunci `groupby` untuk menghitung rata-rata dua fitur per kelas.

```text
--8<-- "expected/classification/workflow_data.txt:15:18"
```

**Interpretasi.** Rata-rata `mean radius` 12,15 (jinak) vs 17,46 (ganas), dan `mean concavity` 0,046 vs 0,161: kedua fitur tampak membedakan kelas. Ini baru gambaran awal dari rata-rata, bukan bukti bahwa fitur tersebut cukup sendirian.

??? example "Program lengkap (`workflow_data.py`)"
    ```python title="workflow_data.py" linenums="1"
    --8<-- "classification/workflow_data.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memulai dari algoritma tanpa merumuskan metrik dan biaya kesalahan.
    - Mengabaikan makna label sehingga salah menafsirkan precision/recall.
    - Memakai fitur yang baru diketahui setelah kejadian (kebocoran).
    - Menganggap skor tinggi pada dataset mainan berlaku di dunia nyata.

## Latihan

1. Tulis rumusan masalah untuk deteksi *spam* email: siapa kelas positif, apa biaya kedua jenis kesalahan, metrik apa yang cocok?
2. Hitung rasio kelas pada dataset `load_wine`. Apakah termasuk multikelas? Berapa kelasnya?
3. Fitur `worst radius` dan `mean radius` kemungkinan berkorelasi tinggi. Hitung korelasinya.

## Ringkasan

Mulailah dari rumusan masalah dan biaya kesalahan. Kenali dataset: isi, label, batasan. Periksa distribusi kelas dan skala fitur sebelum memilih praproses dan metrik.

Berikutnya: [Praproses, Pembagian Data, dan Baseline](preprocessing-baseline.md).
