# Overfitting, Underfitting, dan Kebocoran Data

## Tujuan pembelajaran

- [x] Membedakan *underfitting*, *overfitting*, dan model yang menggeneralisasi dengan baik.
- [x] Mendiagnosis keduanya dari selisih skor latih dan uji.
- [x] Mendemonstrasikan kebocoran data (*data leakage*) dan cara mencegahnya.

## Konsep dasar

- ***Underfitting***: model terlalu sederhana; skor buruk **baik pada data latih maupun uji**.
- ***Overfitting***: model terlalu kompleks; skor latih sangat tinggi tetapi skor uji jauh lebih rendah, karena model menghafal derau (*noise*).
- ***Generalisasi***: performa pada data baru; inilah yang sebenarnya ingin dicapai.
- ***Data leakage***: informasi dari luar data latih (data uji, masa depan, atau target) ikut memengaruhi pelatihan, sehingga skor tampak lebih baik daripada kenyataan.

| Gejala | Skor latih | Skor uji | Tindakan umum |
|---|---|---|---|
| Underfitting | rendah | rendah | model lebih fleksibel, tambah fitur informatif, kurangi regularisasi |
| Cocok | tinggi | tinggi, dekat latih | pertahankan |
| Overfitting | sangat tinggi | jauh lebih rendah | sederhanakan model, regularisasi, tambah data, validasi silang |

## Implementasi langkah demi langkah

Skrip memuat tiga percobaan: (1) kedalaman pohon pada Breast Cancer (data publik bawaan scikit-learn), (2) regresi polinomial pada data sintetis \(y = \sin(2\pi x) + \text{noise}\), dan (3) kebocoran data pada data acak sintetis. Program lengkap ada di akhir bagian ini.

### Percobaan 1: overfitting pada pohon keputusan

**Tujuan.** Melihat bagaimana skor latih dan skor uji berpisah ketika model dibuat makin kompleks.

```python title="overfitting_leakage.py" linenums="1"
--8<-- "fundamentals/overfitting_leakage.py:1:21"
```

| Kode | Penjelasan |
|---|---|
| `load_breast_cancer(return_X_y=True)` | Memuat data publik 569 sampel × 30 fitur; `return_X_y=True` langsung mengembalikan `X` dan `y`. |
| `train_test_split(..., test_size=0.3, stratify=y, random_state=SEED)` | 70% latih, 30% uji (171 sampel), proporsi kelas terjaga. |
| `for depth in [1, 2, 3, 5, 10, None]` | Mengulang untuk tiap nilai `max_depth` (kedalaman maksimum pohon); `None` = tanpa batas. |
| `DecisionTreeClassifier(max_depth=depth, ...).fit(X_train, y_train)` | **Menginisialisasi** pohon dengan hyperparameter tertentu lalu **melatihnya** dalam satu baris (`fit` mengembalikan objek pohon itu sendiri). |
| `tree.score(X_train, y_train)` / `tree.score(X_test, y_test)` | Akurasi pada data **latih** dan pada data **uji**. Membandingkan keduanya adalah cara mendiagnosis *overfitting*. |

```text
--8<-- "expected/fundamentals/overfitting_leakage.txt:1:7"
```

**Interpretasi.** Akurasi latih naik terus sampai 1,000 pada `max_depth` 10 dan tanpa batas, sedangkan akurasi uji bertahan di kisaran 0,91–0,93 (tertinggi 0,930 pada kedalaman 5). Selisih latih–uji melebar (0,065 pada kedalaman 5 vs 0,082 tanpa batas), ciri awal *overfitting*. Karena data uji hanya 171 sampel, selisih uji 0,01–0,02 berada dalam kisaran ketidakpastian; jangan memilih kedalaman dari angka tunggal ini.

### Percobaan 2: *underfitting* dan *overfitting* pada regresi

**Konsep.** Derajat polinomial menentukan kelenturan kurva: derajat 1 = garis lurus (terlalu kaku), derajat 15 = sangat lentur (menghafal derau).

```python title="overfitting_leakage.py" linenums="23"
--8<-- "fundamentals/overfitting_leakage.py:23:35"
```

| Kode | Penjelasan |
|---|---|
| `np.sin(2 * np.pi * x_all) + rng.normal(0, 0.3, 130)` | Data **sintetis**: gelombang sinus ditambah derau acak (simpangan baku 0,3). |
| `x_all[:30].reshape(-1, 1)` | 30 titik pertama menjadi data latih. `reshape(-1, 1)` mengubah array 1 dimensi menjadi kolom `(30, 1)`, karena `X` harus 2 dimensi (`-1` berarti "hitung otomatis"). |
| `x_all[30:]` | Sisanya (100 titik) menjadi data uji. |
| `make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())` | Pipeline: membuat fitur \(x, x^2, \dots, x^{d}\), menstandardisasinya, lalu regresi linear. `make_pipeline` memberi nama langkah otomatis. |
| `model.score(X, y)` | Untuk regresi, `score` bawaan adalah \(R^2\). |

```text
--8<-- "expected/fundamentals/overfitting_leakage.txt:9:12"
```

**Interpretasi.**

- Derajat 1: \(R^2\) latih 0,560 dan uji 0,615, **underfitting**: garis lurus tidak dapat menangkap gelombang sinus (skor buruk di kedua data).
- Derajat 4: latih 0,862 dan uji 0,823, **cukup baik**.
- Derajat 15: latih 0,912 tetapi uji turun ke 0,750, **overfitting**: kurva mengikuti derau 30 titik latih.

### Percobaan 3: kebocoran data pada seleksi fitur

**Konsep.** Bila pemilihan fitur melihat label **seluruh data** sebelum validasi silang, informasi lipatan validasi bocor. Agar kebocoran terlihat jelas, datanya sengaja acak murni: tidak ada pola nyata sehingga akurasi yang wajar sekitar 0,50.

```python title="overfitting_leakage.py" linenums="37"
--8<-- "fundamentals/overfitting_leakage.py:37:53"
```

| Kode | Penjelasan |
|---|---|
| `rng.normal(size=(100, 2000))`, `rng.integers(0, 2, size=100)` | 100 sampel dengan 2000 fitur acak dan label 0/1 acak. **Sintetis.** |
| `SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)` | Memilih 20 fitur yang paling "berhubungan" dengan label, **memakai seluruh data**. Dari 2000 fitur acak, 20 terbaik pasti tampak berhubungan secara kebetulan. |
| `cross_val_score(LogisticRegression(...), X_selected, y_noise, cv=5)` | Validasi silang **setelah** seleksi: **salah**, karena seleksi sudah melihat label data validasi. |
| `make_pipeline(SelectKBest(...), LogisticRegression(...))` | Seleksi fitur dijadikan **bagian pipeline**, sehingga dilatih ulang hanya pada data latih tiap lipatan: **benar**. |

```text
--8<-- "expected/fundamentals/overfitting_leakage.txt:14:16"
```

**Interpretasi.** Labelnya acak murni, jadi akurasi yang wajar sekitar 0,50. Seleksi di luar validasi silang memberi akurasi **0,820**: angka palsu karena seleksi sudah melihat label data validasi. Seleksi yang sama **di dalam `Pipeline`** memberi **0,470**, mendekati tebakan, yaitu gambaran jujur bahwa tidak ada pola nyata.

!!! danger "Pelajaran utama"
    Kebocoran membuat model tampak hebat di eksperimen dan gagal di dunia nyata. Semua langkah yang belajar dari data (imputasi, *scaling*, seleksi fitur, *encoding*) harus berada di dalam `Pipeline` sehingga hanya melihat data latih pada tiap lipatan. Skor yang "terlalu bagus" patut dicurigai.

??? example "Program lengkap (`overfitting_leakage.py`)"
    ```python title="overfitting_leakage.py" linenums="1"
    --8<-- "fundamentals/overfitting_leakage.py"
    ```

## Jenis kebocoran yang sering terjadi

| Jenis | Contoh | Pencegahan |
|---|---|---|
| Praproses sebelum split | *Scaler* di-*fit* pada seluruh data | `Pipeline` |
| Seleksi fitur sebelum validasi | Contoh di atas | `Pipeline` |
| Fitur dari masa depan/target | Kolom "tanggal berhenti" saat memprediksi churn | Audit fitur: tersedia saat prediksi? |
| Duplikat lintas split | Baris sama di latih dan uji | Hapus duplikat sebelum split |
| Split acak pada deret waktu | Latih memuat data setelah uji | Split kronologis |

## Kesalahan umum

!!! warning "Hindari"
    - Menyimpulkan model baik hanya dari skor latih.
    - Menyetel hyperparameter berdasarkan data uji.
    - Skor yang "terlalu bagus" tidak dicurigai; periksa kemungkinan kebocoran.
    - Mengatasi *overfitting* hanya dengan menambah kompleksitas.

## Latihan

1. Ubah `k` pada `SelectKBest` menjadi 5 dan 100. Bagaimana skor "bocor" berubah?
2. Tambahkan derajat 2 dan 8 pada demonstrasi polinomial. Pada derajat berapa uji mulai turun?
3. Ganti `random_state` pada split pohon. Apakah kedalaman terbaik tetap sama?
4. Sebutkan satu fitur pada data pelanggan yang berpotensi bocor bila ditambahkan.

## Ringkasan

Underfitting: buruk di mana-mana; overfitting: bagus di latih, buruk di uji. Diagnosis dengan membandingkan skor latih dan uji. Kebocoran data menghasilkan skor palsu; cegah dengan `Pipeline` dan split yang tepat.

Selesai dengan Fundamentals. Lanjut ke bab [Klasifikasi](../classification/index.md).
