# Proyek: Prediksi Churn Pelanggan End-to-End

## Tujuan proyek

Menerapkan seluruh alur klasifikasi pada data tabular bertipe campuran: pembagian data, *pipeline*, *baseline*, perbandingan model, optimasi ringan, pemilihan ambang, evaluasi akhir, dan penyimpanan model.

## Masalah dan data

**Masalah:** memprediksi pelanggan yang akan berhenti berlangganan (`churn = 1`) agar tim retensi dapat menghubungi mereka lebih dulu. Menghubungi pelanggan yang sebenarnya tidak akan berhenti punya biaya kecil; melewatkan pelanggan yang akan berhenti kehilangan pendapatan. Karena itu recall kelas churn penting, tetapi tidak boleh membuat precision runtuh.

**Data:** data pelanggan **sintetis** dari `customer_data.py` (lihat [pandas](../fundamentals/pandas.md#dataset-yang-dipakai)), berisi nilai hilang, *outlier*, dan duplikat. Karena sintetis, hasilnya tidak menjelaskan pelanggan nyata. Skrip proyek mengimpor pembangkit data dari `examples/fundamentals/`.

## Alur kerja

1. Hapus duplikat dan pisahkan data uji (*stratified*).
2. Praproses di dalam `ColumnTransformer` (imputasi, *scaling*, *one-hot*).
3. Bandingkan *baseline* dan empat model dengan validasi silang berulang (ROC-AUC, PR-AUC, F1).
4. Setel hyperparameter kandidat terbaik.
5. Pilih ambang dari prediksi validasi silang pada **data latih**.
6. Evaluasi akhir sekali pada data uji; simpan model.

## Implementasi langkah demi langkah

Program `examples/classification/project.py` mengimpor pembangkit data dari `examples/fundamentals/` (lihat [pandas](../fundamentals/pandas.md#dataset-yang-dipakai)). Konsep di tiap langkah sudah dibahas di pelajaran sebelumnya; di sini fokusnya pada **bagaimana langkah-langkah itu disatukan** dan keputusan yang diambil. Program lengkap ada di akhir bagian ini.

### Langkah 1: data, pembersihan duplikat, dan data uji

```python title="project.py" linenums="33"
--8<-- "classification/project.py:33:45"
```

| Kode | Penjelasan |
|---|---|
| `sys.path.insert(0, ...)` dan `from customer_data import ...` | Menambahkan folder `fundamentals` ke jalur pencarian modul Python agar `customer_data.py` dapat diimpor dari folder lain. |
| `make_customer_data().drop_duplicates().reset_index(drop=True)` | Data sintetis lalu buang 5 baris duplikat (langkah tanpa parameter terlatih, aman sebelum split). |
| `X, y = df.drop(columns="churn"), df["churn"]` | Memisahkan fitur dan target. |
| `train_test_split(..., test_size=0.2, stratify=y, ...)` | Menyisihkan data uji berstrata. |

```text
--8<-- "expected/classification/project.txt:1:1"
```

**Hasil.** 480 data latih dan 120 data uji, dengan 7 kolom fitur; proporsi churn pada data latih 0,338.

### Langkah 2: praproses di dalam pipeline

**Konsep.** Data ini campuran angka dan teks, dan ada nilai hilang, sehingga tiap model memerlukan praproses yang sama. Agar tidak mengulang kode, praproses dibungkus dalam sebuah **fungsi** yang mengembalikan pipeline lengkap untuk model apa pun.

```python title="project.py" linenums="47"
--8<-- "classification/project.py:47:61"
```

| Kode | Penjelasan |
|---|---|
| `def make_pipeline_with(clf, scale=True)` | Fungsi dengan dua **parameter**: model `clf` dan `scale` (bawaan `True`). Mengembalikan pipeline. |
| `num_steps = [("impute", SimpleImputer(strategy="median"))]` | Daftar langkah numerik, dimulai dari imputasi median. |
| `if scale: num_steps.append(...)` | **Percabangan**: scaler hanya ditambahkan bila `scale=True`. Random Forest tidak memerlukannya, sehingga dipanggil dengan `scale=False`. |
| `ColumnTransformer([("num", ...), ("cat", ...)])` | Praproses berbeda untuk kolom numerik dan kategorik ([penjelasan](../fundamentals/sklearn-pipeline.md)). |
| `return Pipeline([("prep", prep), ("clf", clf)])` | Menggabungkan praproses dan model. |

### Langkah 3: membandingkan kandidat dengan *baseline*

```python title="project.py" linenums="64"
--8<-- "classification/project.py:64:79"
```

Lima kandidat disimpan dalam kamus: *baseline* mayoritas dan empat model. Perulangan menilai tiap model dengan `RepeatedStratifiedKFold` (5 lipatan × 2 pengulangan) pada **data latih** dengan tiga metrik: ROC-AUC, PR-AUC (`average_precision`), dan F1 pada ambang 0,5. Baris `print` memformat hasilnya sebagai tabel teks.

```text
--8<-- "expected/classification/project.txt:3:8"
```

**Interpretasi.** *Baseline* memberi ROC-AUC 0,500 dan PR-AUC 0,338 (= proporsi churn), F1 0. Semua model mengalahkannya: Random Forest ROC-AUC 0,720 dan PR-AUC 0,561, Logistic Regression 0,706 dan 0,522, KNN 0,675 dan 0,494, SVM 0,670 dan 0,519. **Sinyalnya moderat** (ROC-AUC ≈ 0,7), wajar karena pembangkit data memang berderau. Selisih Random Forest dan Logistic Regression kecil; tanpa pengulangan lebih banyak, tidak ada pemenang yang pasti.

### Langkah 4: optimasi ringan

```python title="project.py" linenums="81"
--8<-- "classification/project.py:81:89"
```

`GridSearchCV` menyetel `C` regresi logistik (`clf__C`: nama langkah `clf`, parameter `C`) dengan ROC-AUC sebagai metrik; `search.best_estimator_` adalah pipeline terbaik yang sudah dilatih ulang pada seluruh data latih. Regresi logistik dipilih karena sederhana dan skornya hampir setara dengan Random Forest. Ini keputusan praktis, bukan klaim bahwa ia terbaik.

```text
--8<-- "expected/classification/project.txt:10:10"
```

**Interpretasi.** `C` terbaik 0,01 (regularisasi kuat), ROC-AUC CV 0,707, hampir sama dengan regresi logistik awal 0,706: penyetelan tidak memberi perubahan berarti.

### Langkah 5: memilih ambang tanpa menyentuh data uji

**Konsep.** Ambang 0,5 memberi F1 rendah (kisaran 0,36–0,44 pada tabel). Ambang yang lebih rendah mengutamakan *recall* churn. Ambang dipilih dari prediksi **validasi silang pada data latih**, bukan data uji.

```python title="project.py" linenums="91"
--8<-- "classification/project.py:91:97"
```

| Kode | Penjelasan |
|---|---|
| `cross_val_predict(best, X_train, y_train, cv=..., method="predict_proba")` | Untuk tiap sampel latih, menghasilkan probabilitas yang diprediksi oleh model yang **tidak melihat sampel itu** (*out-of-fold*). |
| `np.arange(0.2, 0.71, 0.05)` | Kandidat ambang dari 0,20 sampai 0,70 dengan langkah 0,05. |
| `f1_score(y_train, oof >= t)` | F1 pada tiap ambang; `oof >= t` menghasilkan `True`/`False` yang dibaca sebagai prediksi 1/0. |
| `thresholds[int(np.argmax(f1s))]` | Ambang dengan F1 tertinggi (`argmax` = indeks nilai terbesar). |

```text
--8<-- "expected/classification/project.txt:11:11"
```

**Hasil.** Ambang 0,30 terpilih (F1 CV 0,567).

### Langkah 6: evaluasi akhir sekali pada data uji

```python title="project.py" linenums="99"
--8<-- "classification/project.py:99:104"
```

`best.predict_proba(X_test)[:, 1]` memberi probabilitas churn; `(proba >= threshold).astype(int)` mengubahnya menjadi prediksi 0/1 dengan ambang terpilih.

```text
--8<-- "expected/classification/project.txt:13:24"
```

**Interpretasi.**

- ROC-AUC uji 0,677 dan PR-AUC 0,632; ROC-AUC uji **lebih rendah** daripada validasi silang (0,707). Dengan 120 sampel (40 churn), selisih seperti ini masih dalam kisaran ketidakpastian.
- *Confusion matrix*: dari 40 churn, 31 tertangkap (*recall* 0,775) dan 9 terlewat; dari 80 yang bertahan, 46 diperingatkan keliru (*precision* churn 0,403).
- **Akurasi 0,542 lebih rendah daripada *baseline* mayoritas (sekitar 0,667).** Ini bukan berarti model lebih buruk: ambang 0,30 sengaja menukar akurasi dengan *recall*. Layak tidaknya bergantung pada biaya menghubungi pelanggan dibanding biaya kehilangan pelanggan, informasi yang tidak ada di data ini.

### Langkah 7: menyimpan model beserta ambangnya

```python title="project.py" linenums="106"
--8<-- "classification/project.py:106:111"
```

Yang disimpan adalah **kamus** berisi pipeline dan ambang, karena keduanya diperlukan untuk memprediksi dengan benar. Setelah dimuat ulang, prediksi dicocokkan dengan prediksi sebelumnya.

```text
--8<-- "expected/classification/project.txt:26:26"
```

**Hasil.** Prediksi identik setelah model dimuat ulang (`True`).

??? example "Program lengkap (`project.py`)"
    ```python title="project.py" linenums="1"
    --8<-- "classification/project.py"
    ```

## Refleksi dan pengembangan

- Sinyal data lemah: tambahkan fitur yang bermakna (sulit untuk data sintetis ini) atau kumpulkan data lebih baik daripada memperumit model.
- Gunakan ambang yang dipilih dari biaya bisnis, bukan sekadar F1.
- Tambahkan pemantauan pergeseran data seperti di [lesson sebelumnya](saving-deployment.md).

## Latihan

1. Setel `RandomForestClassifier` dengan `RandomizedSearchCV`, lalu bandingkan dengan Logistic Regression yang disetel.
2. Ganti kriteria pemilihan ambang dari F1 ke "recall minimal 0,8 dengan precision setinggi mungkin".
3. Tambahkan `Pipeline` baseline `DummyClassifier(strategy="stratified")` dan lihat PR-AUC-nya.
4. Tulis laporan satu halaman untuk tim retensi: apa yang dapat dan tidak dapat disimpulkan dari model ini.

## Ringkasan

Proyek ini memperlihatkan alur lengkap dan keputusan metodologisnya: praproses di dalam *pipeline*, *baseline*, perbandingan adil, ambang dari data latih, evaluasi sekali di data uji, dan penyimpanan. Hasil yang moderat pada data sintetis ini adalah pelajaran: laporkan apa adanya.

Selesai dengan bab Klasifikasi. Lanjut ke bab [Estimasi (Regresi)](../estimation/index.md).
