# Menyimpan Model, Deployment, dan Monitoring Regresi

## Tujuan pembelajaran

- [x] Menyimpan pipeline regresi beserta metadata rentang fitur.
- [x] Mendeteksi masukan di luar rentang latih (risiko ekstrapolasi).
- [x] Memantau galat dan bias setelah model dipakai.

!!! info "Prasyarat"
    Dasar `joblib`, keamanan berkas model, bentuk deployment, dan pemantauan pergeseran fitur dibahas di [Klasifikasi: Simpan, Deploy, Monitor](../classification/saving-deployment.md). Lesson ini hanya menambahkan hal yang khas bagi regresi.

## Yang khas untuk regresi

- **Ekstrapolasi**: model linear tetap menghasilkan angka untuk masukan jauh di luar rentang latih (dan hasilnya bisa tidak masuk akal); model berbasis pohon justru tertahan pada nilai di tepi rentang. Simpan rentang fitur data latih dan beri peringatan bila masukan di luar rentang.
- **Metrik pemantauan**: begitu nilai sebenarnya diketahui, hitung MAE/RMSE per periode dan **bias** (rata-rata galat). Bias yang menjauhi nol berarti model secara sistematis terlalu rendah atau terlalu tinggi.
- **Kapan melatih ulang**: bila MAE atau bias melewati ambang yang disepakati, atau data masukan bergeser.

## Implementasi langkah demi langkah

Model: pipeline `StandardScaler` + `Ridge(alpha=10)` pada Diabetes. Batch "bergeser" di langkah terakhir adalah **simulasi sintetis** (target ditambah 30), bukan data nyata. Program `examples/estimation/saving.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: menyimpan model dan rentang fitur

**Tujuan.** Selain model, kita menyimpan **rentang nilai tiap fitur pada data latih** supaya masukan di luar rentang itu dapat ditandai saat prediksi.

```python title="saving.py" linenums="16"
--8<-- "estimation/saving.py:16:31"
```

| Kode | Penjelasan |
|---|---|
| `make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(...)` | Melatih pipeline regresi pada data latih. |
| `X_train.min().to_dict()`, `X_train.max().to_dict()` | Nilai minimum dan maksimum tiap kolom pada data latih, disimpan sebagai kamus `{kolom: nilai}`. |
| `float(mean_absolute_error(...))` | MAE acuan; `float(...)` mengubah angka NumPy menjadi `float` biasa agar dapat ditulis ke JSON. |
| `joblib.dump(model, ...)`, `json.dumps(meta)` | Menyimpan pipeline dan metadata. Dasar-dasar keduanya dibahas di [pelajaran klasifikasi](../classification/saving-deployment.md). |

### Langkah 2: memuat ulang dan memverifikasi

```python title="saving.py" linenums="33"
--8<-- "estimation/saving.py:33:35"
```

`np.allclose(a, b)` memeriksa apakah dua array **hampir sama** dalam toleransi pembulatan (lebih tepat daripada kesamaan persis untuk angka desimal).

```text
--8<-- "expected/estimation/saving.txt:1:1"
```

**Hasil.** `True`: prediksi model termuat sama dengan model asli.

### Langkah 3: prediksi dengan pemeriksaan rentang masukan

**Konsep.** Model linear tetap menghasilkan angka untuk masukan yang jauh di luar data latih (ekstrapolasi linear), padahal hasilnya mungkin tidak masuk akal. Fungsi berikut mengembalikan prediksi **dan** penanda masukan di luar rentang.

```python title="saving.py" linenums="38"
--8<-- "estimation/saving.py:38:48"
```

| Kode | Penjelasan |
|---|---|
| `lo, hi = pd.Series(meta["train_min"]), pd.Series(meta["train_max"])` | Batas bawah dan atas tiap fitur sebagai `Series`. |
| `(df < lo) \| (df > hi)` | Membandingkan tiap sel dengan batas kolomnya; `\|` berarti "atau". Hasilnya tabel `True`/`False`. |
| `.any(axis=1)` | Per **baris**: apakah ada satu fitur pun yang di luar rentang? (`axis=1` = sepanjang kolom; lihat [NumPy](../fundamentals/numpy.md)). |
| `sample.loc[sample.index[1], "bmi"] = ... * 2` | Sengaja menjadikan `bmi` sampel kedua dua kali nilai maksimum latih (nilai ekstrem **buatan**). `.copy()` sebelumnya mencegah mengubah `X_test` asli. |

```text
--8<-- "expected/estimation/saving.txt:2:2"
```

**Interpretasi.** Sampel pertama normal (`False`). Sampel kedua ditandai `True`, namun model tetap menghasilkan prediksi 334,8 (di atas sebagian besar nilai target latih): contoh model linear yang mengekstrapolasi tanpa peringatan bila tidak diperiksa.

### Langkah 4: memantau galat dan bias

**Konsep.** Setelah label sebenarnya diketahui, hitung MAE dan **bias** (rata-rata galat). Bias yang menjauhi nol berarti model secara sistematis terlalu rendah atau terlalu tinggi.

```python title="saving.py" linenums="50"
--8<-- "estimation/saving.py:50:55"
```

`shifted = y_test + 30` mensimulasikan perubahan: nilai sebenarnya naik 30 sementara model belum tahu. `(shifted - prediksi).mean()` adalah bias.

```text
--8<-- "expected/estimation/saving.txt:4:7"
```

**Interpretasi.** MAE pada data uji 42,86; pada batch dengan target bergeser +30 MAE naik menjadi 51,49 dan bias +33,9 (rata-rata prediksi terlalu rendah sekitar 34). Bias yang tampak jelas seperti ini adalah sinyal untuk memeriksa data dan melatih ulang.

??? example "Program lengkap (`saving.py`)"
    ```python title="saving.py" linenums="1"
    --8<-- "estimation/saving.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Menerima prediksi untuk masukan di luar rentang latih tanpa peringatan.
    - Hanya memantau MAE total tanpa melihat bias dan rentang.
    - Memuat berkas model dari sumber tidak tepercaya.
    - Tidak menyimpan rentang fitur dan versi pustaka.

## Latihan

1. Tambahkan pemeriksaan nilai hilang pada `predict_checked`.
2. Hitung MAE per bulan pada data simulasi dengan tanggal buatan sendiri.
3. Tentukan ambang bias yang memicu pelatihan ulang dan alasannya.

## Ringkasan

Simpan pipeline utuh dan rentang fitur, periksa masukan di luar rentang, dan pantau MAE serta bias saat label sebenarnya tersedia.

Berikutnya: [Proyek Regresi End-to-End](project.md).
