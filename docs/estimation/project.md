# Proyek: Memperkirakan Harga Rumah End-to-End

## Tujuan proyek

Menerapkan seluruh alur regresi pada data tabular bertipe campuran: EDA singkat, *pipeline* praproses, *baseline*, perbandingan model, keputusan transformasi target, optimasi ringan, evaluasi akhir, analisis galat per kelompok, dan penyimpanan.

## Masalah dan data

**Masalah:** memperkirakan harga rumah (juta Rupiah) dari luas, jumlah kamar, usia, jarak ke pusat kota, kecamatan, dan kondisi.

**Data:** **sintetis**, dibangkitkan oleh fungsi `make_housing_data` dengan seed tetap (800 baris). Harga dibuat dari hubungan **multiplikatif** (harga ∝ luas^0,9 dikali faktor usia, jarak, kecamatan, dan kondisi) ditambah derau acak 20% pada skala log, serta sebagian nilai hilang (5% `age_years`, 4% `condition`). Karena kita yang membuat datanya, kita tahu bahwa derau acak ini tidak dapat diprediksi oleh model mana pun; pada data nyata batas seperti ini tidak diketahui. Hasil tidak menggambarkan pasar properti nyata.

## Alur kerja

1. EDA singkat dan pemisahan data uji (acak, 80/20).
2. Praproses dalam `ColumnTransformer` (imputasi, *scaling*, *one-hot*).
3. Bandingkan *baseline*, Ridge, dan Random Forest, masing-masing dengan target asli dan target log (`TransformedTargetRegressor`).
4. Setel `alpha` Ridge pada target log.
5. Evaluasi sekali pada data uji dan periksa galat per kecamatan.
6. Simpan dan uji muat ulang.

## Implementasi langkah demi langkah

Program `examples/estimation/project.py`. Konsep tiap langkah sudah dibahas di pelajaran sebelumnya; di sini fokusnya pada penyatuan langkah dan keputusan yang diambil. Program lengkap ada di akhir bagian ini.

### Langkah 1: membangkitkan data sintetis

**Tujuan.** Membuat data harga rumah yang strukturnya **kita ketahui**, agar hasil model dapat dibandingkan dengan batas yang diketahui.

```python title="project.py" linenums="21"
--8<-- "estimation/project.py:21:39"
```

| Bagian | Penjelasan |
|---|---|
| `rng = np.random.default_rng(seed)` | Generator acak dengan seed tetap (data identik tiap dijalankan). |
| `area = rng.uniform(30, 300, n)`; `distance = rng.gamma(2.0, 4.0, n)` | Fitur numerik dengan distribusi masing-masing (merata dan miring ke kanan). |
| `rng.choice([...], n, p=[...])` | Fitur kategorik dengan peluang tiap kategori. |
| `log_price = 2.0 + 0.9*np.log(area) - ... + rng.normal(0, 0.2, n)` | Harga ditentukan pada skala **log** oleh luas, usia, jarak, kecamatan, dan kondisi, ditambah derau normal (simpangan baku 0,2). |
| `np.exp(log_price)` | Kembali ke skala asli: hubungan menjadi **multiplikatif** (setiap faktor mengalikan harga). |
| `df.loc[rng.choice(n, ...), "age_years"] = np.nan` | Sebagian nilai dikosongkan agar ada data hilang yang harus ditangani. |

### Langkah 2: EDA singkat dan data uji

```python title="project.py" linenums="42"
--8<-- "estimation/project.py:42:50"
```

`df.isna().sum()[lambda s: s > 0].to_dict()` menghitung data hilang per kolom lalu menyaring kolom yang punya data hilang (fungsi `lambda` sebagai syarat). `np.log(df["price"]).skew()` menghitung kemiringan harga setelah logaritma. `train_test_split` membagi tanpa `stratify` (target kontinu).

```text
--8<-- "expected/estimation/project.txt:1:3"
```

**Interpretasi.** 800 baris; data hilang: 40 pada `age_years` dan 32 pada `condition`. Harga median 505, rata-rata 557, maksimum 2068 juta; *skewness* harga 0,86 (miring ke kanan), dan setelah log -0,50 (agak miring ke kiri): log mengoreksi kemiringan tetapi sedikit berlebihan.

### Langkah 3: praproses dan pembungkus target log

**Konsep.** Praproses (imputasi, *scaling*, *one-hot*) dibangun sekali sebagai `prep`. Karena harga multiplikatif, kita mencoba juga memprediksi **log harga** lalu mengembalikannya ke skala asli. `TransformedTargetRegressor` melakukannya otomatis: ia melatih model pada `func(y)` dan menerapkan `inverse_func` pada prediksi.

```python title="project.py" linenums="52"
--8<-- "estimation/project.py:52:75"
```

| Kode | Penjelasan |
|---|---|
| `ColumnTransformer([...])` | Imputasi median + *scaling* untuk numerik; imputasi modus + *one-hot* untuk kategorik ([penjelasan](../fundamentals/sklearn-pipeline.md)). |
| `def with_log_target(est): return TransformedTargetRegressor(...)` | Fungsi yang membungkus pipeline (praproses + model) agar targetnya di-`np.log` saat pelatihan dan di-`np.exp` saat prediksi. |
| `candidates = {...}` | *Baseline* rata-rata serta Ridge dan Random Forest masing-masing dengan target asli dan target log. |

### Langkah 4: perbandingan dengan validasi silang

```python title="project.py" linenums="77"
--8<-- "estimation/project.py:77:84"
```

Perulangan menilai tiap kandidat pada data latih dengan tiga metrik; tanda minus mengembalikan galat negatif menjadi positif, dan `R²` ditampilkan apa adanya.

```text
--8<-- "expected/estimation/project.txt:5:10"
```

**Interpretasi.** *Baseline* RMSE 324,4 dan R² ≈ 0. Ridge target asli 147,4 (MAE 110,9; R² 0,792); Ridge target log 149,9 (MAE 106,6; R² 0,783); Random Forest target asli 150,6 (MAE 107,2); Random Forest target log 149,9 (MAE 105,7). Keempatnya sangat dekat. **Target log memperbaiki MAE sedikit tetapi tidak memperbaiki RMSE**; keduanya menimbang galat besar dan kecil secara berbeda. Pilih berdasarkan metrik yang sesuai tujuan bisnis.

### Langkah 5: optimasi ringan

```python title="project.py" linenums="86"
--8<-- "estimation/project.py:86:90"
```

Nama parameter `regressor__model__alpha` menelusuri lapisan bersarang: `TransformedTargetRegressor` menyimpan pipeline di atribut `regressor`, yang memiliki langkah `model`, yang memiliki parameter `alpha`. `search.best_estimator_` adalah model terbaik yang sudah dilatih ulang.

```text
--8<-- "expected/estimation/project.txt:12:12"
```

**Interpretasi.** `alpha` terbaik 10 (RMSE CV 147,5): perubahan kecil dibanding `alpha=1`, jadi Ridge tidak sensitif pada rentang ini.

### Langkah 6: evaluasi akhir dan galat per kecamatan

```python title="project.py" linenums="92"
--8<-- "estimation/project.py:92:98"
```

`err` adalah tabel berisi kecamatan, galat mutlak, dan galat **relatif** (galat dibagi harga sebenarnya, dalam persen). `groupby("district").agg(...)` menghitung rata-rata keduanya per kecamatan.

```text
--8<-- "expected/estimation/project.txt:14:20"
```

**Interpretasi.** Data uji (sekali): MAE 102,8; RMSE 149,5; R² 0,788 (juta Rp), konsisten dengan validasi silang (R² ≈ 0,78). Galat relatif rata-rata sekitar 20–21% di semua kecamatan (20,2% Barat, 21,5% Pusat, 21,0% Selatan, 20,7% Timur), meski galat absolut berbeda (MAE 84–124 juta) karena harga di Pusat lebih tinggi. Derau 20% pada skala log yang kita masukkan berarti galat relatif rata-rata sekitar 16% (kira-kira 0,8 × 0,2) bahkan bagi model yang sempurna mengenali hubungan sebenarnya. Galat 20–21% di sini cukup dekat dengan batas itu, sehingga ruang perbaikan model terbatas dan Ridge maupun Random Forest tidak perlu diperumit lebih jauh.

### Langkah 7: menyimpan dan memuat ulang

```python title="project.py" linenums="100"
--8<-- "estimation/project.py:100:103"
```

```text
--8<-- "expected/estimation/project.txt:21:21"
```

**Hasil.** Model disimpan dan prediksinya identik setelah dimuat (`np.allclose`).

??? example "Program lengkap (`project.py`)"
    ```python title="project.py" linenums="1"
    --8<-- "estimation/project.py"
    ```

## Refleksi

- Bandingkan galat dengan batas yang bisa dicapai: pada data sintetis ini kita tahu batasnya, pada data nyata gunakan model sederhana sebagai pembanding.
- Metrik relatif (galat persen) lebih bermakna untuk harga yang bervariasi besar.
- Pemantauan setelah deployment: lihat [lesson monitoring](saving-deployment.md).

## Latihan

1. Tambahkan `GradientBoostingRegressor` dan bandingkan dengan Ridge dan Random Forest.
2. Hitung galat relatif median (bukan rata-rata) per kecamatan.
3. Ubah derau dalam `make_housing_data` dari 0,2 menjadi 0,05. Bagaimana R² dan kesimpulan tentang model berubah?
4. Tulis ringkasan untuk pembaca non-teknis: seberapa akurat estimasi ini?

## Ringkasan

Proyek ini menunjukkan alur regresi lengkap: praproses di dalam *pipeline*, *baseline*, keputusan transformasi target dengan bukti, penyetelan ringan, evaluasi sekali di data uji, dan analisis galat per kelompok. Karena derau data dikendalikan, kita dapat melihat bahwa galat model sudah dekat dengan batas yang tak dapat dihindari.

Selesai dengan bab Estimasi. Lanjut ke bab [Forecasting](../forecasting/index.md).
