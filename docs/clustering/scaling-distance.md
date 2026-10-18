# Scaling dan Ukuran Jarak

## Tujuan pembelajaran

- [x] Menjelaskan mengapa klastering berbasis jarak sangat dipengaruhi skala fitur.
- [x] Membandingkan ukuran jarak Euclidean, Manhattan, dan cosine.
- [x] Membandingkan `StandardScaler`, `MinMaxScaler`, dan `RobustScaler`.
- [x] Mengenali bahwa metrik internal dapat menyesatkan bila skala tidak diperbaiki.

## Jarak antar-sampel

Hampir semua algoritma klastering menggunakan jarak. Untuk dua sampel \(\mathbf{a}\) dan \(\mathbf{b}\) dengan \(p\) fitur:

$$
d_{\text{Euclid}} = \sqrt{\sum_{j=1}^{p}(a_j - b_j)^2}, \qquad
d_{\text{Manhattan}} = \sum_{j=1}^{p}|a_j - b_j|, \qquad
d_{\text{cosine}} = 1 - \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
$$

- **Euclidean**: jarak garis lurus; menghukum selisih besar pada satu fitur (kuadrat).
- **Manhattan**: jumlah selisih mutlak; lebih tahan terhadap satu fitur yang menyimpang.
- **Cosine**: hanya sudut, bukan panjang; cocok untuk teks atau data yang besarnya tidak penting.

Karena fitur dijumlahkan, **fitur berskala besar menenggelamkan fitur berskala kecil** pada jarak. Karena itu *scaling* wajib sebelum klastering berbasis jarak.

## Pilihan scaling

| Scaler | Rumus | Catatan |
|---|---|---|
| `StandardScaler` | \((x-\mu)/\sigma\) | Umum; sensitif pada *outlier* |
| `MinMaxScaler` | \((x-x_{\min})/(x_{\max}-x_{\min})\) | Rentang 0–1; sangat dipengaruhi *outlier* |
| `RobustScaler` | \((x-\text{median})/\text{IQR}\) | Tahan *outlier* |

## Implementasi langkah demi langkah

Dataset **Wine** bawaan scikit-learn (178 sampel, 13 fitur kimia; tiga kultivar). Label kultivar `y` tidak dipakai untuk klastering, hanya untuk validasi eksternal (ARI). Program `examples/clustering/scaling_distance.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: skala fitur

```python title="scaling_distance.py" linenums="8"
--8<-- "clustering/scaling_distance.py:8:15"
```

`X.agg(["min", "max"]).T` menghitung minimum dan maksimum tiap kolom, lalu `.T` memutarnya menjadi tabel dengan fitur sebagai baris (diberi nama `rng` di sini, bukan generator acak). `.loc["proline", "min"]` mengambil sel pada baris `proline` kolom `min`. `X.std().idxmax()` adalah **nama** fitur dengan simpangan baku terbesar.

```text
--8<-- "expected/clustering/scaling_distance.txt:1:2"
```

**Interpretasi.** `proline` berkisar 278–1680, sedangkan `hue` 0,48–1,71. Simpangan baku `proline` 314,9 vs `nonflavanoid_phenols` 0,12: perbedaan ribuan kali lipat.

### Langkah 2: jarak sebelum dan sesudah *scaling*

**Tujuan.** Melihat bagaimana satu fitur berskala besar menenggelamkan fitur lain dalam jarak Euclidean.

```python title="scaling_distance.py" linenums="17"
--8<-- "clustering/scaling_distance.py:17:26"
```

| Kode | Penjelasan |
|---|---|
| `a, b = X.iloc[0].to_numpy(), X.iloc[100].to_numpy()` | Dua sampel (baris 0 dan 100) sebagai vektor 13 angka. |
| `np.sqrt(((a - b) ** 2).sum())` | Jarak Euclidean: selisih tiap fitur dikuadratkan, dijumlahkan, diakarkan. |
| `contrib = ((a - b) ** 2) / ((a - b) ** 2).sum()` | **Kontribusi** tiap fitur terhadap jumlah kuadrat jarak (proporsi per fitur, total 1). |
| `contrib[X.columns.get_loc("proline")]` | Kontribusi fitur `proline` (`get_loc` memberi nomor posisi kolom). |
| `StandardScaler().fit_transform(X)` | Standardisasi: tiap fitur dikurangi rata-rata dan dibagi simpangan baku. |

```text
--8<-- "expected/clustering/scaling_distance.txt:4:5"
```

**Interpretasi.** Jarak mentah sampel 0 dan 100 adalah 356,3, dan **99,3%** dari kuadrat jarak itu berasal dari `proline` saja: dua belas fitur lain nyaris diabaikan. Setelah standardisasi jarak menjadi 5,32 dan kontribusi `proline` hanya 4,5%, jadi ketiga belas fitur ikut berperan.

### Langkah 3: ukuran jarak lain

```python title="scaling_distance.py" linenums="28"
--8<-- "clustering/scaling_distance.py:28:30"
```

`cdist([za], [zb], "euclidean")` dari SciPy menghitung jarak antar dua himpunan titik dengan metrik pilihan (`"cityblock"` = Manhattan; `"cosine"` = jarak sudut); `[0, 0]` mengambil satu-satunya nilai hasilnya.

```text
--8<-- "expected/clustering/scaling_distance.txt:7:7"
```

**Interpretasi.** Euclidean 5,32; Manhattan 16,37; cosine 1,040. Angkanya berbeda skala dan tidak dapat dibandingkan langsung; yang penting konsisten dipakai.

### Langkah 4: dampak *scaling* pada k-Means

**Tujuan.** Membandingkan k-Means (k=3) tanpa scaling dan dengan tiga jenis scaler, diukur dengan ARI terhadap kultivar (validasi eksternal) dan silhouette (validasi internal).

```python title="scaling_distance.py" linenums="32"
--8<-- "clustering/scaling_distance.py:32:39"
```

Perulangan memasangkan nama perlakuan dengan data hasilnya. `KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(data)` mengelompokkan data menjadi 3 klaster dan mengembalikan label tiap sampel. `adjusted_rand_score(y, labels)` membandingkan dengan kultivar; `silhouette_score(data, labels)` menilai kekompakan klaster pada ruang data yang sama.

```text
--8<-- "expected/clustering/scaling_distance.txt:9:13"
```

**Interpretasi.**

- Tanpa *scaling* ARI terhadap kultivar hanya **0,371**; dengan `StandardScaler` **0,897**, `MinMaxScaler` 0,854, `RobustScaler` 0,895.
- **Silhouette dapat menyesatkan**: tanpa *scaling*, silhouette justru **lebih tinggi** (0,571) daripada dengan *scaling* (0,285), padahal hasil kelompoknya jauh lebih buruk. Metrik internal dihitung pada ruang jarak yang sama; bila ruang itu didominasi satu fitur, klaster tampak "rapi" pada fitur itu saja. Selalu perbaiki skala sebelum menilai.

??? example "Program lengkap (`scaling_distance.py`)"
    ```python title="scaling_distance.py" linenums="1"
    --8<-- "clustering/scaling_distance.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Tidak melakukan *scaling* lalu mengandalkan silhouette yang tinggi.
    - Memakai `MinMaxScaler` pada data ber-*outlier* ekstrem.
    - *Fit* scaler pada seluruh data sebelum memeriksa kestabilan.
    - Mencampur fitur biner/kategorik dengan jarak Euclidean tanpa pertimbangan.

## Latihan

1. Tambahkan jarak `chebyshev` ke perbandingan.
2. Klaster hanya dengan `proline` dan `alcohol`; lakukan dengan dan tanpa *scaling*.
3. Beri bobot dua kali lipat pada fitur `flavanoids` setelah standardisasi. Bagaimana ARI berubah?

## Ringkasan

Jarak dijumlahkan lintas fitur, sehingga fitur berskala besar mendominasi. Lakukan *scaling* sebelum klastering, pilih scaler sesuai *outlier*, dan jangan percaya metrik internal pada data yang skalanya belum diperbaiki.

Berikutnya: [k-Means](kmeans.md).
