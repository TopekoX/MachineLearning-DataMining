# Perbandingan Algoritma dan Optimasi Parameter

## Tujuan pembelajaran

- [x] Membandingkan algoritma klastering pada data yang sama.
- [x] Mengoptimasi parameter tanpa label dan memahami bahayanya.
- [x] Menyadari bahwa metrik internal dapat memilih hasil yang tidak berguna.

## Ringkasan algoritma

| Aspek | k-Means | Agglomerative | DBSCAN | Gaussian Mixture |
|---|---|---|---|---|
| Prinsip | Pusat dan jarak | Penggabungan bertahap | Kepadatan | Campuran distribusi normal (probabilitas) |
| Jumlah klaster | Ditentukan | Ditentukan atau dari ambang | Otomatis | Ditentukan |
| Bentuk klaster | Bulat | Bergantung *linkage* | Bebas | Elips |
| *Noise* | Tidak ada | Tidak ada | Ya | Tidak ada |
| Hyperparameter kunci | `n_clusters`, `n_init` | `linkage`, `n_clusters` | `eps`, `min_samples` | `n_components`, `covariance_type` |
| *Scaling* | Wajib | Wajib | Wajib | Disarankan |

## Optimasi parameter tanpa label

Karena tidak ada label, parameter disetel dengan metrik internal (misalnya silhouette), kestabilan, atau kebutuhan bisnis. Bahaya utamanya: metrik internal dapat menyukai hasil yang secara praktis buruk, misalnya sedikit klaster dengan banyak titik dibuang sebagai *noise*.

## Implementasi langkah demi langkah

Wine terstandardisasi. DBSCAN disetel pada grid `eps` × `min_samples` dengan silhouette (dihitung hanya pada titik non-*noise*). Label kultivar dipakai **hanya sebagai validasi eksternal**, bukan untuk menyetel. Program `examples/clustering/comparison_tuning.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: fungsi metrik internal yang menangani *noise*

```python title="comparison_tuning.py" linenums="14"
--8<-- "clustering/comparison_tuning.py:14:23"
```

Metrik internal seperti silhouette tidak dapat menilai titik *noise* (label -1). Fungsi `internal(labels)` membuang titik *noise* dengan *mask* (`mask = labels != -1`), dan mengembalikan `NaN` bila klaster tersisa kurang dari dua (silhouette tidak terdefinisi). Metrik dihitung hanya pada data non-*noise* `Z[mask]`.

### Langkah 2: menyetel parameter DBSCAN tanpa label

**Konsep.** Parameter disetel dengan **metrik internal** (silhouette) karena tidak ada label.

```python title="comparison_tuning.py" linenums="26"
--8<-- "clustering/comparison_tuning.py:26:35"
```

Dua perulangan bersarang mencoba semua pasangan `eps` (5 nilai) dan `min_samples` (3 nilai) = 15 kombinasi. `best` menyimpan tuple `(silhouette, eps, min_samples, label)` kombinasi terbaik sejauh ini; `best is None or sil > best[0]` memperbaruinya bila ditemukan silhouette lebih tinggi.

```text
--8<-- "expected/clustering/comparison_tuning.txt:1:1"
```

### Langkah 3: membandingkan empat algoritma

```python title="comparison_tuning.py" linenums="37"
--8<-- "clustering/comparison_tuning.py:37:50"
```

| Kode | Penjelasan |
|---|---|
| `GaussianMixture(n_components=3, random_state=SEED).fit_predict(Z)` | Model campuran Gaussian: tiap klaster dimodelkan sebagai distribusi normal; `fit_predict` memberi klaster dengan probabilitas tertinggi. |
| `"DBSCAN (parameter di atas)": best[3]` | Label DBSCAN dari kombinasi terbaik (elemen ke-4 tuple `best`). |
| `adjusted_rand_score`, `normalized_mutual_info_score` | Validasi eksternal: ARI dan NMI terhadap kultivar. |
| `100 * np.mean(lab == -1)` | Persentase titik *noise*. |

```text
--8<-- "expected/clustering/comparison_tuning.txt:3:8"
```

**Interpretasi.**

- **DBSCAN terpilih oleh silhouette**: `eps=2,5`, `min_samples=10` (silhouette 0,323; 2 klaster; 18% *noise*). Ia punya silhouette **tertinggi** dan Davies-Bouldin **terendah** (1,196) pada tabel, tetapi ARI terhadap kultivar hanya **0,424** (NMI 0,531). Metrik internal hanya menilai titik non-*noise* dan menyukai sedikit klaster padat sehingga memberi skor tinggi pada hasil yang membuang 18% data dan hanya menemukan 2 klaster, bukan 3. Metrik internal bukan bukti bahwa hasil itu paling berguna.
- **k-Means (k=3)**: ARI 0,897; silhouette 0,285; Davies-Bouldin 1,389.
- **Gaussian Mixture (k=3)**: hasilnya sama dengan k-Means, kemungkinan karena klaster Wine kira-kira bulat dan terpisah cukup baik (penjelasan ini tidak diuji secara khusus).
- **Agglomerative Ward (k=3)**: ARI 0,790.

### Langkah 4: kesepakatan antar-algoritma

```python title="comparison_tuning.py" linenums="52"
--8<-- "clustering/comparison_tuning.py:52:55"
```

```text
--8<-- "expected/clustering/comparison_tuning.txt:10:11"
```

**Interpretasi.** ARI antara k-Means dan Ward 0,853 (mirip tetapi tidak identik) dan antara k-Means dan Gaussian Mixture 1,000. **Kesepakatan antar-algoritma** adalah petunjuk tambahan: bila beberapa algoritma berbeda menemukan pengelompokan yang mirip, struktur itu lebih mungkin nyata. Kesimpulan bergantung data: pada Wine, algoritma berbasis pusat unggul; pada bulan sabit ([DBSCAN](dbscan.md)) justru DBSCAN. Pilih algoritma berdasarkan bentuk data dan tujuan, bukan peringkat satu metrik.

??? example "Program lengkap (`comparison_tuning.py`)"
    ```python title="comparison_tuning.py" linenums="1"
    --8<-- "clustering/comparison_tuning.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memilih algoritma atau parameter hanya dengan silhouette tertinggi tanpa memeriksa jumlah *noise* dan jumlah klaster.
    - Membandingkan metrik internal antar-algoritma seolah skalanya setara.
    - Menyetel parameter memakai label lalu menyebutnya "tak terawasi".
    - Mengabaikan kestabilan dan interpretasi.

## Latihan

1. Perluas grid DBSCAN dan tambahkan syarat noise < 10%. Apakah pilihan berubah?
2. Tambahkan `covariance_type="diag"` pada Gaussian Mixture.
3. Bandingkan hasil keempat metode setelah mereduksi dimensi dengan PCA 5 komponen.

## Ringkasan

Bandingkan algoritma pada data yang sama dengan beberapa metrik dan kesepakatan antar-metode. Optimasi parameter tanpa label rawan menyesatkan; periksa selalu jumlah klaster, *noise*, kestabilan, dan kegunaan.

Berikutnya: [Profil dan Interpretasi Klaster](profiling.md).
