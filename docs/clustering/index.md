# Klaster (Clustering)

Klastering mengelompokkan data berdasarkan kemiripan **tanpa label**, misalnya segmentasi pelanggan. Ini adalah *unsupervised learning*: tidak ada jawaban benar yang tersedia sebagai acuan.

## Tujuan pembelajaran

- Menjelaskan perbedaan pembelajaran terawasi dan tak terawasi, serta mengapa train/test terawasi tidak berlaku langsung.
- Menerapkan k-Means, DBSCAN, dan Agglomerative Clustering dengan *scaling* yang benar.
- Memilih jumlah klaster dan menilai kualitas dengan metrik internal, kestabilan, dan validasi eksternal bila ada.
- Menafsirkan dan memberi profil pada klaster, lalu menyimpan pipeline yang dapat dipakai ulang.

## Prasyarat

[Dasar-dasar](../fundamentals/index.md), terutama [Praproses Data](../fundamentals/data-preprocessing.md) (*scaling*) dan [EDA](../fundamentals/eda.md).

## Tahap alur kerja yang relevan

| Tahap | Status | Catatan |
|---|---|---|
| Definisi masalah | Wajib | Tujuan eksploratif; apa gunanya segmen? |
| EDA dan praproses | Wajib | ***Scaling* wajib**; transformasi log untuk fitur miring |
| Rekayasa fitur | Opsional | Seleksi fitur, PCA |
| Pembagian data | Tidak berlaku | Tidak ada data uji berlabel; periksa kestabilan |
| Pemilihan metode dan pelatihan | Wajib | Sesuaikan algoritma dengan bentuk data |
| Evaluasi | Berbeda | Silhouette, Davies-Bouldin, profil; ARI/NMI hanya bila ada label acuan |
| Optimasi | Opsional | Memilih \(k\) atau parameter DBSCAN |
| Penyimpanan | Opsional | Pipeline + nama segmen |
| Deployment dan monitoring | Opsional | Penempatan data baru, pemantauan jarak dan proporsi |

## Pelajaran

| # | Pelajaran | Isi utama |
|---|---|---|
| 1 | [Konsep dan Persiapan Data](concepts-data.md) | Tanpa label, kestabilan, EDA |
| 2 | [Scaling dan Ukuran Jarak](scaling-distance.md) | Euclidean/Manhattan/cosine, scaler |
| 3 | [k-Means](kmeans.md) | Inertia, inisialisasi, keterbatasan |
| 4 | [DBSCAN](dbscan.md) | Kepadatan, `eps`, *noise* |
| 5 | [Agglomerative Clustering](agglomerative.md) | *Linkage*, dendrogram |
| 6 | [Memilih Jumlah Klaster dan Validasi](choose-k-validation.md) | Elbow, silhouette, kestabilan, ARI |
| 7 | [Perbandingan dan Optimasi Parameter](comparison-tuning.md) | Algoritma, bahaya metrik internal |
| 8 | [Profil dan Interpretasi](profiling.md) | Profil, indeks, penamaan |
| 9 | [Menyimpan Pipeline dan Memantau](saving-pipeline.md) | Penempatan data baru, pergeseran |
| 10 | [Proyek: Segmentasi Pelanggan](project.md) | Alur lengkap data RFM sintetis |

## Data yang dipakai

- **Wine** (data publik bawaan scikit-learn) untuk membandingkan algoritma dan memilih \(k\); label kultivar hanya untuk validasi eksternal.
- **Data sintetis** (pelanggan, bulan sabit, RFM) untuk segmentasi dan keterbatasan algoritma; selalu ditandai sebagai sintetis.

Contoh kode ada di `examples/clustering/`; keluaran yang ditampilkan berasal dari program yang dijalankan.
