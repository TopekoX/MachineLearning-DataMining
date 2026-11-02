# Klaster (Clustering)

Klastering mengelompokkan data berdasarkan kemiripan **tanpa label**, misalnya segmentasi pelanggan. Ini adalah *unsupervised learning*: tidak ada jawaban benar yang tersedia sebagai acuan.

## Mulai dari sini: apa itu klastering?

**Klastering** adalah cara komputer mengelompokkan data yang mirip, tanpa diberi tahu lebih dulu kelompok yang benar.

!!! tip "Analogi"
    Bayangkan seorang pegawai toko menerima seribu kancing campur aduk. Tanpa petunjuk apa pun, ia menyortir kancing berdasarkan warna, ukuran, dan bahan. Kancing yang mirip ia kumpulkan dalam satu kotak. Itulah klastering: tidak ada kotak yang sudah ditentukan, kelompok muncul dari kemiripan data itu sendiri.

**Mengapa dipelajari?** Banyak masalah nyata tidak punya "kunci jawaban". Contohnya mengelompokkan pelanggan menurut kebiasaan belanja agar promosi bisa dibuat berbeda untuk tiap kelompok.

**Perbedaan dengan klasifikasi.** Klasifikasi belajar dari contoh yang sudah berlabel (misalnya "spam" atau "bukan spam"). Klastering tidak punya label, sehingga hasilnya harus kita nilai dan tafsirkan sendiri.

### Peta belajar

Urutan pelajaran mengikuti pertanyaan yang muncul saat bekerja dengan klastering:

1. **Apa yang sedang kita lakukan, dan data seperti apa yang disiapkan?** → pelajaran 1
2. **Bagaimana komputer mengukur "mirip"?** → pelajaran 2 (jarak dan *scaling*)
3. **Bagaimana cara mengelompokkannya?** → pelajaran 3 sampai 5 (tiga algoritma dengan cara berpikir berbeda)
4. **Berapa kelompok yang sebaiknya dibuat, dan apakah hasilnya baik?** → pelajaran 6 dan 7
5. **Apa arti tiap kelompok, dan bagaimana memakainya?** → pelajaran 8 dan 9
6. **Bisakah semuanya dirangkai dalam satu proyek?** → pelajaran 10

Setiap pelajaran dibuka dengan bagian **Gambaran singkat** yang menjelaskan topiknya dengan bahasa sehari-hari sebelum masuk ke kode dan rumus.

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
