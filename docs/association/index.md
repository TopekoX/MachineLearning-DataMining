# Aturan Asosiasi (Association Rule Mining)

Aturan asosiasi menemukan **barang yang sering muncul bersama** dalam transaksi, misalnya barang yang sering dibeli bersamaan (*market basket analysis*). Ini adalah *unsupervised learning* berbasis pola: tidak ada target yang diprediksi.

## Mulai dari sini: apa itu aturan asosiasi?

**Aturan asosiasi** menemukan barang yang **sering muncul bersama**, tanpa ada label yang harus ditebak.

!!! tip "Analogi"
    Kasir minimarket melihat ribuan struk dan menyadari banyak orang yang membeli roti juga membeli mentega. Informasi itu dipakai untuk menata rak atau membuat paket hemat.

**Mengapa dipelajari?** Pola belanja membantu rekomendasi produk, penataan toko, dan promosi.

### Peta belajar

1. **Seperti apa datanya?** → transaksi dan pengodean
2. **Bagaimana menemukan kombinasi yang sering?** → Apriori dan FP-Growth
3. **Bagaimana mengukur kekuatan sebuah aturan?** → *support*, *confidence*, *lift*
4. **Bagaimana menyaring aturan yang berguna?** → ambang dan keterbatasan
5. **Bagaimana merangkai semuanya?** → proyek minimarket

Setiap pelajaran dibuka dengan **Gambaran singkat** yang menjelaskan topiknya dengan bahasa sehari-hari.

## Tujuan pembelajaran

- Memahami transaksi, itemset, dan aturan asosiasi.
- Menyiapkan dan mengodekan data transaksi.
- Menerapkan Apriori dan FP-Growth, dan menghitung serta menafsirkan *support*, *confidence*, dan *lift*.
- Memilih ambang, menyaring aturan, menguji kestabilan, dan mengenali keterbatasan.

## Prasyarat

[Dasar-dasar](../fundamentals/index.md), khususnya [pandas](../fundamentals/pandas.md) dan probabilitas dasar pada [Statistik](../fundamentals/statistics.md).

!!! note "Pustaka tambahan"
    Bab ini memakai `mlxtend` (`TransactionEncoder`, `apriori`, `fpgrowth`, `association_rules`). Pasang lewat `requirements-ml.txt`.

## Gambaran konsep

Sebuah aturan berbentuk \(A \Rightarrow B\): bila itemset \(A\) muncul dalam sebuah transaksi, \(B\) cenderung muncul juga. Kekuatannya diukur dengan:

$$
\text{lift}(A \Rightarrow B) = \frac{P(A \cap B)}{P(A)\,P(B)}
$$

Lift > 1 berarti \(A\) dan \(B\) muncul bersama lebih sering daripada kebetulan.

## Tahap alur kerja dan alternatifnya

Bab ini sengaja **tidak** memaksakan tahap yang tidak berlaku:

| Tahap | Status | Catatan |
|---|---|---|
| Definisi masalah, pengumpulan data | Wajib | Format transaksi |
| EDA | Berbeda | Frekuensi barang, ukuran keranjang |
| Praproses | Berbeda | Normalisasi, hapus duplikat, matriks *one-hot* |
| Rekayasa fitur | Tidak berlaku | Opsional: mengelompokkan barang |
| Pembagian data | Tidak berlaku | **Pengganti:** uji kestabilan aturan pada bagian transaksi lain |
| Penemuan pola | Wajib | Itemset sering, lalu aturan |
| Evaluasi | Berbeda | *Support*, *confidence*, *lift*; bukan *accuracy* |
| Optimasi | Opsional | Menyapu ambang *support* dan *confidence* |
| Penyimpanan | Opsional | **Pengganti model:** simpan tabel aturan (CSV) |
| Deployment | Opsional | **Pengganti API:** tinjauan merchandising, bundling, rekomendasi; pantau berkala |

## Pelajaran

| # | Pelajaran | Isi utama |
|---|---|---|
| 1 | [Data Transaksi dan Pengodean](transactions-encoding.md) | Transaksi, *support*, *one-hot* |
| 2 | [Itemset Sering dan Apriori](apriori.md) | Anti-monoton, `min_support`, ledakan kombinatorial |
| 3 | [FP-Growth](fpgrowth.md) | FP-tree, hasil sama dengan Apriori |
| 4 | [Support, Confidence, Lift](rules-metrics.md) | Aturan, ukuran, bahaya barang populer |
| 5 | [Penyaringan, Ambang, Keterbatasan](filtering-thresholds.md) | Ambang, kestabilan, keterbatasan |
| 6 | [Proyek: Keranjang Belanja](project.md) | Alur lengkap sampai tabel aturan |

## Data yang dipakai

Transaksi **sintetis** (dibangkitkan dengan seed tetap, 20 barang, dengan pola yang sengaja ditanam). Contoh kode ada di `examples/association/`; keluaran yang ditampilkan berasal dari program yang dijalankan.
