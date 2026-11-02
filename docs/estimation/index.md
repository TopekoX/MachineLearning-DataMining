# Estimasi (Regresi)

Estimasi atau regresi memprediksi **nilai numerik kontinu**, misalnya harga rumah atau konsumsi listrik. Ini adalah *supervised learning* dengan target berupa angka.

## Mulai dari sini: apa itu estimasi (regresi)?

**Regresi** adalah cara komputer menebak sebuah **angka** (bukan kategori) setelah belajar dari contoh yang jawabannya diketahui.

!!! tip "Analogi"
    Agen properti menaksir harga rumah dari luas, jumlah kamar, dan lokasi. Jawabannya angka, misalnya 850 juta, dan kita menilai seberapa dekat tebakan dengan harga sebenarnya.

**Mengapa dipelajari?** Banyak keputusan bergantung pada angka: harga, konsumsi listrik, lama pengiriman.

### Peta belajar

1. **Apa masalahnya dan bagaimana data diperiksa?** → alur kerja dan data
2. **Model paling dasar** → *baseline* dan regresi linear
3. **Bagaimana mengukur kesalahan?** → metrik dan residual
4. **Bagaimana mencegah model terlalu rumit?** → Ridge dan Lasso
5. **Model yang bisa menangkap hubungan melengkung** → KNN, pohon, dan Random Forest
6. **Bagaimana menguji, menyetel, dan membandingkan?** → validasi, optimasi, perbandingan
7. **Bagaimana memakainya?** → simpan, deploy, lalu proyek

Setiap pelajaran dibuka dengan **Gambaran singkat** yang menjelaskan topiknya dengan bahasa sehari-hari.

## Tujuan pembelajaran

- Membedakan regresi dari klasifikasi dan memilih metrik yang sesuai.
- Membangun model regresi linear dan non-linear dengan *pipeline* tanpa kebocoran data.
- Mengevaluasi dengan MAE, RMSE, R², dan analisis residual.
- Mengendalikan *overfitting* dengan regularisasi dan validasi silang.

## Prasyarat

Bab [Dasar-dasar](../fundamentals/index.md) dan sebaiknya bab [Klasifikasi](../classification/index.md): *pipeline*, pembagian data, validasi silang, dan penyetelan dibahas di sana, sehingga bab ini hanya membahas perbedaannya untuk regresi.

## Tahap alur kerja yang relevan

| Tahap | Status | Catatan |
|---|---|---|
| Definisi masalah sampai praproses | Wajib | Periksa distribusi target dan *outlier* |
| Rekayasa fitur | Opsional | Fitur polinomial, transformasi log |
| Pembagian data | Wajib | Train/test dan validasi silang acak (tanpa *stratify*) |
| Pemilihan, pelatihan model | Wajib | Mulai dari *baseline* rata-rata |
| Evaluasi | Wajib | MAE, RMSE, R², plot residual |
| Optimasi | Wajib | Regularisasi dan pencarian hyperparameter |
| Penyimpanan model | Wajib | `joblib` |
| Deployment dan monitoring | Opsional | Pemeriksaan rentang masukan, pemantauan galat dan bias |

## Pelajaran

| # | Pelajaran | Isi utama |
|---|---|---|
| 1 | [Alur Kerja dan Data](workflow-and-data.md) | Regresi vs klasifikasi, dataset Diabetes, EDA |
| 2 | [Baseline dan Regresi Linear](linear-regression.md) | *Baseline*, kuadrat terkecil, koefisien |
| 3 | [Ridge dan Lasso](ridge-lasso.md) | Regularisasi L2 dan L1 |
| 4 | [KNN, Tree, Random Forest Regressor](nonlinear-regressors.md) | Model non-linear dan batasannya |
| 5 | [Perbandingan Model](model-comparison.md) | Validasi silang berulang, interpretasi |
| 6 | [Metrik dan Analisis Residual](metrics-residuals.md) | MAE, RMSE, R², plot residual |
| 7 | [Validasi Silang dan Optimasi](validation-tuning.md) | Grid/Random Search, *nested CV* |
| 8 | [Simpan, Deploy, Monitor](saving-deployment.md) | Ekstrapolasi, bias, pemantauan |
| 9 | [Proyek: Harga Rumah](project.md) | Alur end-to-end pada data sintetis |

## Data yang dipakai

- **Diabetes**, data publik bawaan scikit-learn, untuk pelajaran 1–8.
- **Data harga rumah sintetis** untuk proyek (ditandai sebagai sintetis).

Contoh kode ada di `examples/estimation/`; keluaran yang ditampilkan berasal dari program yang dijalankan.
