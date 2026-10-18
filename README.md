# Data Mining dan Machine Learning dengan Python

Materi belajar berbahasa Indonesia tentang alur kerja *data mining* dan *machine learning* dengan Python dan scikit-learn, disajikan sebagai situs [MkDocs](https://www.mkdocs.org/) (tema [Material](https://squidfunk.github.io/mkdocs-material/)). Setiap pelajaran menjelaskan konsep, memecah kode menjadi langkah-langkah, menampilkan keluaran dari program yang benar-benar dijalankan, dan menafsirkan hasilnya.

## Isi materi

| Bab | Isi | Pelajaran |
|---|---|---|
| [Dasar-dasar](docs/fundamentals/index.md) | Python, NumPy, pandas, visualisasi, statistik, EDA, praproses, rekayasa fitur, pembagian data, metrik, pipeline, overfitting dan kebocoran data | 12 |
| [Klasifikasi](docs/classification/index.md) | Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, perbandingan model, evaluasi, kelas tidak seimbang, tuning, penyimpanan, proyek churn | 14 |
| [Estimasi (Regresi)](docs/estimation/index.md) | Regresi linear, Ridge/Lasso, KNN/Tree/Forest regressor, metrik dan residual, tuning, proyek harga rumah | 9 |
| [Forecasting](docs/forecasting/index.md) | Komponen deret waktu, fitur lag, validasi kronologis, exponential smoothing, ARIMA/SARIMA, interval prediksi, proyek penjualan | 11 |
| [Klaster](docs/clustering/index.md) | Scaling dan jarak, k-Means, DBSCAN, Agglomerative, memilih k, profil segmen, proyek segmentasi | 10 |
| [Aturan Asosiasi](docs/association/index.md) | Transaksi, Apriori, FP-Growth, support/confidence/lift, penyaringan, proyek keranjang belanja | 6 |

Peta belajar dan sumber belajar ada di `docs/roadmap/` dan `docs/resources/`.

## Struktur repositori

```
mkdocs.yml                 identitas situs, tema, navigasi, ekstensi Markdown
requirements.txt           pustaka untuk membangun situs
requirements-ml.txt        pustaka untuk menjalankan contoh kode materi
docs/                      seluruh halaman materi (per bab) dan aset gambar
  assets/img/              gambar yang dihasilkan skrip contoh
  javascripts/, stylesheets/, assets/vendor/mermaid/   MathJax, Mermaid, dan CSS situs
examples/<bab>/            skrip contoh yang dikutip oleh halaman materi
examples/expected/         keluaran tercatat dari tiap skrip contoh
tools/run_examples.py      menjalankan semua contoh dan membandingkan keluarannya
CLAUDE.md                  aturan penulisan materi
PRD.md, TECHNICAL_DESIGN.md, PROJECT_AUDIT.md, IMPLEMENTATION_PLAN.md, VALIDATION_REPORT.md   dokumen perencanaan
```

## Menyiapkan lingkungan

Contoh kode diuji dengan Python 3.14 (NumPy 2.5, pandas 3.0, scikit-learn 1.9, statsmodels 0.15, mlxtend 0.25). Versi lain dapat memberi angka atau tampilan yang sedikit berbeda.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # membangun situs
pip install -r requirements-ml.txt     # menjalankan contoh kode
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, lalu perintah `pip install` yang sama. Paket `jupyter` di `requirements-ml.txt` bersifat opsional dan tidak diuji.

## Menjalankan situs

```powershell
mkdocs serve              # pratinjau di http://127.0.0.1:8000
mkdocs build --strict     # membangun ke site/; gagal bila ada tautan atau peringatan bermasalah
```

Rumus matematika dirender MathJax dari CDN (`cdn.jsdelivr.net`, versi 3.2.2), sehingga rumus memerlukan koneksi internet. Mermaid sudah disertakan di repositori. Build menampilkan kotak pemberitahuan "MkDocs 2.0" dari Material; itu pengumuman upstream, bukan peringatan proyek.

## Menjalankan contoh kode

Jalankan dari root repositori (beberapa skrip menulis gambar ke `docs/assets/img/`):

```powershell
python examples/fundamentals/numpy_basics.py     # satu contoh
python tools/run_examples.py                      # semua contoh, bandingkan dengan keluaran tercatat
python tools/run_examples.py classification       # satu bab saja
python tools/run_examples.py --update             # perbarui keluaran tercatat setelah mengubah contoh
```

Skrip contoh berada di `examples/<bab>/` dan memakai seed tetap, sehingga keluarannya dapat diulang. Berkas pembantu data (`customer_data.py`, `forecast_utils.py`, `segment_data.py`, `basket_data.py`) harus tetap satu folder dengan skrip yang mengimpornya.

## Data

- **Data publik** bawaan pustaka: Breast Cancer, Wine, Diabetes (scikit-learn) dan CO2 Mauna Loa (statsmodels). Tidak ada unduhan yang diperlukan.
- **Data sintetis** (churn, harga rumah, penjualan, pelanggan, transaksi) dibangkitkan program dan selalu ditandai sebagai sintetis pada pelajaran yang memakainya. Hasilnya menggambarkan mekanisme, bukan dunia nyata.

Daftar dan sumber dataset ada di `docs/resources/datasets.md`.

## Menambah atau mengubah pelajaran

1. Tulis skrip contoh di `examples/<bab>/` dan jalankan `python tools/run_examples.py --update <bab>` untuk mencatat keluarannya.
2. Buat halaman `docs/<bab>/<nama>.md` yang diawali satu `# Judul`. Kutip kode dengan potongan baris, misalnya `--8<-- "classification/knn.py:14:20"`, dan keluaran dengan `--8<-- "expected/classification/knn.txt:1:8"`.
3. Daftarkan halaman di `nav:` pada `mkdocs.yml` lalu jalankan `mkdocs build --strict`.
4. Jalankan ulang `python tools/run_examples.py` bila skrip berubah; nomor baris yang dikutip harus tetap sesuai.

Konvensi penulisan (bahasa Indonesia, istilah Inggris dalam tanda kurung, struktur pelajaran, aturan kebocoran data) ada di `CLAUDE.md`.

## Lisensi

[MIT License](LICENSE). Pustaka Mermaid di `docs/assets/vendor/mermaid/` juga berlisensi MIT dan menyertakan berkas `LICENSE`-nya sendiri.
