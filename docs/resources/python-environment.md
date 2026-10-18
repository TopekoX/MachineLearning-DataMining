# Menjalankan Situs dan Contoh Kode

Instalasi Python dan pustaka untuk mengikuti materi dijelaskan di [Lingkungan Python untuk Data Mining](../fundamentals/python-for-data-mining.md). Halaman ini merangkum perintah praktis.

## Menjalankan contoh kode

Dari root repositori, dengan venv aktif dan `pip install -r requirements-ml.txt`:

```bash
python examples/fundamentals/numpy_basics.py
python tools/run_examples.py              # jalankan semua contoh dan bandingkan dengan keluaran tercatat
python tools/run_examples.py --update     # tulis ulang keluaran tercatat (setelah mengubah contoh)
```

Skrip yang membuat gambar menulis ke `docs/assets/img/`, sehingga jalankan dari root repositori.

## Membangun situs dokumentasi

```bash
pip install -r requirements.txt
mkdocs serve              # pratinjau di http://127.0.0.1:8000
mkdocs build --strict     # periksa tautan dan peringatan
```

!!! note "Akses jaringan"
    Rumus matematika dirender oleh MathJax dari CDN, sehingga membutuhkan koneksi internet.
