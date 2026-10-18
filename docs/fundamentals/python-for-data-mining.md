# Lingkungan Python untuk Data Mining

## Tujuan pembelajaran

- [x] Membuat *virtual environment* yang terisolasi untuk proyek.
- [x] Memasang pustaka yang dibutuhkan dan memeriksa bahwa semuanya berfungsi.
- [x] Membedakan skrip Python dan Jupyter Notebook, serta kapan memakainya.
- [x] Menerapkan kebiasaan dasar agar hasil eksperimen dapat diulang (*reproducible*).

## Mengapa perlu lingkungan terisolasi?

Setiap proyek bisa membutuhkan versi pustaka yang berbeda. Jika semuanya dipasang di satu tempat, pembaruan untuk proyek A dapat merusak proyek B. *Virtual environment* (venv) membuat salinan Python dan pustaka khusus untuk satu proyek. Ini juga membuat hasil lebih mudah direproduksi: Anda dapat mencatat versi persis yang dipakai.

## Langkah 1: buat dan aktifkan venv

=== "Windows (PowerShell)"

    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

Setelah aktif, *prompt* biasanya berawalan `(.venv)`. Semua `pip install` berikutnya hanya memengaruhi lingkungan ini.

## Langkah 2: pasang pustaka

Repositori ini menyediakan `requirements-ml.txt`:

```bash
pip install -r requirements-ml.txt
```

Pustaka inti: **NumPy** (array numerik), **pandas** (data tabular), **Matplotlib** dan **Seaborn** (visualisasi), **scikit-learn** (algoritma dan *pipeline*). `statsmodels` dipakai pada bab Forecasting dan `mlxtend` pada bab Aturan Asosiasi.

!!! note "Yang sudah dan belum diuji"
    Contoh pada materi ini dijalankan dengan Python 3.14, NumPy 2.5, pandas 3.0, dan scikit-learn 1.9. Paket `jupyter` tercantum sebagai opsional dan **tidak** diuji pada materi ini. Versi lain dapat menghasilkan angka atau tampilan yang sedikit berbeda.

## Langkah 3: periksa instalasi

**Tujuan.** Memastikan semua pustaka dapat diimpor sebelum mulai belajar, supaya galat instalasi tidak bercampur dengan galat kode materi.

**Kode** (`check_environment.py`):

```python title="check_environment.py" linenums="1"
--8<-- "fundamentals/check_environment.py"
```

**Penjelasan kode.**

| Baris | Arti |
|---|---|
| 1–2 | `import` memuat modul. `importlib` memungkinkan mengimpor pustaka dari nama berupa teks; `sys` memberi informasi tentang Python yang berjalan. |
| 4 | `sys.version_info` berisi versi Python; f-string (`f"..."`) menyisipkan nilai variabel ke dalam teks. |
| 6 | `for name in [...]` mengulang blok di bawahnya untuk tiap nama pustaka dalam *list*. Perhatikan `sklearn`: nama paket scikit-learn saat diimpor adalah `sklearn`. |
| 7–9 | `try` mencoba mengimpor. Bila berhasil, cetak `OK` beserta `__version__` (nomor versi pustaka). |
| 10–11 | `except ImportError` dijalankan hanya bila pustaka belum terpasang, sehingga program tidak berhenti dengan galat. |

**Hasil** pada lingkungan penulis:

```text
--8<-- "expected/fundamentals/check_environment.txt"
```

Jika ada pustaka berstatus `BELUM TERPASANG`, ulangi langkah 2 pada venv yang aktif. Nomor versi Anda boleh berbeda dari di atas; yang penting semua baris berstatus `OK`.

## Skrip atau Notebook?

| | Skrip `.py` | Jupyter Notebook |
|---|---|---|
| Cocok untuk | Kode yang dijalankan ulang, diuji, dan disimpan di Git | Eksplorasi interaktif, EDA, presentasi hasil |
| Kelebihan | Mudah dibaca ulang dan di-*diff*, urutan eksekusi jelas | Kode, grafik, dan catatan dalam satu dokumen |
| Risiko | Kurang interaktif | Sel dijalankan tidak berurutan sehingga hasil tidak dapat diulang |

Materi ini memakai skrip sebagai sumber contoh, sehingga setiap keluaran yang ditampilkan benar-benar berasal dari program yang dijalankan. Anda boleh menyalin kodenya ke notebook. Bila memakai notebook, jalankan *Restart & Run All* sebelum mempercayai hasilnya.

## Kebiasaan agar hasil dapat diulang

1. **Tetapkan seed acak**: `np.random.default_rng(42)` dan `random_state=42` pada fungsi scikit-learn.
2. **Catat versi pustaka** (`pip freeze > requirements-lock.txt`).
3. **Jangan mengubah data mentah**; simpan hasil olahan di berkas terpisah.
4. **Pisahkan data uji sejak awal** (lihat [Data Latih, Validasi, dan Uji](data-splitting.md)).

## Python minimum yang dipakai di materi

Bagian ini memperkenalkan potongan Python yang akan terus muncul. Jalankan `examples/fundamentals/python_basics.py` sambil membaca; tiap langkah di bawah mengutip bagian kodenya dan menunjukkan keluarannya.

### Langkah 1: variabel dan tipe data

**Konsep.** *Variabel* adalah nama yang menunjuk ke sebuah nilai. Tanda `=` berarti **penugasan** ("simpan nilai di kanan ke nama di kiri"), bukan "sama dengan" seperti dalam matematika. Tiap nilai memiliki **tipe**.

```python title="python_basics.py" linenums="1"
--8<-- "fundamentals/python_basics.py:1:7"
```

| Tipe | Contoh | Dipakai untuk |
|---|---|---|
| `str` (*string*) | `"Rina"` | Teks, nama kolom |
| `int` | `21` | Bilangan bulat, hitungan |
| `float` | `1.62` | Bilangan desimal, ukuran |
| `bool` | `True` | Hasil perbandingan; benar/salah |

`type(x).__name__` menampilkan nama tipe. Teks setelah `#` adalah **komentar**: dibaca manusia, diabaikan Python.

```text
--8<-- "expected/fundamentals/python_basics.txt:1:2"
```

**Mengapa penting.** Banyak galat pemula berasal dari tipe: misalnya `"21"` (teks) tidak dapat dijumlahkan seperti `21` (angka). Di pandas, kolom berisi angka yang terbaca sebagai teks menyebabkan hasil aneh.

### Langkah 2: list, indeks, dan irisan

**Konsep.** *List* adalah daftar berurutan dalam kurung siku. Setiap elemen memiliki **indeks** yang **dimulai dari 0**; indeks negatif menghitung dari belakang (`-1` = terakhir). *Irisan* (`[a:b]`) mengambil indeks `a` sampai **sebelum** `b`.

```python title="python_basics.py" linenums="9"
--8<-- "fundamentals/python_basics.py:9:15"
```

| Ekspresi | Hasil | Penjelasan |
|---|---|---|
| `scores[0]` | `80` | Elemen pertama |
| `scores[-1]` | `60` | Elemen terakhir |
| `scores[1:3]` | `[75, 90]` | Indeks 1 dan 2 (indeks 3 tidak ikut) |
| `scores.append(95)` | menambah `95` di akhir | Metode `append` mengubah list yang sudah ada |
| `len(scores)` | `5` | Jumlah elemen |

```text
--8<-- "expected/fundamentals/python_basics.txt:4:7"
```

### Langkah 3: dictionary

**Konsep.** *Dictionary* menyimpan pasangan **kunci → nilai** dalam kurung kurawal; nilai diambil dengan kuncinya, bukan nomor urut. Pola ini mirip satu baris data atau satu set parameter model (`{"C": 1, "gamma": 0.01}`).

```python title="python_basics.py" linenums="17"
--8<-- "fundamentals/python_basics.py:17:20"
```

```text
--8<-- "expected/fundamentals/python_basics.txt:9:10"
```

### Langkah 4: perulangan dan percabangan

**Konsep.** `for` mengulang blok kode untuk setiap elemen. `if`/`else` memilih jalur berdasarkan kondisi benar atau salah. **Indentasi** (4 spasi) menentukan kode mana yang termasuk blok; indentasi yang salah menyebabkan galat atau perilaku keliru.

```python title="python_basics.py" linenums="22"
--8<-- "fundamentals/python_basics.py:22:32"
```

Penjelasan: variabel `total` dimulai dari 0 lalu bertambah dengan setiap `s`; setelah perulangan selesai, `mean = total / len(scores)`. `mean >= 75` adalah **ekspresi boolean** yang bernilai `True` atau `False`. Pada format `%d` dan `%.1f`, angka dimasukkan ke dalam teks sebagai bilangan bulat dan desimal satu angka di belakang koma.

```text
--8<-- "expected/fundamentals/python_basics.txt:12:13"
```

Hasil: 80 + 75 + 90 + 60 + 95 = 400, dibagi 5 elemen = 80,0; karena 80,0 ≥ 75, status `lulus`.

### Langkah 5: fungsi

**Konsep.** *Fungsi* adalah blok kode bernama yang dapat dipakai ulang. Nilai yang diterimanya disebut **parameter** (`values`); nilai yang dikembalikan lewat `return` adalah **keluaran**. Fungsi mencegah penulisan kode yang sama berulang kali, dan hampir semua pustaka ML berupa kumpulan fungsi.

```python title="python_basics.py" linenums="35"
--8<-- "fundamentals/python_basics.py:35:41"
```

```text
--8<-- "expected/fundamentals/python_basics.txt:15:16"
```

`average(scores)` menghitung 400 / 5 = 80,0, sedangkan `average([10, 20])` = 15,0: fungsi yang sama dipakai untuk data berbeda.

### Langkah 6: kondisi boolean dan list comprehension

**Konsep.** Membandingkan nilai menghasilkan `True`/`False`. *List comprehension* `[ekspresi for x in daftar if kondisi]` membangun list baru secara ringkas. Pola penyaringan ini mirip *boolean mask* pada NumPy dan pandas.

```python title="python_basics.py" linenums="43"
--8<-- "fundamentals/python_basics.py:43:46"
```

```text
--8<-- "expected/fundamentals/python_basics.txt:18:19"
```

Baris pertama menunjukkan hasil perbandingan tiap nilai terhadap 75; baris kedua hanya menyimpan nilai yang memenuhi syarat.

### Langkah 7: membaca pesan galat

```python title="python_basics.py" linenums="48"
--8<-- "fundamentals/python_basics.py:48:52"
```

```text
--8<-- "expected/fundamentals/python_basics.txt:21:21"
```

`scores` hanya punya indeks 0 sampai 4, jadi `scores[10]` menimbulkan `IndexError`. `try`/`except` menangkap galat agar program lanjut; biasanya Anda tidak menangkapnya, tetapi **membaca** pesannya: nama galat (`IndexError`) menunjukkan jenis masalah dan baris yang disebut menunjukkan lokasinya.

??? example "Program lengkap (`python_basics.py`)"
    ```python title="python_basics.py" linenums="1"
    --8<-- "fundamentals/python_basics.py"
    ```

Bila bagian ini terasa terlalu cepat, pelajari tutorial resmi: [docs.python.org/3/tutorial](https://docs.python.org/3/tutorial/).

## Kesalahan umum

!!! warning "Hindari"
    - Memasang pustaka tanpa mengaktifkan venv (pustaka masuk ke Python sistem).
    - Menjalankan notebook tidak berurutan lalu menganggap hasilnya sah.
    - Tidak menetapkan seed sehingga angka berubah setiap dijalankan.
    - Mencampur versi pustaka antar proyek tanpa mencatatnya.

## Latihan

1. Buat venv baru dan jalankan `check_environment.py`. Apakah semua pustaka berstatus `OK`?
2. Tambahkan pemeriksaan `scipy` pada skrip tersebut.
3. Jalankan `np.random.default_rng(1).integers(0, 10, 5)` dua kali. Apakah hasilnya sama? Ganti seed menjadi 2 dan amati.

## Ringkasan

- Gunakan venv per proyek dan catat versi pustaka.
- Pasang pustaka dari `requirements-ml.txt`, lalu verifikasi dengan skrip pemeriksaan.
- Skrip cocok untuk kode yang diulang; notebook untuk eksplorasi, dengan disiplin urutan eksekusi.
- Seed acak dan pencatatan versi adalah dasar reproduksibilitas.

Berikutnya: [NumPy](numpy.md).
