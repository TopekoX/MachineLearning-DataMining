# Data Transaksi dan Pengodean

## Tujuan pembelajaran

- [x] Memahami data transaksi dan analisis keranjang belanja (*market basket analysis*).
- [x] Membersihkan data transaksi dan mengubahnya menjadi matriks *one-hot*.
- [x] Menghitung *support* sederhana dari data dan membaca ringkasan keranjang.

## Konsep

Analisis keranjang belanja mencari **barang yang sering dibeli bersamaan**. Satuannya adalah **transaksi** (satu keranjang atau satu struk) yang berisi himpunan barang (*item*); sekumpulan barang disebut ***itemset***.

| Istilah | Arti |
|---|---|
| Transaksi | Satu keranjang: himpunan barang yang dibeli bersamaan |
| Itemset | Himpunan satu atau lebih barang, misalnya {roti, mentega} |
| *Support* | Proporsi transaksi yang memuat sebuah itemset |

$$
\text{support}(X) = \frac{\text{jumlah transaksi yang memuat } X}{\text{jumlah seluruh transaksi}}
$$

Berbeda dengan bab lain, tidak ada target yang diprediksi dan tidak ada pembagian latih/uji wajib. Tujuannya menemukan pola; cara menilai kualitas dibahas di [lesson aturan](rules-metrics.md) dan [lesson penyaringan](filtering-thresholds.md).

## Persiapan data

1. **Satu baris per barang per transaksi** pada data mentah (format panjang), lalu dikelompokkan menjadi daftar barang per transaksi.
2. **Normalisasi nama barang**: huruf kecil, hapus spasi berlebih, samakan ejaan.
3. **Hapus duplikat barang dalam satu transaksi**: aturan asosiasi klasik hanya peduli ada atau tidaknya barang, bukan jumlahnya.
4. **Buang transaksi kosong**, dan pertimbangkan transaksi berisi satu barang (tidak menyumbang pasangan, tetapi memengaruhi *support*).
5. **Tentukan tingkat barang**: "roti tawar" dan "roti gandum" dipisah atau digabung menjadi "roti"? Tingkat yang terlalu rinci membuat *support* sangat kecil.
6. **Pengodean** ke matriks transaksi × barang bernilai benar/salah (`TransactionEncoder`).

## Implementasi langkah demi langkah

Bagian 1 memakai 5 transaksi kecil agar dapat dihitung tangan; bagian 3–4 memakai **2000 transaksi sintetis** dari `basket_data.py` (20 barang, dibangkitkan dengan seed tetap, dengan beberapa pola yang sengaja ditanam). Karena sintetis, pola yang ditemukan tidak mencerminkan perilaku belanja nyata. Program `examples/association/transactions_encoding.py`; program lengkap ada di akhir bagian ini.

??? example "Kode pembangkit data (`basket_data.py`)"
    ```python title="basket_data.py"
    --8<-- "association/basket_data.py"
    ```

### Langkah 1: contoh kecil dan matriks *one-hot*

**Konsep.** Algoritma asosiasi bekerja pada tabel **transaksi × barang** bernilai benar/salah: sel `True` berarti barang itu ada di keranjang itu. `TransactionEncoder` membuatnya dari daftar-daftar barang.

```python title="transactions_encoding.py" linenums="9"
--8<-- "association/transactions_encoding.py:9:15"
```

| Kode | Penjelasan |
|---|---|
| `tiny = [["roti", "mentega"], ...]` | Daftar transaksi; tiap transaksi adalah *list* barang. |
| `TransactionEncoder()` | Pengubah dari daftar barang ke matriks (mirip *encoder* scikit-learn). |
| `te.fit(tiny).transform(tiny)` | `fit` mempelajari semua barang unik; `transform` menghasilkan array `True`/`False`. Nama kolom ada di `te.columns_`. |
| `onehot.astype(int)` | Menampilkan `True`/`False` sebagai 1/0 agar mudah dibaca. |
| `onehot["roti"].sum()`, `.mean()` | Jumlah dan proporsi keranjang yang memuat roti = *support*. |
| `(onehot["roti"] & onehot["mentega"])` | `&` memilih keranjang yang memuat **kedua** barang. |

```text
--8<-- "expected/association/transactions_encoding.txt:1:9"
```

**Hasil.** Roti muncul pada 4 dari 5 keranjang (support 0,80); roti dan mentega bersama pada 3 dari 5 (0,60). Perhitungan ini persis yang dilakukan algoritma pada skala besar.

### Langkah 2: pembersihan transaksi

```python title="transactions_encoding.py" linenums="17"
--8<-- "association/transactions_encoding.py:17:20"
```

`{item.strip().lower() for item in basket}` adalah *set comprehension*: `strip()` membuang spasi di tepi, `lower()` mengecilkan huruf, dan **himpunan** (`set`) otomatis membuang duplikat. `sorted(...)` mengubahnya kembali menjadi daftar berurutan.

```text
--8<-- "expected/association/transactions_encoding.txt:11:12"
```

**Hasil.** `"Roti"`, `"roti "`, dan `"MENTEGA"` menjadi `roti` dan `mentega`, dan duplikat dalam satu keranjang hilang.

### Langkah 3: ringkasan data transaksi

```python title="transactions_encoding.py" linenums="22"
--8<-- "association/transactions_encoding.py:22:31"
```

| Kode | Penjelasan |
|---|---|
| `sizes = pd.Series([len(t) for t in transactions])` | Jumlah barang tiap keranjang. |
| `sizes.value_counts().sort_index().to_dict()` | Distribusi ukuran keranjang: berapa keranjang berisi 1, 2, 3, ... barang. |
| `Counter(i for t in transactions for i in t)` | Menghitung berapa kali tiap barang muncul (`Counter` dari modul `collections`); generator dengan dua `for` meratakan daftar bersarang. |
| `top.head(5)`, `top.tail(3)` | Lima barang paling populer dan tiga paling jarang. |

```text
--8<-- "expected/association/transactions_encoding.txt:14:17"
```

**Interpretasi.** 2000 transaksi dengan rata-rata 3,31 barang per keranjang (1 sampai 11); 377 keranjang hanya berisi satu barang. Barang terpopuler: `kantong_plastik` (895 keranjang, 44,8%), `air_mineral` (650), `roti` (527); paling jarang: `popok` (152), `deterjen` (140), `apel` (128).

### Langkah 4: matriks *one-hot* data besar

```python title="transactions_encoding.py" linenums="33"
--8<-- "association/transactions_encoding.py:33:36"
```

`df.to_numpy().mean()` merata-ratakan seluruh sel `True`/`False` (dihitung 1/0), yaitu proporsi sel bernilai benar.

```text
--8<-- "expected/association/transactions_encoding.txt:19:19"
```

**Interpretasi.** Ukuran matriks 2000 × 20 dan hanya 16,6% selnya bernilai benar. Matriks ini jarang (*sparse*); pada ribuan barang, ukurannya menjadi masalah memori sehingga representasi jarang atau penyaringan barang langka diperlukan.

??? example "Program lengkap (`transactions_encoding.py`)"
    ```python title="transactions_encoding.py" linenums="1"
    --8<-- "association/transactions_encoding.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memasukkan kuantitas atau harga seolah barang berbeda (aturan klasik hanya melihat ada/tidaknya barang).
    - Membiarkan barang berduplikat atau ejaan berbeda.
    - Tingkat barang yang terlalu rinci sehingga tidak ada itemset yang sering.
    - Memasukkan barang yang selalu ada (kantong, biaya kirim) tanpa menyadari efeknya pada aturan.

## Latihan

1. Hitung manual *support* {susu, telur} pada contoh kecil, lalu cek dengan kode.
2. Buat transaksi dari tabel format panjang (`transaction_id`, `item`) dengan `groupby`.
3. Hapus `kantong_plastik` dari data sintetis dan hitung ulang rata-rata barang per keranjang.

## Ringkasan

Data transaksi berupa himpunan barang per keranjang. Bersihkan nama dan duplikat, tentukan tingkat barang, lalu ubah menjadi matriks *one-hot*. *Support* adalah proporsi transaksi yang memuat suatu itemset.

Berikutnya: [Itemset Sering dan Apriori](apriori.md).
