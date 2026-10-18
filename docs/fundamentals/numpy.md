# NumPy

## Tujuan pembelajaran

- [x] Membuat array dan memahami `shape` serta `dtype`.
- [x] Memilih elemen dengan *indexing*, *slicing*, dan *boolean mask*.
- [x] Memakai operasi vektor, agregasi dengan `axis`, dan *broadcasting*.
- [x] Menghitung jarak Euclidean, yang dipakai KNN dan k-Means.

## Mengapa NumPy?

Hampir semua pustaka ML (pandas, scikit-learn, Matplotlib) berbasis array NumPy. Array menyimpan data numerik sejenis secara efisien, dan operasinya dijalankan sekaligus pada seluruh elemen (**vektorisasi**) sehingga lebih singkat dan jauh lebih cepat daripada `for` biasa.

## Konsep kunci

- **`shape`**: ukuran tiap dimensi. `(2, 3)` berarti 2 baris dan 3 kolom. Data tabular ML umumnya berbentuk `(jumlah_sampel, jumlah_fitur)`.
- **`dtype`**: tipe data seluruh elemen (`int64`, `float64`, ...).
- **`axis`**: arah operasi. `axis=0` melintasi baris (hasil per kolom); `axis=1` melintasi kolom (hasil per baris).
- **Broadcasting**: aturan yang memperluas array berukuran lebih kecil agar cocok dengan yang lebih besar, misalnya mengurangkan rata-rata kolom dari tiap baris.

## Implementasi langkah demi langkah

Seluruh contoh ada di `examples/fundamentals/numpy_basics.py`. Di bawah, program dipecah menjadi tujuh langkah; program lengkap ada di akhir bagian ini.

### Langkah 1: mengimpor NumPy dan membuat array

**Tujuan.** Menyiapkan data numerik dalam bentuk array, struktur yang dipahami semua pustaka ML.

```python title="numpy_basics.py" linenums="1"
--8<-- "fundamentals/numpy_basics.py:1:8"
```

| Kode | Penjelasan |
|---|---|
| `import numpy as np` | Memuat NumPy dan memberinya nama pendek `np`, sebuah konvensi yang dipakai di mana-mana. |
| `np.array([1, 2, ...])` | Mengubah *list* Python menjadi array NumPy. Semua elemen array bertipe sama. |
| `a.shape` | Bentuk array dalam *tuple*. `(6,)` berarti satu dimensi berisi 6 elemen. |
| `a.dtype` | Tipe data elemen. `int64` berarti bilangan bulat 64-bit. |
| `a.reshape(2, 3)` | Menyusun ulang 6 elemen yang sama menjadi 2 baris × 3 kolom (jumlah elemen harus tetap 6). |

```text
--8<-- "expected/fundamentals/numpy_basics.txt:1:5"
```

**Interpretasi.** `a` memiliki satu sumbu; `M` memiliki dua sumbu: sumbu 0 = baris, sumbu 1 = kolom. Data tabular ML hampir selalu berbentuk `(jumlah_sampel, jumlah_fitur)`, yaitu 2 dimensi seperti `M`.

### Langkah 2: mengambil elemen (indexing dan slicing)

**Konsep.** Elemen diambil dengan indeks yang dimulai dari 0. Pada array 2 dimensi gunakan `[baris, kolom]`; tanda `:` berarti "semua".

```python title="numpy_basics.py" linenums="10"
--8<-- "fundamentals/numpy_basics.py:10:13"
```

| Kode | Hasil | Penjelasan |
|---|---|---|
| `M[1, 2]` | `6` | Baris ke-1, kolom ke-2 (mulai dari 0) |
| `M[:, 1]` | `[2 5]` | Semua baris, hanya kolom ke-1 |
| `a[a > 3]` | `[4 5 6]` | *Boolean mask*: `a > 3` menghasilkan `[F, F, F, T, T, T]`; hanya elemen bernilai `True` yang diambil |

```text
--8<-- "expected/fundamentals/numpy_basics.txt:6:8"
```

**Mengapa penting.** Pola `data[data > batas]` dipakai terus: memilih baris yang memenuhi syarat, mencari *outlier*, atau menyaring data pada pandas.

### Langkah 3: operasi vektor

**Konsep.** Operasi aritmetika pada array berlaku untuk **setiap elemen sekaligus** (*vektorisasi*), tanpa `for`.

```python title="numpy_basics.py" linenums="15"
--8<-- "fundamentals/numpy_basics.py:15:17"
```

```text
--8<-- "expected/fundamentals/numpy_basics.txt:9:10"
```

`a * 2` mengalikan tiap elemen dengan 2; `a ** 2` memangkatkan tiap elemen. Selain lebih singkat, cara ini jauh lebih cepat daripada perulangan Python pada data besar.

### Langkah 4: agregasi dan `axis`

**Konsep.** Fungsi seperti `sum()` dan `mean()` meringkas array. Parameter `axis` menentukan **sumbu mana yang dihilangkan**: `axis=0` meringkas sepanjang baris sehingga hasilnya satu angka **per kolom**; `axis=1` menghasilkan satu angka **per baris**.

```python title="numpy_basics.py" linenums="19"
--8<-- "fundamentals/numpy_basics.py:19:22"
```

```text
--8<-- "expected/fundamentals/numpy_basics.txt:11:12"
```

Untuk `M = [[1, 2, 3], [4, 5, 6]]`: jumlah per kolom = `[1+4, 2+5, 3+6] = [5 7 9]`; jumlah per baris = `[1+2+3, 4+5+6] = [6 15]`; rata-rata seluruh elemen = 21/6 = 3,5. `std()` adalah simpangan baku populasi (1,708).

**Kesalahan umum.** Terbalik memakai `axis=0` dan `axis=1`. Cara mengingatnya: sumbu yang Anda sebut adalah sumbu yang **hilang** dari hasil.

### Langkah 5: broadcasting

**Konsep.** Bila dua array berbeda bentuk, NumPy memperluas yang lebih kecil agar cocok (*broadcasting*), selama bentuknya kompatibel. Di sini `M` berbentuk `(2, 3)` dikurangi `M.mean(axis=0)` berbentuk `(3,)`: vektor rata-rata kolom dikurangkan dari **setiap baris**.

```python title="numpy_basics.py" linenums="24"
--8<-- "fundamentals/numpy_basics.py:24:26"
```

```text
--8<-- "expected/fundamentals/numpy_basics.txt:13:15"
```

Rata-rata kolom adalah `[2.5, 3.5, 4.5]`, sehingga kolom pertama menjadi `[1-2.5, 4-2.5] = [-1.5, 1.5]`, dan seterusnya. Operasi "kurangi rata-rata tiap kolom" ini adalah langkah pertama standardisasi data yang akan sering dipakai.

### Langkah 6: bilangan acak yang dapat diulang

```python title="numpy_basics.py" linenums="28"
--8<-- "fundamentals/numpy_basics.py:28:30"
```

```text
--8<-- "expected/fundamentals/numpy_basics.txt:16:16"
```

`np.random.default_rng(42)` membuat generator acak dengan **seed** 42. Seed yang sama selalu menghasilkan urutan angka yang sama, sehingga eksperimen dapat diulang dan hasilnya dapat dicocokkan dengan materi ini. `integers(0, 10, size=5)` mengambil 5 bilangan bulat dari 0 sampai 9. Mengganti seed akan mengganti angkanya.

### Langkah 7: jarak Euclidean

```python title="numpy_basics.py" linenums="32"
--8<-- "fundamentals/numpy_basics.py:32:34"
```

```text
--8<-- "expected/fundamentals/numpy_basics.txt:17:17"
```

`p - q` mengurangkan tiap koordinat, `** 2` mengkuadratkan, `.sum()` menjumlahkan, `np.sqrt` mengakarkan: \(\sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9+16} = 5\). Langkah ini adalah rumus yang dijelaskan di bawah.

??? example "Program lengkap (`numpy_basics.py`)"
    ```python title="numpy_basics.py" linenums="1"
    --8<-- "fundamentals/numpy_basics.py"
    ```

## Rumus: jarak Euclidean

Jarak antara dua titik \(\mathbf{p}\) dan \(\mathbf{q}\) dengan \(n\) fitur:

$$
d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
$$

Untuk \(\mathbf{p} = (1, 2)\) dan \(\mathbf{q} = (4, 6)\): \(\sqrt{3^2 + 4^2} = 5\), sesuai keluaran `jarak: 5.0`. Jarak ini dipakai KNN dan k-Means; karena itu skala fitur penting (lihat [Praproses Data](data-preprocessing.md)).

## Kesalahan umum

!!! warning "Hindari"
    - Bingung antara `shape` `(n,)` dan `(n, 1)`; banyak fungsi scikit-learn menuntut `X` berbentuk 2 dimensi.
    - Terbalik memakai `axis=0` dan `axis=1`.
    - Mengira `b = a` menyalin array; keduanya menunjuk data yang sama. Gunakan `a.copy()`.
    - Memakai `for` pada array besar padahal operasi vektor tersedia.

## Latihan

1. Buat array `0..11`, ubah menjadi `(3, 4)`, lalu hitung rata-rata tiap baris.
2. Ambil semua elemen genap dari array tersebut dengan *boolean mask*.
3. Hitung jarak Euclidean antara dua titik dengan 3 fitur, tanpa `for`.
4. Standardisasi tiap kolom matriks: kurangi rata-rata kolom, bagi simpangan baku kolom.

??? success "Petunjuk"
    1. `np.arange(12).reshape(3, 4).mean(axis=1)`
    2. `a[a % 2 == 0]`
    4. `(M - M.mean(axis=0)) / M.std(axis=0)`

## Ringkasan

NumPy menyediakan array dan operasi vektor yang menjadi fondasi seluruh ekosistem ML. Pahami `shape`, `axis`, *mask*, dan *broadcasting*.

Berikutnya: [pandas](pandas.md).
