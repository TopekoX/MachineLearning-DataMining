# Decision Tree

## Tujuan pembelajaran

- [x] Menjelaskan bagaimana pohon keputusan membagi data dan memilih pertanyaan.
- [x] Mengatur kompleksitas pohon dengan `max_depth` dan `min_samples_leaf`.
- [x] Membaca aturan pohon dan menafsirkan *feature importance* dengan hati-hati.

## Intuisi

Pohon keputusan adalah rangkaian pertanyaan ya/tidak ("`worst radius` ≤ 16,80?"). Data mengalir dari akar ke daun, dan daun menentukan kelas. Pada tiap simpul, pohon memilih pertanyaan yang membuat simpul anak paling **murni**. Ukuran ketidakmurnian yang umum adalah Gini:

$$
G = 1 - \sum_{c=1}^{C} p_c^2
$$

\(p_c\) adalah proporsi kelas \(c\) pada simpul. Simpul 50%–50% memiliki \(G = 0{,}5\) (paling tidak murni untuk dua kelas); simpul satu kelas memiliki \(G = 0\). Pohon yang dibiarkan tumbuh akan terus membagi sampai daun murni, yang berarti menghafal data latih.

## Ringkasan model

| Aspek | Keterangan |
|---|---|
| Kelebihan | Mudah dijelaskan dan divisualisasikan, menangkap non-linearitas dan interaksi, tidak perlu *scaling* |
| Keterbatasan | Mudah *overfitting*, tidak stabil terhadap perubahan kecil data, batas keputusan tegak lurus sumbu |
| Hyperparameter penting | `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion`, `ccp_alpha` |
| Praproses | *Scaling* tidak diperlukan |
| Cocok untuk | Kasus yang butuh aturan terbaca; fondasi Random Forest |

## Implementasi langkah demi langkah

Program `examples/classification/decision_tree.py` memakai data Breast Cancer; pemuatan dan pembagian datanya sama dengan [pelajaran dasar klasifikasi](preprocessing-baseline.md#langkah-1-memuat-data-dan-memisahkan-data-uji). Pohon tidak memerlukan *scaling*, sehingga tidak ada `StandardScaler`.

### Langkah 1: pengaruh `max_depth`

**Tujuan.** Menemukan kedalaman pohon yang cukup rumit untuk menangkap pola, tetapi belum menghafal data.

```python title="decision_tree.py" linenums="13"
--8<-- "classification/decision_tree.py:13:19"
```

| Kode | Penjelasan |
|---|---|
| `DecisionTreeClassifier(max_depth=depth, random_state=SEED)` | Menginisialisasi pohon. `max_depth` = jumlah maksimum tingkat pertanyaan (`None` = tanpa batas); `random_state` memastikan hasil sama bila ada seri antar pembagian. |
| `cross_val_score(tree, X_train, y_train, cv=5).mean()` | Akurasi validasi silang rata-rata pada data latih (objek `tree` yang belum dilatih tetap dapat diberikan, karena fungsi ini melatih salinannya per lipatan). |
| `tree.fit(X_train, y_train)` | Melatih pohon pada seluruh data latih (diperlukan agar `score` dan `get_n_leaves` dapat dipakai). |
| `tree.get_n_leaves()` | Jumlah simpul daun; ukuran kompleksitas pohon. |

```text
--8<-- "expected/classification/decision_tree.txt:1:7"
```

**Interpretasi.** Akurasi latih naik terus (0,923 → 1,000) karena pohon makin leluasa menghafal, tetapi akurasi validasi silang tertinggi pada `max_depth=4` (0,938), lalu turun ke 0,910 pada pohon tanpa batas (19 daun). Itulah pola *overfitting*: skor latih naik, skor validasi turun.

### Langkah 2: pengaruh `min_samples_leaf`

```python title="decision_tree.py" linenums="21"
--8<-- "classification/decision_tree.py:21:25"
```

`min_samples_leaf=leaf` melarang daun yang berisi kurang dari `leaf` sampel; daun besar berarti pohon tidak dapat menghafal sampel tunggal. Di sini kedalaman tidak dibatasi, hanya ukuran daun.

```text
--8<-- "expected/classification/decision_tree.txt:9:13"
```

**Interpretasi.** `min_samples_leaf=5` mencapai 0,934, jauh lebih baik daripada `1` (0,910); batas terlalu besar (`20`) menurunkannya ke 0,905 (*underfitting*). Ada nilai tengah yang ideal.

### Langkah 3: membaca aturan pohon

```python title="decision_tree.py" linenums="27"
--8<-- "classification/decision_tree.py:27:29"
```

`export_text(small, feature_names=...)` mencetak pohon sebagai teks. **Cara membaca**: tiap baris `|--- fitur <= nilai` adalah pertanyaan; bagian yang menjorok di bawahnya adalah cabang jawaban "ya"; baris `fitur > nilai` adalah cabang "tidak". `class: 1` di ujung adalah prediksi.

```text
--8<-- "expected/classification/decision_tree.txt:15:33"
```

**Interpretasi.** Pertanyaan pertama: apakah `worst radius <= 16.80`? Bila ya, tanyakan `worst concave points <= 0.14`; bila ya lagi, tanyakan `area error <= 91.56`, dan bila ya, prediksinya kelas 1 (jinak). Setiap sampel mengikuti satu jalur dari atas ke daun. Aturan seperti ini dapat langsung dijelaskan kepada orang lain, kelebihan utama pohon.

### Langkah 4: pentingnya fitur dan evaluasi uji

```python title="decision_tree.py" linenums="31"
--8<-- "classification/decision_tree.py:31:34"
```

`small.feature_importances_` adalah seberapa besar tiap fitur menurunkan ketidakmurnian (Gini) pada pohon yang dilatih; total semua fitur = 1. `pd.Series(..., index=X.columns)` memberi nama dan `sort_values(ascending=False)` mengurutkannya dari terbesar.

```text
--8<-- "expected/classification/decision_tree.txt:35:46"
```

**Interpretasi.** `worst radius` (0,764) menyumbang paling besar pada pohon ini. Nilai ini bukan bukti sebab-akibat; fitur yang berkorelasi dapat saling menggantikan. Pada data uji (pohon `max_depth=3`) akurasi 0,939; *recall* kelas ganas 0,905, yaitu **4 dari 42** kasus ganas terlewat, lebih buruk daripada regresi logistik. Perbedaan ini wajar bagi satu pohon, tetapi dengan 114 data uji ketidakpastiannya besar.

??? example "Program lengkap (`decision_tree.py`)"
    ```python title="decision_tree.py" linenums="1"
    --8<-- "classification/decision_tree.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Membiarkan pohon tumbuh tanpa batas lalu melaporkan akurasi latih.
    - Menganggap pohon pasti stabil; data latih yang sedikit berbeda dapat menghasilkan struktur berbeda.
    - Menafsirkan *feature importance* sebagai pengaruh kausal.
    - Memakai `random_state` berbeda tanpa menyadari hasil dapat berubah.

## Latihan

1. Ubah `criterion` menjadi `"entropy"`. Apakah hasil validasi silang berubah?
2. Coba `ccp_alpha` antara 0 dan 0,02. Bagaimana jumlah daun dan akurasi CV?
3. Ganti `random_state` beberapa kali untuk pohon tanpa batas. Seberapa bervariasi akurasi CV-nya?

## Ringkasan

Pohon keputusan membagi data dengan pertanyaan yang meningkatkan kemurnian simpul. Mudah dibaca tetapi mudah *overfitting*; batasi dengan `max_depth` dan `min_samples_leaf`, dan validasi pilihannya dengan validasi silang.

Berikutnya: [Random Forest](random-forest.md).
