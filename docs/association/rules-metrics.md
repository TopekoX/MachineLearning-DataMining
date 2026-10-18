# Aturan Asosiasi: Support, Confidence, dan Lift

## Tujuan pembelajaran

- [x] Membentuk aturan \(A \Rightarrow B\) dari itemset sering.
- [x] Menghitung dan menafsirkan *support*, *confidence*, *lift*, *leverage*, dan *conviction*.
- [x] Mengenali bahaya membaca *confidence* tanpa *lift*.

## Aturan dan ukurannya

Sebuah **aturan** berbentuk \(A \Rightarrow B\) dengan \(A\) (*antecedent*) dan \(B\) (*consequent*) itemset yang tidak beririsan: "bila \(A\) dibeli, \(B\) cenderung dibeli juga". Ukurannya:

$$
\text{support}(A\Rightarrow B) = P(A \cap B), \qquad
\text{confidence}(A\Rightarrow B) = \frac{P(A \cap B)}{P(A)}, \qquad
\text{lift}(A\Rightarrow B) = \frac{P(A \cap B)}{P(A)\,P(B)}
$$

$$
\text{leverage} = P(A\cap B) - P(A)P(B), \qquad
\text{conviction} = \frac{1 - P(B)}{1 - \text{confidence}}
$$

\(P(\cdot)\) adalah proporsi transaksi yang memuat itemset tersebut.

| Ukuran | Arti | Nilai "tidak ada hubungan" |
|---|---|---|
| *Support* | Seberapa sering \(A\) dan \(B\) muncul bersama | (tidak ada) |
| *Confidence* | Dari transaksi yang memuat \(A\), berapa yang juga memuat \(B\) | \(P(B)\) |
| *Lift* | Seberapa lebih sering bersama dibanding bila saling bebas | **1** |
| *Leverage* | Selisih frekuensi bersama dan yang diharapkan jika bebas | 0 |
| *Conviction* | Seberapa sering aturan "salah" dibanding bila bebas | 1 |

**Lift > 1**: \(A\) dan \(B\) muncul bersama lebih sering dari kebetulan; **= 1**: tidak ada hubungan; **< 1**: cenderung saling menghindari. *Confidence* saja dapat menyesatkan: bila \(B\) sangat populer, *confidence* \(A\Rightarrow B\) akan tinggi bagaimanapun \(A\).

## Implementasi langkah demi langkah

Itemset sering dari FP-Growth (`min_support=0,05`), lalu `association_rules` dengan `min_confidence=0,3`. Data: 2000 transaksi sintetis ([penjelasan](transactions-encoding.md)). Program `examples/association/rules.py`; program lengkap ada di akhir bagian ini.

### Langkah 1: dari itemset sering ke aturan

```python title="rules.py" linenums="7"
--8<-- "association/rules.py:7:22"
```

| Kode | Penjelasan |
|---|---|
| `N = len(df)` | Jumlah transaksi total (2000). |
| `fpgrowth(df, min_support=0.05, use_colnames=True)` | Itemset sering (support ≥ 5%). |
| `association_rules(itemsets, num_itemsets=N, metric="confidence", min_threshold=0.3)` | Membentuk **semua aturan** \(A \Rightarrow B\) dari itemset sering dan menyimpan yang *confidence*-nya ≥ 0,3. Pada versi `mlxtend` yang dipakai, `num_itemsets` adalah jumlah transaksi. Hasilnya tabel dengan kolom `antecedents`, `consequents`, `support`, `confidence`, `lift`, dan lain-lain. |
| `def label(s): return " + ".join(sorted(s))` | Fungsi pembantu: mengubah himpunan barang (`frozenset`) menjadi teks. |
| `rules["rule"] = antecedents + "  ->  " + consequents` | Kolom teks aturan agar mudah dibaca, misalnya `roti  ->  mentega`. |

```text
--8<-- "expected/association/rules.txt:1:1"
```

**Hasil.** 45 aturan lolos `min_confidence=0,3`.

### Langkah 2: aturan dengan lift tertinggi

```python title="rules.py" linenums="24"
--8<-- "association/rules.py:24:26"
```

`rules.sort_values("lift", ascending=False).head(6)` mengurutkan dari lift terbesar dan mengambil enam teratas; `top[cols]` memilih kolom yang ingin ditampilkan.

```text
--8<-- "expected/association/rules.txt:3:9"
```

**Interpretasi.** Lift tertinggi: `sabun → sampo` dan `sampo → sabun` (lift 4,66), `popok → susu` dan `susu → popok` (4,45), serta `gula + kantong_plastik → kopi` (4,37). Aturan berpasangan arah (A→B dan B→A) memiliki *support* dan *lift* sama, tetapi *confidence* berbeda: `popok → susu` 0,763 (76% pembeli popok membeli susu) vs `susu → popok` 0,338.

### Langkah 3: memverifikasi satu aturan dengan hitungan tangan

**Tujuan.** Memastikan bahwa angka dari `mlxtend` memang sama dengan definisi: *support* = \(n_{AB}/N\), *confidence* = \(n_{AB}/n_A\), *lift* = confidence / \((n_B/N)\).

```python title="rules.py" linenums="28"
--8<-- "association/rules.py:28:38"
```

`df["roti"].sum()` menghitung keranjang yang memuat roti (kolom bernilai `True` dijumlahkan sebagai 1); `(df["roti"] & df["mentega"]).sum()` menghitung keranjang yang memuat **keduanya**. Baris `row = rules[rules["rule"] == "roti  ->  mentega"].iloc[0]` mengambil baris aturan yang sama dari tabel `mlxtend`.

```text
--8<-- "expected/association/rules.txt:11:13"
```

**Interpretasi.** Roti pada 527 keranjang, mentega 492, keduanya 439 dari 2000. Maka support = 439/2000 = 0,220; confidence = 439/527 = 0,833; lift = 0,833 / (492/2000) = 3,386. Nilai-nilai ini sama dengan keluaran `mlxtend`.

### Langkah 4: confidence tinggi tetapi lift rendah

**Konsep.** Bila konsekuen sangat populer, *confidence* tinggi sekalipun tidak ada hubungan. *Lift* membandingkannya dengan peluang kebetulan.

```python title="rules.py" linenums="40"
--8<-- "association/rules.py:40:43"
```

`rules["consequents"].apply(lambda s: s == frozenset({"kantong_plastik"}))` menghasilkan `True` untuk aturan yang konsekuennya persis `{kantong_plastik}`; hasilnya dipakai menyaring tabel, lalu diurutkan menurut *confidence*.

```text
--8<-- "expected/association/rules.txt:15:19"
```

**Interpretasi.** `kantong_plastik` ada di 44,8% keranjang. Aturan dengan konsekuen itu mendapat confidence 0,50 (`mie_instan + telur`), 0,50 (`telur`), dan 0,46 (`susu`), tampak lumayan, tetapi lift hanya 1,12; 1,12; dan 1,03, yaitu hampir sama dengan kebetulan. Melihat *confidence* saja akan salah menyimpulkan.

!!! note "Asosiasi bukan sebab-akibat"
    Lift tinggi menunjukkan pasangan yang layak diuji untuk bundling, penempatan rak berdekatan, atau rekomendasi. Ini **asosiasi, bukan sebab-akibat**: membeli popok tidak "menyebabkan" membeli susu.

??? example "Program lengkap (`rules.py`)"
    ```python title="rules.py" linenums="1"
    --8<-- "association/rules.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Mengurutkan aturan hanya dengan *confidence*.
    - Menganggap aturan berarti sebab-akibat.
    - Mengabaikan *support*: lift tinggi pada itemset sangat langka bisa hanya kebetulan.
    - Menganggap A→B sama informatifnya dengan B→A.

## Latihan

1. Hitung manual *leverage* dan *conviction* aturan `roti → mentega` dari angka di atas.
2. Urutkan aturan menurut *leverage*. Apakah urutannya berbeda dari urutan lift?
3. Cari aturan dengan lift < 1 dengan menurunkan `min_threshold` pada metrik `lift`.

## Ringkasan

*Support* mengukur frekuensi, *confidence* mengukur peluang bersyarat, dan *lift* membandingkan dengan kebebasan statistik; gunakan lift (dan ukuran lain) bersama confidence agar tidak tertipu barang populer.

Berikutnya: [Penyaringan Aturan, Ambang Batas, dan Keterbatasan](filtering-thresholds.md).
