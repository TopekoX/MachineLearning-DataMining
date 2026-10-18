# Statistik untuk Data Mining

## Tujuan pembelajaran

- [x] Memilih ukuran pemusatan dan sebaran yang tepat.
- [x] Mendeteksi *outlier* dengan aturan IQR.
- [x] Menafsirkan korelasi beserta batasannya.
- [x] Membaca probabilitas bersyarat langsung dari data.

## Ukuran pemusatan dan sebaran

- **Rata-rata** (*mean*) \(\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i\) sensitif terhadap nilai ekstrem.
- **Median** (nilai tengah) tahan terhadap nilai ekstrem.
- **Simpangan baku** (*standard deviation*) mengukur sebaran di sekitar rata-rata:

$$
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

Bila rata-rata jauh lebih besar dari median, distribusi miring ke kanan (*skewness* positif), biasanya karena nilai ekstrem.

## Kuartil dan aturan IQR

Kuartil membagi data terurut menjadi empat bagian. \(Q_1\) = persentil ke-25, \(Q_3\) = persentil ke-75, dan \(\text{IQR} = Q_3 - Q_1\). Aturan umum menandai data di luar

$$
[\,Q_1 - 1{,}5\,\text{IQR},\; Q_3 + 1{,}5\,\text{IQR}\,]
$$

sebagai calon *outlier*. Ini **penanda untuk diperiksa**, bukan perintah untuk menghapus data.

## Korelasi

Korelasi Pearson \(r\) mengukur kekuatan hubungan **linear** antara dua variabel, dari -1 (berlawanan sempurna) sampai +1 (searah sempurna); 0 berarti tidak ada hubungan linear:

$$
r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\,\sqrt{\sum_i (y_i - \bar{y})^2}}
$$

## Probabilitas bersyarat

\(P(A \mid B)\) adalah peluang \(A\) bila \(B\) diketahui terjadi. Dari data, itu sama dengan proporsi \(A\) di dalam kelompok \(B\): misalnya \(P(\text{churn} \mid \text{plan}=\text{basic})\).

## Implementasi langkah demi langkah

Contoh memakai data pelanggan sintetis ([pandas](pandas.md#dataset-yang-dipakai)) dan dataset Diabetes bawaan scikit-learn (442 pasien, 10 fitur, target berupa ukuran progres penyakit). Program lengkap ada di akhir bagian ini.

### Langkah 1: menyiapkan data

```python title="statistics_demo.py" linenums="1"
--8<-- "fundamentals/statistics_demo.py:1:8"
```

`df["income"].dropna()` mengambil kolom pendapatan **tanpa nilai hilang**; kebanyakan fungsi statistik pandas sebenarnya mengabaikan `NaN`, tetapi membuangnya secara eksplisit membuat jumlah data yang dipakai jelas.

### Langkah 2: pemusatan dan sebaran

**Tujuan.** Meringkas satu kolom dengan beberapa angka: di mana pusatnya, seberapa lebar sebarannya, dan apakah bentuknya miring.

```python title="statistics_demo.py" linenums="10"
--8<-- "fundamentals/statistics_demo.py:10:14"
```

`"%.0f" % nilai` memformat angka sebagai bilangan **tanpa desimal**; `"%.2f"` dua desimal. Metode `mean()`, `median()`, `std()` (simpangan baku sampel, pembagi \(n-1\)), dan `skew()` (kemiringan) berasal dari `Series` pandas.

```text
--8<-- "expected/fundamentals/statistics_demo.txt:1:4"
```

**Interpretasi.** Rata-rata (7,45 juta) lebih besar dari median (6,05 juta), dan *skewness* 13,78: distribusi `income` sangat miring ke kanan karena nilai ekstrem menarik rata-rata ke atas, sedangkan median tidak terpengaruh. Median lebih mewakili "pelanggan tipikal". (Pada data ini ada tiga nilai yang sengaja diperbesar 20 kali sebagai *outlier* buatan.)

### Langkah 3: kuartil dan aturan IQR untuk *outlier*

**Konsep.** \(Q_1\) dan \(Q_3\) adalah nilai yang memisahkan 25% terbawah dan 25% teratas; selisihnya adalah IQR. Data di luar \([Q_1 - 1{,}5\,\text{IQR},\; Q_3 + 1{,}5\,\text{IQR}]\) ditandai sebagai calon *outlier*.

```python title="statistics_demo.py" linenums="16"
--8<-- "fundamentals/statistics_demo.py:16:22"
```

| Kode | Penjelasan |
|---|---|
| `income.quantile([0.25, 0.75])` | Mengembalikan dua nilai (kuartil 1 dan 3). `q1, q3 = ...` menyimpannya ke dua variabel sekaligus (*unpacking*). |
| `lower, upper = ...` | Menghitung batas bawah dan atas dengan rumus IQR. |
| `income[(income < lower) \| (income > upper)]` | Menyaring nilai di bawah batas bawah **atau** (`\|`) di atas batas atas; tiap syarat dalam kurung. |
| `len(outliers)` | Jumlah calon *outlier*. |

```text
--8<-- "expected/fundamentals/statistics_demo.txt:5:6"
```

**Interpretasi.** \(Q_1\) = 4,13 juta, \(Q_3\) = 8,08 juta, IQR = 3,95 juta. Batas atas ≈ 14,0 juta; **24** nilai melewatinya. Sebagian berasal dari ekor distribusi yang wajar, bukan kesalahan. Aturan ini hanya **penanda untuk diperiksa**, bukan perintah menghapus data.

### Langkah 4: korelasi dan batasannya

**Konsep.** Korelasi Pearson mengukur kekuatan hubungan **linear** (-1 sampai +1). Ia tidak mendeteksi hubungan melengkung dan tidak membuktikan sebab-akibat.

```python title="statistics_demo.py" linenums="24"
--8<-- "fundamentals/statistics_demo.py:24:31"
```

| Kode | Penjelasan |
|---|---|
| `rng.uniform(-3, 3, 200)` | 200 bilangan acak merata antara -3 dan 3 (generator ber-seed 0, jadi dapat diulang). |
| `y = x ** 2 + rng.normal(0, 0.3, 200)` | `y` ditentukan oleh `x` (kuadratnya) ditambah sedikit derau; hubungannya jelas tetapi berbentuk "U", bukan garis. Data ini **sintetis**. |
| `np.corrcoef(x, y)[0, 1]` | Matriks korelasi 2×2; elemen `[0, 1]` adalah korelasi `x` dan `y`. |
| `diabetes.corr()["target"]` | Korelasi tiap kolom dengan kolom `target`; `.drop("target")` membuang korelasi target dengan dirinya; `.sort_values(ascending=False).head(3)` mengambil tiga terbesar. |

```text
--8<-- "expected/fundamentals/statistics_demo.txt:7:11"
```

**Interpretasi.** Korelasi `x` dan `x²` hanya **0,10**, padahal `y` sepenuhnya ditentukan `x`. Korelasi mendekati 0 **tidak** berarti tidak ada hubungan, hanya tidak ada hubungan **linear**. Pada Diabetes, `bmi` (0,59) dan `s5` (0,57) paling berkorelasi dengan target; ini hubungan statistik, bukan bukti sebab-akibat.

### Langkah 5: probabilitas bersyarat dari data

**Konsep.** \(P(\text{churn} \mid \text{plan}=\text{basic})\) adalah proporsi churn **di dalam** kelompok `basic`.

```python title="statistics_demo.py" linenums="33"
--8<-- "fundamentals/statistics_demo.py:33:35"
```

`pd.crosstab(df["plan"], df["churn"], normalize="index")` membuat tabel silang paket × churn; `normalize="index"` membagi tiap baris dengan jumlah barisnya sehingga setiap baris berjumlah 1 (proporsi dalam kelompok). `df["churn"].mean()` adalah proporsi churn keseluruhan.

```text
--8<-- "expected/fundamentals/statistics_demo.txt:12:16"
```

**Interpretasi.** \(P(\text{churn}\mid\text{basic}) = 0{,}391\), \(P(\text{churn}\mid\text{premium}) = 0{,}219\), dibanding \(P(\text{churn}) = 0{,}339\) secara keseluruhan. Paket tampaknya terkait churn pada data ini. Hubungan ini baru gambaran; belum menunjukkan bahwa paket **menyebabkan** churn.

??? example "Program lengkap (`statistics_demo.py`)"
    ```python title="statistics_demo.py" linenums="1"
    --8<-- "fundamentals/statistics_demo.py"
    ```

## Kesalahan umum

!!! warning "Hindari"
    - Memakai rata-rata pada data yang sangat miring tanpa melihat median.
    - Menghapus *outlier* otomatis tanpa memahami penyebabnya.
    - Menyimpulkan "korelasi = sebab-akibat" atau "korelasi 0 = tidak berhubungan".
    - Membandingkan rata-rata dua kelompok tanpa memperhatikan ukuran sampelnya.

## Latihan

1. Hitung median dan rata-rata `avg_spend`. Mana yang lebih mewakili, dan mengapa?
2. Terapkan aturan IQR pada `age`. Berapa *outlier*-nya?
3. Hitung \(P(\text{churn}\mid\text{city})\) untuk tiap kota. Apakah perbedaannya cukup besar untuk menyimpulkan sesuatu?
4. Buat \(y = \sin(x)\) untuk \(x\) dalam \([0, 2\pi]\) lalu hitung korelasinya dengan \(x\). Bagaimana hasilnya?

## Ringkasan

Gunakan median dan IQR untuk data miring dan berpotensi ber-*outlier*; korelasi Pearson hanya mengukur hubungan linear; probabilitas bersyarat dapat dihitung dari proporsi dalam kelompok.

Berikutnya: [Exploratory Data Analysis (EDA)](eda.md).
