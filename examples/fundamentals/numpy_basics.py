import numpy as np

# 1. Membuat array
a = np.array([1, 2, 3, 4, 5, 6])
print("a      :", a, "| shape:", a.shape, "| dtype:", a.dtype)

M = a.reshape(2, 3)  # 2 baris, 3 kolom
print("M:\n", M, "\nshape:", M.shape)

# 2. Indexing dan slicing
print("M[1, 2]  :", M[1, 2])
print("M[:, 1]  :", M[:, 1])        # seluruh baris, kolom ke-1
print("a[a > 3] :", a[a > 3])       # boolean mask

# 3. Operasi vektor (tanpa for-loop)
print("a * 2    :", a * 2)
print("a ** 2   :", a ** 2)

# 4. Agregasi dan axis
print("sum semua:", M.sum(), "| sum per kolom (axis=0):", M.sum(axis=0),
      "| sum per baris (axis=1):", M.sum(axis=1))
print("mean:", M.mean(), "| std:", M.std().round(3))

# 5. Broadcasting: kurangi rata-rata setiap kolom
centered = M - M.mean(axis=0)
print("M - mean kolom:\n", centered)

# 6. Bilangan acak yang dapat diulang
rng = np.random.default_rng(42)
print("acak:", rng.integers(0, 10, size=5))

# 7. Jarak Euclidean antara dua titik (dipakai KNN dan k-Means)
p, q = np.array([1.0, 2.0]), np.array([4.0, 6.0])
print("jarak:", np.sqrt(((p - q) ** 2).sum()))
