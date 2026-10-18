from collections import Counter

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from basket_data import make_transactions

# 1. Contoh kecil yang dapat dihitung tangan: 5 transaksi
tiny = [["roti", "mentega"], ["roti", "susu"], ["roti", "mentega", "telur"], ["susu", "telur"], ["roti", "mentega"]]
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(tiny).transform(tiny), columns=te.columns_)
print(onehot.astype(int).to_string())
print("\nsupport {roti}          = %d/5 = %.2f" % (onehot["roti"].sum(), onehot["roti"].mean()))
print("support {roti, mentega} = %d/5 = %.2f" % ((onehot["roti"] & onehot["mentega"]).sum(), (onehot["roti"] & onehot["mentega"]).mean()))

# 2. Pembersihan: duplikat barang dalam satu transaksi dan huruf besar/kecil
raw = [["Roti", "roti ", "MENTEGA"], ["susu", "susu", "telur"]]
clean = [sorted({item.strip().lower() for item in basket}) for basket in raw]
print("\nsebelum:", raw, "\nsesudah:", clean)

# 3. Data transaksi sintetis (2000 keranjang) dan ringkasannya
transactions = make_transactions()
sizes = pd.Series([len(t) for t in transactions])
print("\njumlah transaksi: %d | rata-rata barang per keranjang: %.2f | ukuran keranjang min/maks: %d/%d" % (
    len(transactions), sizes.mean(), sizes.min(), sizes.max()))
print("distribusi ukuran keranjang:", sizes.value_counts().sort_index().to_dict())
freq = Counter(i for t in transactions for i in t)
top = pd.Series(freq).sort_values(ascending=False)
print("barang terpopuler (jumlah keranjang):", top.head(5).to_dict())
print("barang paling jarang:", top.tail(3).to_dict())

# 4. Pengodean one-hot (matriks transaksi x barang, boolean)
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
print("\nukuran matriks: %s | proporsi sel bernilai True: %.3f" % (df.shape, df.to_numpy().mean()))
