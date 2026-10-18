import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from basket_data import make_transactions

# 1. Langkah pertama FP-Growth pada contoh kecil: hitung frekuensi barang, urutkan menurun,
#    lalu buang barang di bawah support minimum (di sini minimal 2 dari 5 transaksi)
tiny = [["roti", "mentega"], ["roti", "susu"], ["roti", "mentega", "telur"], ["susu", "telur"], ["roti", "mentega"]]
counts = pd.Series([i for t in tiny for i in t]).value_counts()
print("frekuensi barang:", counts.to_dict())
order = counts[counts >= 2].index.tolist()
print("urutan header (menurun):", order)
for t in tiny:
    print("transaksi diurutkan:", sorted(t, key=order.index))

# 2. Apriori dan FP-Growth menghasilkan itemset yang SAMA
transactions = make_transactions()
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

for ms in [0.05, 0.01]:
    a = apriori(df, min_support=ms, use_colnames=True)
    f = fpgrowth(df, min_support=ms, use_colnames=True)
    sa = {frozenset(x): round(s, 10) for x, s in zip(a["itemsets"], a["support"])}
    sf = {frozenset(x): round(s, 10) for x, s in zip(f["itemsets"], f["support"])}
    print(f"\nmin_support={ms}: Apriori {len(a)} itemset | FP-Growth {len(f)} itemset | identik: {sa == sf}")

# 3. Urutan baris keluaran tidak dijamin sama; bandingkan sebagai himpunan, bukan sebagai urutan
print("\nurutan baris sama persis? ", [frozenset(x) for x in a["itemsets"]] == [frozenset(x) for x in f["itemsets"]])
