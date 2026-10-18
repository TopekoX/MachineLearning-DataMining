import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

from basket_data import make_transactions

transactions = make_transactions()
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
print("transaksi:", len(df), "| barang:", df.shape[1])

# 1. Itemset sering dengan min_support = 0.05 (minimal 5% transaksi)
itemsets = apriori(df, min_support=0.05, use_colnames=True)
itemsets["size"] = itemsets["itemsets"].apply(len)
print("\nitemset sering: %d | menurut ukuran: %s" % (len(itemsets), itemsets["size"].value_counts().sort_index().to_dict()))

# 2. Itemset berukuran >= 2 dengan support tertinggi
multi = itemsets[itemsets["size"] >= 2].sort_values("support", ascending=False)
multi["itemsets"] = multi["itemsets"].apply(lambda s: " + ".join(sorted(s)))
print("\n" + multi.head(6)[["itemsets", "support"]].round(3).to_string(index=False))

# 3. Sifat anti-monoton: support sebuah itemset tidak pernah melebihi support himpunan bagiannya
sup = {frozenset(r.itemsets): r.support for r in itemsets.itertuples()}
pair = frozenset({"roti", "mentega"})
print("\nsupport {roti}=%.3f, {mentega}=%.3f, {roti, mentega}=%.3f" % (
    sup[frozenset({"roti"})], sup[frozenset({"mentega"})], sup[pair]))

# 4. Pengaruh min_support terhadap jumlah itemset (ledakan kombinatorial)
print("\nmin_support  jumlah itemset sering")
for ms in [0.2, 0.1, 0.05, 0.02, 0.01]:
    print(f"{ms:<12} {len(apriori(df, min_support=ms, use_colnames=True))}")

# 5. Membatasi ukuran itemset dengan max_len
print("\nmin_support=0.01 dengan max_len=2:", len(apriori(df, min_support=0.01, use_colnames=True, max_len=2)))
