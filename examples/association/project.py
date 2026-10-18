"""Proyek analisis keranjang belanja minimarket (transaksi SINTETIS)."""
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from basket_data import make_transactions

# 1. Data dan pembersihan: normalisasi nama, hapus duplikat barang, buang keranjang kosong
raw = make_transactions(n=3000, seed=11)
transactions = [sorted({item.strip().lower() for item in basket}) for basket in raw if basket]
print("jumlah transaksi:", len(transactions), "| rata-rata barang per keranjang: %.2f" % (
    sum(map(len, transactions)) / len(transactions)))

te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# 2. Penambangan pada data penuh (tanpa target; tidak ada pembagian latih/uji)
itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)
rules = association_rules(itemsets, num_itemsets=len(df), metric="lift", min_threshold=1.2)
rules = rules[(rules["confidence"] >= 0.5) & (rules["antecedents"].apply(len) <= 2)]
print("itemset sering: %d | aturan setelah penyaringan (lift >= 1.2, confidence >= 0.5, antecedent <= 2 barang): %d" % (
    len(itemsets), len(rules)))


def name(s):
    return " + ".join(sorted(s))


# 3. Kurangi redundansi: pertahankan aturan dengan lift tertinggi untuk tiap itemset gabungan
rules = rules.assign(items=[frozenset(a | c) for a, c in zip(rules["antecedents"], rules["consequents"])])
best = rules.sort_values(["lift", "confidence"], ascending=False).drop_duplicates("items")
best["rule"] = best["antecedents"].apply(name) + "  ->  " + best["consequents"].apply(name)
print("itemset gabungan unik:", len(best))
print("\n" + best.head(8)[["rule", "support", "confidence", "lift"]].round(3).to_string(index=False))

# 4. Pemeriksaan kestabilan sebagai pengganti validasi: aturan dari paruh pertama diuji pada paruh kedua
half = len(df) // 2
first, second = df.iloc[:half], df.iloc[half:]
r1 = association_rules(fpgrowth(first, min_support=0.02, use_colnames=True), num_itemsets=len(first),
                       metric="lift", min_threshold=1.2)
r1 = r1[(r1["confidence"] >= 0.5) & (r1["antecedents"].apply(len) <= 2)]
top = r1.sort_values("lift", ascending=False).head(5)


def lift_in(data, ant, con):
    p_a, p_c = data[list(ant)].all(axis=1).mean(), data[list(con)].all(axis=1).mean()
    p_ac = data[list(ant | con)].all(axis=1).mean()
    return p_ac / (p_a * p_c)


print("\nStabilitas 5 aturan teratas paruh 1 -> lift pada paruh 2")
for r in top.itertuples():
    print(f"{name(r.antecedents):<22} -> {name(r.consequents):<10} paruh1 {r.lift:.2f} | paruh2 {lift_in(second, r.antecedents, r.consequents):.2f}")

# 5. Simpan tabel aturan (keluaran proyek ini adalah tabel, bukan model prediksi)
out = best[["rule", "support", "confidence", "lift"]].reset_index(drop=True)
print("\ntabel aturan siap disimpan sebagai CSV: %d baris, kolom %s" % (len(out), out.columns.tolist()))
