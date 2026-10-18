import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from basket_data import make_transactions

transactions = make_transactions()
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)


def mine(data, min_support, min_confidence):
    its = fpgrowth(data, min_support=min_support, use_colnames=True)
    if len(its) == 0:
        return pd.DataFrame()
    return association_rules(its, num_itemsets=len(data), metric="confidence", min_threshold=min_confidence)


# 1. Jumlah aturan menurut kombinasi ambang
print("min_support  min_confidence=0.3  0.5  0.7")
for ms in [0.01, 0.02, 0.05]:
    counts = [len(mine(df, ms, mc)) for mc in [0.3, 0.5, 0.7]]
    print(f"{ms:<12} {counts[0]:<19} {counts[1]:<4} {counts[2]}")

# 2. Penyaringan bertahap pada min_support=0.02, min_confidence=0.3
rules = mine(df, 0.02, 0.3)
step1 = rules[rules["lift"] >= 1.2]
step2 = step1[step1["confidence"] >= 0.5]
print("\naturan awal: %d -> lift >= 1.2: %d -> confidence >= 0.5: %d" % (len(rules), len(step1), len(step2)))
single = step2[(step2["antecedents"].apply(len) == 1) & (step2["consequents"].apply(len) == 1)]
print("dengan satu barang di tiap sisi:", len(single))

# 3. Aturan yang lolos confidence tetapi gagal lift (barang populer)
popular = rules[(rules["confidence"] >= 0.4) & (rules["lift"] < 1.2)]
print("aturan confidence >= 0.4 tetapi lift < 1.2: %d (konsekuen terbanyak: %s)" % (
    len(popular), popular["consequents"].apply(lambda s: ",".join(sorted(s))).value_counts().head(2).to_dict()))

# 4. Pengganti pembagian latih/uji: uji kestabilan aturan pada paruh transaksi berikutnya
half = len(df) // 2
first, second = df.iloc[:half], df.iloc[half:]
rules_a = mine(first, 0.02, 0.3)
rules_a = rules_a[(rules_a["lift"] >= 1.2) & (rules_a["confidence"] >= 0.5)]


def lift_in(data, ant, con):
    p_a, p_c = data[list(ant)].all(axis=1).mean(), data[list(con)].all(axis=1).mean()
    p_ac = data[list(ant | con)].all(axis=1).mean()
    return p_ac / (p_a * p_c) if p_a > 0 and p_c > 0 and p_ac > 0 else 0.0


rules_a["lift_paruh2"] = [lift_in(second, a, c) for a, c in zip(rules_a["antecedents"], rules_a["consequents"])]
kept = (rules_a["lift_paruh2"] >= 1.2).mean()
print("\nAturan dari paruh 1: %d | masih lift >= 1.2 pada paruh 2: %.0f%%" % (len(rules_a), 100 * kept))
diff = rules_a["lift"] - rules_a["lift_paruh2"]
print("selisih lift (paruh 1 - paruh 2): median %.3f | terkecil %.2f | terbesar %.2f" % (diff.median(), diff.min(), diff.max()))
rare = rules_a.assign(n=(rules_a["support"] * len(first)).round().astype(int)).nsmallest(1, "support")
print("aturan dengan support terkecil: %d keranjang pada paruh 1" % rare["n"].iloc[0])
