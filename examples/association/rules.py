import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from basket_data import make_transactions

transactions = make_transactions()
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
N = len(df)

itemsets = fpgrowth(df, min_support=0.05, use_colnames=True)
rules = association_rules(itemsets, num_itemsets=N, metric="confidence", min_threshold=0.3)


def label(s):
    return " + ".join(sorted(s))


rules["rule"] = rules["antecedents"].apply(label) + "  ->  " + rules["consequents"].apply(label)
cols = ["rule", "support", "confidence", "lift", "leverage", "conviction"]
print("jumlah aturan (min_support=0.05, min_confidence=0.3):", len(rules))

# 1. Aturan dengan lift tertinggi
top = rules.sort_values("lift", ascending=False).head(6)
print("\n" + top[cols].round(3).to_string(index=False))

# 2. Verifikasi manual satu aturan: roti -> mentega, dari hitungan langsung
n_roti = int(df["roti"].sum())
n_mentega = int(df["mentega"].sum())
n_both = int((df["roti"] & df["mentega"]).sum())
support = n_both / N
confidence = n_both / n_roti
lift = confidence / (n_mentega / N)
print("\nroti -> mentega: n_roti=%d, n_mentega=%d, n_keduanya=%d, N=%d" % (n_roti, n_mentega, n_both, N))
print("support=%.3f  confidence=%.3f  lift=%.3f" % (support, confidence, lift))
row = rules[rules["rule"] == "roti  ->  mentega"].iloc[0]
print("mlxtend  : support=%.3f  confidence=%.3f  lift=%.3f" % (row["support"], row["confidence"], row["lift"]))

# 3. Aturan dengan confidence tinggi tetapi lift sekitar 1 (barang populer)
pop = rules[(rules["consequents"].apply(lambda s: s == frozenset({"kantong_plastik"})))]
print("\nAturan dengan konsekuen kantong_plastik (support kantong_plastik = %.3f):" % df["kantong_plastik"].mean())
print(pop.sort_values("confidence", ascending=False).head(3)[cols].round(3).to_string(index=False))
