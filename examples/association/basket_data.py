"""Transaksi belanja SINTETIS (dibangkitkan dengan seed tetap) untuk latihan aturan asosiasi.

Pola yang ditanam: roti+mentega(+selai), mie_instan+telur, kopi+gula, popok+susu, sabun+sampo.
Barang 'kantong_plastik' sengaja sangat populer agar terlihat bahaya membaca confidence tanpa lift.
"""
import numpy as np

# barang -> peluang dasar muncul pada sebuah transaksi (di luar pola yang ditanam)
BASE = {
    "roti": 0.06, "mentega": 0.03, "selai": 0.04, "susu": 0.12, "telur": 0.12, "mie_instan": 0.14,
    "kopi": 0.07, "gula": 0.05, "teh": 0.10, "beras": 0.12, "minyak": 0.10, "sabun": 0.08,
    "sampo": 0.04, "popok": 0.03, "pisang": 0.10, "apel": 0.07, "kantong_plastik": 0.45,
    "tisu": 0.10, "deterjen": 0.07, "air_mineral": 0.25,
}
PATTERNS = [  # (peluang pola muncul, barang pada pola)
    (0.16, ["roti", "mentega"]),
    (0.07, ["roti", "mentega", "selai"]),
    (0.10, ["mie_instan", "telur"]),
    (0.12, ["kopi", "gula"]),
    (0.06, ["popok", "susu"]),
    (0.07, ["sabun", "sampo"]),
]


def make_transactions(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    transactions = []
    for _ in range(n):
        basket = {item for item, p in BASE.items() if rng.random() < p}
        for p, items in PATTERNS:
            if rng.random() < p:
                basket.update(items)
        if not basket:  # transaksi kosong tidak ada
            basket.add("air_mineral")
        transactions.append(sorted(basket))
    return transactions
