"""Data pelanggan SINTETIS untuk latihan segmentasi (dibangkitkan dengan seed tetap).

Ada empat kelompok tersembunyi (`segment_true`). Kolom ini HANYA untuk validasi eksternal
dalam materi; pada masalah nyata kelompok semacam itu tidak diketahui.
"""
import numpy as np
import pandas as pd

SEGMENTS = {
    # nama: (spend juta Rp/tahun, kunjungan/bulan, keranjang ribu Rp, recency hari, usia)
    "pemburu_diskon": (6, 9, 90, 15, 24),
    "loyal": (30, 8, 250, 10, 38),
    "musiman": (14, 1.5, 420, 70, 45),
    "tidak_aktif": (3, 0.8, 110, 220, 40),
}


def make_customer_segments(n=600, seed=42):
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    labels = rng.choice(names, n, p=[0.3, 0.2, 0.25, 0.25])
    rows = []
    for name in labels:
        spend, visits, basket, recency, age = SEGMENTS[name]
        rows.append([
            rng.lognormal(np.log(spend), 0.45),
            max(rng.normal(visits, visits * 0.4), 0.1),
            max(rng.normal(basket, basket * 0.35), 10),
            max(rng.normal(recency, recency * 0.45), 1),
            np.clip(rng.normal(age, 9), 18, 75),
        ])
    df = pd.DataFrame(rows, columns=["annual_spend", "visits_per_month", "avg_basket", "recency_days", "age"])
    df["segment_true"] = labels
    return df.round(1)
