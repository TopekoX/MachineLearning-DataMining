"""Data pelanggan SINTETIS (dibangkitkan dengan seed tetap) untuk latihan.

Target `churn`: 1 = pelanggan berhenti berlangganan. Data sengaja dibuat
"kotor": ada nilai hilang, outlier, dan baris duplikat.
"""
import numpy as np
import pandas as pd


def make_customer_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    city = rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan"], n, p=[0.4, 0.2, 0.25, 0.15])
    plan = rng.choice(["basic", "premium"], n, p=[0.7, 0.3])
    tenure_months = rng.integers(1, 61, n)
    income = rng.lognormal(mean=15.6, sigma=0.5, size=n).round(-3)  # rupiah/bulan
    n_orders = rng.poisson(8, n)
    avg_spend = rng.gamma(shape=4, scale=50_000, size=n).round(-2)

    # Peluang churn lebih tinggi: langganan baru, paket basic, jarang bertransaksi
    logit = (-1.2 - 0.04 * (tenure_months - 30) + 0.7 * (plan == "basic")
             - 0.12 * (n_orders - 8))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"age": age, "city": city, "plan": plan,
                       "tenure_months": tenure_months, "income": income,
                       "n_orders": n_orders, "avg_spend": avg_spend, "churn": churn})

    # Kotorkan data
    df.loc[rng.choice(n, int(0.08 * n), replace=False), "income"] = np.nan
    df.loc[rng.choice(n, int(0.03 * n), replace=False), "city"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "income"] *= 20  # outlier
    df = pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)  # duplikat
    return df
