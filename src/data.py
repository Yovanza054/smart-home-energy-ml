import numpy as np
import pandas as pd


def buat_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)

    suhu = rng.normal(31, 3, n).clip(23, 38)
    kelembapan = rng.normal(75, 8, n).clip(50, 95)
    penghuni = rng.integers(1, 7, n)
    perangkat = rng.integers(3, 25, n)
    jam_ac = np.where(suhu > 30, rng.uniform(2, 12, n), rng.uniform(0, 5, n))
    akhir_pekan = rng.integers(0, 2, n)

    kwh = (
        2.0
        + 0.35 * (suhu - 23)
        + 0.02 * (kelembapan - 50)
        + 0.60 * penghuni
        + 0.25 * perangkat
        + 0.90 * jam_ac
        + 1.2 * akhir_pekan
        + rng.normal(0, 1.2, n)
    ).clip(1)

    df = pd.DataFrame(
        {
            "suhu_c": suhu.round(1),
            "kelembapan_persen": kelembapan.round(1),
            "jumlah_penghuni": penghuni,
            "perangkat_aktif": perangkat,
            "jam_ac": jam_ac.round(1),
            "akhir_pekan": akhir_pekan,
            "konsumsi_kwh": kwh.round(2),
        }
    )

    batas_rendah, batas_tinggi = df["konsumsi_kwh"].quantile([0.33, 0.66])
    df["kategori"] = pd.cut(
        df["konsumsi_kwh"],
        bins=[-np.inf, batas_rendah, batas_tinggi, np.inf],
        labels=["Hemat", "Normal", "Boros"],
    )
    return df
