from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import buat_dataset

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

FITUR = [
    "suhu_c",
    "kelembapan_persen",
    "jumlah_penghuni",
    "perangkat_aktif",
    "jam_ac",
    "akhir_pekan",
]


def regresi(df, laporan):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FITUR], df["konsumsi_kwh"], test_size=0.2, random_state=42
    )
    model = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    }
    laporan.append("== REGRESI: prediksi konsumsi_kwh ==")
    terbaik, pred_terbaik, skor_terbaik = None, None, -np.inf
    for nama, m in model.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        laporan.append(f"{nama:<18} MAE={mae:.3f} kWh | R2={r2:.3f}")
        if r2 > skor_terbaik:
            terbaik, pred_terbaik, skor_terbaik = nama, pred, r2
    laporan.append(f"Model regresi terbaik: {terbaik}")

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, pred_terbaik, s=10, alpha=0.5)
    batas = [y_test.min(), y_test.max()]
    plt.plot(batas, batas, "r--")
    plt.xlabel("Aktual (kWh)")
    plt.ylabel("Prediksi (kWh)")
    plt.title(f"Aktual vs Prediksi - {terbaik}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regresi_aktual_vs_prediksi.png", dpi=120)
    plt.close()


def klasifikasi(df, laporan):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FITUR],
        df["kategori"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=df["kategori"],
    )
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    laporan.append("")
    laporan.append("== KLASIFIKASI: kategori Hemat / Normal / Boros ==")
    laporan.append(f"Accuracy  = {accuracy_score(y_test, pred):.3f}")
    laporan.append(f"Precision = {precision_score(y_test, pred, average='macro'):.3f}")
    laporan.append(f"Recall    = {recall_score(y_test, pred, average='macro'):.3f}")
    laporan.append(f"F1-score  = {f1_score(y_test, pred, average='macro'):.3f}")

    penting = sorted(zip(FITUR, clf.feature_importances_), key=lambda x: -x[1])
    laporan.append("Fitur paling berpengaruh:")
    for nama, nilai in penting:
        laporan.append(f"  - {nama}: {nilai:.3f}")

    plt.figure(figsize=(6, 3.5))
    plt.barh([p[0] for p in penting][::-1], [p[1] for p in penting][::-1])
    plt.title("Pengaruh Fitur terhadap Kategori Konsumsi")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "klasifikasi_fitur_penting.png", dpi=120)
    plt.close()


def clustering(df, laporan):
    X = StandardScaler().fit_transform(df[FITUR])
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    df = df.assign(klaster=km.fit_predict(X))
    ringkas = df.groupby("klaster")[
        ["jumlah_penghuni", "perangkat_aktif", "jam_ac", "konsumsi_kwh"]
    ].mean()
    laporan.append("")
    laporan.append("== CLUSTERING: profil rumah (K-Means, k=3) ==")
    laporan.append(ringkas.round(2).to_string())

    plt.figure(figsize=(6, 4.5))
    plt.scatter(df["jam_ac"], df["konsumsi_kwh"], c=df["klaster"], s=10, cmap="viridis")
    plt.xlabel("Jam pemakaian AC")
    plt.ylabel("Konsumsi (kWh)")
    plt.title("Klaster Profil Rumah")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "clustering_profil_rumah.png", dpi=120)
    plt.close()


def main():
    df = buat_dataset()
    df.to_csv(DATA_DIR / "konsumsi_listrik.csv", index=False)

    laporan = [f"Jumlah data: {len(df)} baris, {len(FITUR)} fitur", ""]
    regresi(df, laporan)
    klasifikasi(df, laporan)
    clustering(df, laporan)

    teks = "\n".join(laporan)
    (OUT_DIR / "hasil_evaluasi.txt").write_text(teks, encoding="utf-8")
    print(teks)


if __name__ == "__main__":
    main()
