"""
AMAÇ: Oyuncuları özelliklerine göre kendiliğinden gruplara ayırmak (unsupervised learning).
K-Means kullanılıyor - "doğru cevap" yok, amaç anlamlı/yorumlanabilir kümeler bulmak.
Piyasa değeri kümelemeye GİRMEZ; sadece sonradan kümeleri yorumlamak için kullanılır.

K seçimi kuralı: K=2..10 için silhouette skoru hesaplanır. K=2 yalnızca ikili bir ayrım
(çok oynayan / az oynayan) verdiği için segmentasyon amacına yetmez; K >= 3 arasında
silhouette skoru en yüksek olan K seçilir. K=2'nin skoru da raporlanır.

Çalıştırma (proje klasöründen): python src/cluster_players.py
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paths import FIGURES_DIR, MODELS_DIR, PROCESSED_FILE, RANDOM_STATE, REPORTS_DIR

# Sade, yorumlanabilir sayısal özellikler (one-hot sütunlar mesafe hesabını bozar)
CLUSTER_FEATURES = [
    "age", "height_in_cm", "international_caps", "international_goals",
    "total_matches", "total_goals", "total_assists", "total_minutes",
    "goals_per_match", "minutes_per_match",
]
K_RANGE = range(2, 11)
MIN_K = 3
SILHOUETTE_SAMPLE = 6000  # 13 bin oyuncunun tamamında silhouette çok yavaş; sabit örneklem


def build_kmeans(k):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # birkaç eksik boy değeri
        ("scale", StandardScaler()),                   # K-Means mesafe bazlı: ölçekleme şart
        ("kmeans", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)),
    ])


def main():
    df = pd.read_csv(PROCESSED_FILE)
    X = df[CLUSTER_FEATURES]

    # 1) Farklı K değerlerini dene
    scores = []
    for k in K_RANGE:
        model = build_kmeans(k).fit(X)
        X_scaled = model[:-1].transform(X)
        sil = silhouette_score(X_scaled, model["kmeans"].labels_,
                               sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE)
        scores.append({"k": k, "inertia": float(model["kmeans"].inertia_), "silhouette": float(sil)})
        print(f"K={k:2d}  inertia={model['kmeans'].inertia_:>9,.0f}  silhouette={sil:.3f}")
    scores = pd.DataFrame(scores)

    best_k = int(scores[scores["k"] >= MIN_K].sort_values("silhouette", ascending=False).iloc[0]["k"])
    print(f"\nSeçilen K (K>={MIN_K} içinde en yüksek silhouette): {best_k}")

    # 2) Final model
    final = build_kmeans(best_k).fit(X)
    df["cluster"] = final["kmeans"].labels_

    summary = df.groupby("cluster").agg(
        players=("player_id", "count"),
        age=("age", "mean"),
        total_matches=("total_matches", "mean"),
        total_goals=("total_goals", "mean"),
        goals_per_match=("goals_per_match", "mean"),
        minutes_per_match=("minutes_per_match", "mean"),
        international_caps=("international_caps", "mean"),
        median_market_value_eur=("market_value_in_eur", "median"),
        mean_market_value_eur=("market_value_in_eur", "mean"),
    ).round(2)
    print("\n=== KÜME ÖZETİ (piyasa değeri sadece yorum için) ===")
    print(summary.to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / "player_clusters.joblib")
    summary.to_csv(REPORTS_DIR / "cluster_summary.csv")
    (REPORTS_DIR / "metrics_clustering.json").write_text(json.dumps({
        "features": CLUSTER_FEATURES,
        "k_selection_rule": f"K>={MIN_K} içinde en yüksek silhouette",
        "k_scores": scores.to_dict(orient="records"),
        "selected_k": best_k,
        "selected_silhouette": float(scores.loc[scores["k"] == best_k, "silhouette"].iloc[0]),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Grafik 1: elbow + silhouette
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(scores["k"], scores["inertia"], "o-", color="#4C78A8")
    ax1.set(xlabel="K (küme sayısı)", ylabel="Inertia (elbow)")
    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["silhouette"], "s--", color="#E45756")
    ax2.set_ylabel("Silhouette", color="#E45756")
    ax1.axvline(best_k, color="gray", ls=":", lw=1)
    ax1.set_title(f"K seçimi (seçilen K={best_k})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "k_secimi.png", dpi=150)
    plt.close(fig)

    # Grafik 2: kümelerin 2 boyutlu görünümü (PCA sadece çizim için)
    points = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(final[:-1].transform(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(df["cluster"].unique()):
        mask = df["cluster"] == c
        ax.scatter(points[mask, 0], points[mask, 1], s=4, alpha=0.4, label=f"Küme {c} ({mask.sum()})")
    ax.set(xlabel="PCA 1", ylabel="PCA 2", title="Oyuncu kümeleri (PCA ile 2 boyuta indirgenmiş)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kumeler_pca.png", dpi=150)
    plt.close(fig)

    print("\nKaydedildi: models/player_clusters.joblib, reports/cluster_summary.csv, "
          "reports/metrics_clustering.json, reports/figures/")


if __name__ == "__main__":
    main()
