"""
AMAÇ: Oyuncuları özelliklerine göre kendiliğinden gruplara ayırmak (unsupervised learning).
K-Means kullanılıyor - "doğru cevap" yok, amaç anlamlı/yorumlanabilir kümeler bulmak.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Temizlenmiş veriyi oku
df = pd.read_csv("../data/processed/players_cleaned.csv")

# Kümeleme için sade, yorumlanabilir sayısal özellikler seç
# (one-hot encoded sütunlar kullanılmıyor - mesafe hesabını bozar)
cluster_features = [
    "age", "height_in_cm", "international_caps", "international_goals",
    "total_matches", "total_goals", "total_assists", "total_minutes",
    "goals_per_match", "minutes_per_match"
]

X_cluster = df[cluster_features]

# K-Means mesafe bazlı çalıştığı için scaling şart
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("Kümeleme için kullanılan veri şekli:", X_scaled.shape)

# Elbow Method: farklı K değerleri için inertia hesapla
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k} -> Inertia: {kmeans.inertia_:.0f}")

# K=5 ile final K-Means modelini eğit
final_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = final_kmeans.fit_predict(X_scaled)

print("\n=== KÜME DAĞILIMI ===")
print(df["cluster"].value_counts().sort_index())

# Her kümenin ortalama özelliklerine bakalım (yorumlama için)

print("\n=== KÜME BAZINDA ORTALAMA DEĞERLER ===")
cluster_summary = df.groupby("cluster")[cluster_features + ["market_value_in_eur"]].mean()
print(cluster_summary)

from sklearn.metrics import silhouette_score

# Silhouette Score: kümelerin birbirinden ne kadar net ayrıştığını ölçer (0-1 arası, 1'e yakın = iyi)
sil_score = silhouette_score(X_scaled, df["cluster"])
print(f"\nSilhouette Score: {sil_score:.4f}")