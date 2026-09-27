"""
AMAÇ: appearances verisini oyuncu bazında toplayıp players tablosuyla birleştirmek,
sonra eksik verileri (performans yok / hedef değer yok) temizlemek.
"""

import pandas as pd

# Verileri oku
players = pd.read_csv("../data/raw/players.csv")
appearances = pd.read_csv("../data/raw/appearances.csv")

# 1. Appearances'ı oyuncu bazında grupla (her oyuncu için toplamlar)
performance = appearances.groupby("player_id").agg(
    total_matches=("appearance_id", "count"),
    total_goals=("goals", "sum"),
    total_assists=("assists", "sum"),
    total_minutes=("minutes_played", "sum"),
    total_yellow_cards=("yellow_cards", "sum"),
    total_red_cards=("red_cards", "sum")
).reset_index()

print("=== PERFORMANCE (gruplanmış) ===")
print(performance.shape)
print(performance.head())

# 2. Players ile performance'ı birleştir
df = players.merge(performance, on="player_id", how="left")

print("\n=== BİRLEŞTİRİLMİŞ VERİ ===")
print(df.shape)
print(df.columns.tolist())

# 3. Kaç oyuncunun performans verisi eksik (appearances'ta hiç kaydı yok)?
print("\nPerformans verisi olmayan oyuncu sayısı:", df["total_matches"].isnull().sum())

# 4. Performans verisi olmayan oyuncuları sil (hedefimiz performansa dayalı tahmin)
df = df.dropna(subset=["total_matches"])

print("\n=== PERFORMANS VERİSİ OLMAYANLAR SİLİNDİKTEN SONRA ===")
print(df.shape)

# 5. Hedef değişkeni (market_value_in_eur) eksik olanları kontrol et
print("\nHedef değişkeni eksik olan oyuncu sayısı:", df["market_value_in_eur"].isnull().sum())

# 6. Hedef değişkeni eksik olanları sil
df = df.dropna(subset=["market_value_in_eur"])

print("\n=== HEDEF DEĞERİ EKSİK OLANLAR SİLİNDİKTEN SONRA ===")
print(df.shape)

# 6.5. Kulüp bilgisini ekle (sadece lig - en güvenilir kulüp gücü göstergesi)
clubs = pd.read_csv("../data/raw/clubs.csv")

clubs_selected = clubs[["club_id", "domestic_competition_id"]]

df = df.merge(clubs_selected, left_on="current_club_id", right_on="club_id", how="left")

print("\n=== KULÜP BİLGİSİ EKLENDİKTEN SONRA ===")
print(df.shape)

# 7. Performans oranlarını türet (ham toplamlardan daha anlamlı sinyal)
df["goals_per_match"] = df["total_goals"] / df["total_matches"]
df["minutes_per_match"] = df["total_minutes"] / df["total_matches"]

# 8. Gereksiz sütunları çıkar (kimlik bilgisi, metin, veri sızıntısı riski taşıyanlar)
columns_to_keep = [
    "player_id", "country_of_citizenship", "date_of_birth",
    "sub_position", "position", "foot", "height_in_cm",
    "international_caps", "international_goals",
    "domestic_competition_id",
    "market_value_in_eur",
    "total_matches", "total_goals", "total_assists",
    "total_minutes", "total_yellow_cards", "total_red_cards",
    "goals_per_match", "minutes_per_match"
]
df = df[columns_to_keep]

print("\n=== SADECE GEREKLİ SÜTUNLAR ===")
print(df.shape)
print(df.columns.tolist())
print(df.head())