"""
AMAÇ: appearances verisini oyuncu bazında toplayıp players tablosuyla birleştirmek,
sonra eksik verileri (performans yok / hedef değer yok) temizlemek.
"""

import pandas as pd

# Verileri oku
players = pd.read_csv("../data/raw/players.csv")
appearances = pd.read_csv("../data/raw/appearances.csv")

# last_season sütununu incele
print("last_season değerleri:", players["last_season"].unique())
print("last_season min/max:", players["last_season"].min(), players["last_season"].max())

# last_season sütununu incele
print("last_season değerleri:", players["last_season"].unique())
print("last_season min/max:", players["last_season"].min(), players["last_season"].max())

# 0. Sadece güncel/aktif oyuncuları tut (son 2 sezon içinde oynamış olanlar)
before = players.shape[0]
players = players[players["last_season"] >= 2023]
after = players.shape[0]
print(f"\nAktif oyuncu filtresi: {before} -> {after} satır ({before - after} satır silindi)")

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

# 9. Doğum tarihinden yaş hesapla
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
today = pd.Timestamp("today")
df["age"] = (today - df["date_of_birth"]).dt.days // 365

# date_of_birth artık gereksiz, yaşı hesapladık
df = df.drop(columns=["date_of_birth"])

print("\n=== YAŞ EKLENDİKTEN SONRA ===")
print(df[["age"]].describe())

# Yaş NaN olanları kontrol et
print("\nYaş hesaplanamayan satır sayısı:", df["age"].isnull().sum())

# Şüpheli yüksek yaşları kontrol et (örn. 45 üstü kaç kişi var?)
print("45 yaş üstü oyuncu sayısı:", (df["age"] > 45).sum())
print(df[df["age"] > 45][["age", "market_value_in_eur"]].head(10))

# 10. Gerçekçi olmayan yaşları filtrele (muhtemelen hatalı/eski kayıtlar)
before = df.shape[0]
df = df[(df["age"] >= 16) & (df["age"] <= 45)]
after = df.shape[0]

print(f"\nYaş filtresi sonrası: {before} -> {after} satır ({before - after} satır silindi)")
print(df["age"].describe())