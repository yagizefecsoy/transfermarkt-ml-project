"""
AMAÇ: appearances verisini oyuncu bazında toplayıp players tablosuyla birleştirmek,
sonra eksik/hatalı kayıtları temizlemek.

ÖNEMLİ: Burada yalnızca SATIR BAZINDA işlemler yapılır (filtreleme, birleştirme, oran hesaplama).
Veri setinin geneline bakarak öğrenilen adımlar (medyanla doldurma, nadir ülkeleri "Other" yapma,
one-hot encoding) train_model.py içindeki pipeline'da, eğitim/test ayrımından SONRA yapılır.
Böylece test verisi bu adımları etkilemez (veri sızıntısı olmaz).

Çalıştırma (proje klasöründen): python src/prepare_data.py
"""

import pandas as pd

from paths import PROCESSED_FILE, raw_file

ACTIVE_SINCE_SEASON = 2023  # son 3 sezonda oynamış oyuncular
MIN_AGE, MAX_AGE = 16, 45


def main():
    players = pd.read_csv(raw_file("players.csv"))
    appearances = pd.read_csv(raw_file("appearances.csv"))
    clubs = pd.read_csv(raw_file("clubs.csv"))

    # Yaş için sabit referans tarihi: veri setindeki son maç tarihi.
    # "Bugün" kullanılsaydı script her çalıştırıldığında yaşlar (ve sonuçlar) değişirdi.
    appearances["date"] = pd.to_datetime(appearances["date"])
    reference_date = appearances["date"].max()
    print("Yaş referans tarihi (son maç tarihi):", reference_date.date())

    # 0. Sadece güncel/aktif oyuncuları tut
    before = players.shape[0]
    players = players[players["last_season"] >= ACTIVE_SINCE_SEASON]
    print(f"Aktif oyuncu filtresi: {before} -> {players.shape[0]} satır")

    # 1. Appearances'ı oyuncu bazında grupla (her oyuncu için kariyer toplamları)
    performance = appearances.groupby("player_id").agg(
        total_matches=("appearance_id", "count"),
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_minutes=("minutes_played", "sum"),
        total_yellow_cards=("yellow_cards", "sum"),
        total_red_cards=("red_cards", "sum"),
    ).reset_index()

    # 2. Players ile performance'ı birleştir; performans verisi olmayanları çıkar
    df = players.merge(performance, on="player_id", how="inner")
    print("Performans verisi olan aktif oyuncu:", df.shape[0])

    # 3. Hedef değişkeni (market_value_in_eur) eksik olanları çıkar
    df = df.dropna(subset=["market_value_in_eur"])
    print("Hedef değeri olan oyuncu:", df.shape[0])

    # 4. Kulübün ligini ekle
    clubs_selected = clubs[["club_id", "domestic_competition_id"]]
    df = df.merge(clubs_selected, left_on="current_club_id", right_on="club_id", how="left")

    # 5. Performans oranları (ham toplamlardan daha anlamlı sinyal)
    df["goals_per_match"] = df["total_goals"] / df["total_matches"]
    df["assists_per_match"] = df["total_assists"] / df["total_matches"]
    df["minutes_per_match"] = df["total_minutes"] / df["total_matches"]

    # 6. Yaş (sabit referans tarihine göre, tam yıl)
    birth = pd.to_datetime(df["date_of_birth"])
    df["age"] = (reference_date - birth).dt.days / 365.25

    before = df.shape[0]
    df = df[df["age"].between(MIN_AGE, MAX_AGE)]  # doğum tarihi eksik/hatalı olanlar da düşer
    print(f"Yaş filtresi ({MIN_AGE}-{MAX_AGE}): {before} -> {df.shape[0]} satır")

    # 7. Kategorik alanı olmayan az sayıdaki kaydı çıkar
    df = df.dropna(subset=["sub_position", "domestic_competition_id"])

    # 8. Milli takım bilgisi boşsa oyuncu milli takımda oynamamıştır -> 0 (satır bazında, sızıntı yok)
    df["international_caps"] = df["international_caps"].fillna(0)
    df["international_goals"] = df["international_goals"].fillna(0)
    # height_in_cm ve foot eksikleri burada doldurulmaz: pipeline eğitim verisine göre doldurur.

    columns_to_keep = [
        "player_id", "market_value_in_eur",
        "age", "height_in_cm", "foot",
        "position", "sub_position", "country_of_citizenship", "domestic_competition_id",
        "international_caps", "international_goals",
        "total_matches", "total_goals", "total_assists", "total_minutes",
        "total_yellow_cards", "total_red_cards",
        "goals_per_match", "assists_per_match", "minutes_per_match",
    ]
    df = df[columns_to_keep].sort_values("player_id").reset_index(drop=True)

    print("\nEksik değerler (pipeline dolduracak):")
    print(df.isnull().sum()[lambda s: s > 0])

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"\nTemizlenmiş veri: {df.shape[0]} oyuncu, {df.shape[1]} sütun -> {PROCESSED_FILE.name}")


if __name__ == "__main__":
    main()
