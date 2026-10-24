"""
AMAÇ: Kaydedilmiş modellerle kendi girdiğin bir oyuncunun piyasa değerini tahmin etmek
ve hangi kümeye düştüğünü göstermek.

Çalıştırma (proje klasöründen, önce train_model.py ve cluster_players.py çalışmış olmalı):
    python src/predict.py
Lig kodları Transfermarkt'taki gibidir: GB1 Premier League, ES1 La Liga, IT1 Serie A, L1 Bundesliga,
FR1 Ligue 1, TR1 Süper Lig, PO1 Portekiz, NL1 Hollanda. Uyruk İngilizce yazılır (Turkey, Brazil...).
"""

import sys

import joblib
import pandas as pd

from cluster_players import CLUSTER_FEATURES
from paths import MODELS_DIR
from train_model import CATEGORICAL, NUMERIC

POSITIONS = {
    "Goalkeeper": ["Goalkeeper"],
    "Defender": ["Centre-Back", "Left-Back", "Right-Back"],
    "Midfield": ["Defensive Midfield", "Central Midfield", "Attacking Midfield",
                 "Left Midfield", "Right Midfield"],
    "Attack": ["Centre-Forward", "Left Winger", "Right Winger", "Second Striker"],
}
CLUSTER_NAMES = {0: "Deneyimli, düzenli oynayan", 1: "Genç / az deneyimli", 2: "Kariyerli yıldız / golcü"}


def ask(question, cast=str, choices=None):
    """Soruyu ipucu göstermeden sorar; boş veya geçersiz cevapta tekrar sorar."""
    while True:
        answer = input(f"{question}: ").strip()
        if not answer:
            continue
        if choices and answer not in choices:
            print(f"  Seçenekler: {', '.join(choices)}")
            continue
        try:
            return cast(answer)
        except ValueError:
            print("  Lütfen bir sayı gir.")


def ask_player():
    print("\n--- Oyuncu bilgileri ---")
    position = ask("Mevki", choices=list(POSITIONS))
    sub_position = ask("Alt mevki", choices=POSITIONS[position])
    p = {
        "position": position,
        "sub_position": sub_position,
        "domestic_competition_id": ask("Lig kodu").upper(),
        "country_of_citizenship": ask("Uyruk"),
        "foot": ask("Ayak", choices=["right", "left", "both"]),
        "age": ask("Yaş", float),
        "height_in_cm": ask("Boy (cm)", float),
        "international_caps": ask("Milli maç sayısı", int),
        "international_goals": ask("Milli gol sayısı", int),
        "total_matches": ask("Kariyer maç sayısı", int),
        "total_goals": ask("Kariyer gol sayısı", int),
        "total_assists": ask("Kariyer asist sayısı", int),
        "total_minutes": ask("Kariyer toplam dakika", int),
        "total_yellow_cards": ask("Sarı kart", int),
        "total_red_cards": ask("Kırmızı kart", int),
        "value_1y_ago": ask("Bir yıl önceki piyasa değeri (€)", float),
        "contract_years_left": ask("Sözleşmenin bitmesine kalan yıl", float),
    }
    p["value_1y_ago_age_days"] = 365
    matches = max(p["total_matches"], 1)
    p["goals_per_match"] = p["total_goals"] / matches
    p["assists_per_match"] = p["total_assists"] / matches
    p["minutes_per_match"] = p["total_minutes"] / matches
    return pd.DataFrame([p])


def main():
    try:
        value_model = joblib.load(MODELS_DIR / "market_value_model.joblib")
        cluster_model = joblib.load(MODELS_DIR / "player_clusters.joblib")
    except FileNotFoundError:
        sys.exit("Model bulunamadı. Önce çalıştırın: python src/train_model.py ve python src/cluster_players.py")

    while True:
        player = ask_player()
        value = value_model.predict(player[NUMERIC + CATEGORICAL])[0]
        cluster = int(cluster_model.predict(player[CLUSTER_FEATURES])[0])

        print("\n=== SONUÇ ===")
        print(f"Tahmini piyasa değeri: {value:,.0f} €")
        print(f"Küme: {cluster} - {CLUSTER_NAMES.get(cluster, '?')}")

        if input("\nBaşka oyuncu denemek ister misin? (e/h): ").strip().lower() == "h":
            break


if __name__ == "__main__":
    main()
