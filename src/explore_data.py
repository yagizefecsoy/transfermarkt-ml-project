"""
AMAÇ: Ham veriyi ilk kez inceleme (Exploratory look)

Bu script, data/raw/ klasöründeki üç ana tabloyu (players, valuations,
appearances) okuyup ilk yapısal bakışı atar:
- Kaç satır/sütun var (.shape)
- Hangi sütunlar (özellikler) mevcut (.columns)
- İlk birkaç satır nasıl görünüyor (.head())

SONUÇ / KARARLAR:
- players.csv zaten "market_value_in_eur" (piyasa değeri) sütununu içeriyor
  -> Bu bizim hedef değişkenimiz (target) olacak.
- valuations.csv zaman içindeki değer değişimini tutuyor (bir oyuncu için
  birden fazla satır) -> Bu ödev kapsamında KULLANILMAYACAK, karmaşıklığı
  gereksiz artırıyor (zaman serisi analizi gerektirir).
- appearances.csv maç bazlı performans verisi (gol, asist, dakika vb.)
  -> Oyuncu bazında gruplanıp (toplam gol, toplam dakika vb.) players
  tablosuna eklenecek. Bu işlem prepare_data.py'de yapılıyor.
"""

import pandas as pd

from paths import raw_file

# Ana tabloları oku
players = pd.read_csv(raw_file("players.csv"))
valuations = pd.read_csv(raw_file("player_valuations.csv"))
appearances = pd.read_csv(raw_file("appearances.csv"))

# İlk bakış - her tablonun yapısını gör
print("=== PLAYERS ===")
print(players.shape)
print(players.columns.tolist())
print(players.head())

print("\n=== VALUATIONS ===")
print(valuations.shape)
print(valuations.columns.tolist())
print(valuations.head())

print("\n=== APPEARANCES ===")
print(appearances.shape)
print(appearances.columns.tolist())
print(appearances.head())