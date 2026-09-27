"""
AMAÇ: Temizlenmiş veriyi kullanarak piyasa değeri tahmini için
regresyon modeli eğitmek (supervised learning).

NOT: Log dönüşümü (np.log1p) denendi ama Random Forest'ta R²'yi düşürdüğü
için (0.6169 -> 0.5685) geri alındı. Hedef değişken orijinal (dönüşümsüz)
haliyle kullanılıyor.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Temizlenmiş veriyi oku
df = pd.read_csv("../data/processed/players_cleaned.csv")

# Hedef değişken (y) ve özellikler (X) ayrımı
# player_id modele girmemeli (sadece referans/kimlik bilgisi)
X = df.drop(columns=["player_id", "market_value_in_eur"])
y = df["market_value_in_eur"]

print("X şekli (özellikler):", X.shape)
print("y şekli (hedef):", y.shape)

# Eğitim ve test setlerine ayır (%80 train, %20 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ============================================================
# MODEL 1: Lineer Regresyon (baseline)
# ============================================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\n=== LİNEER REGRESYON SONUÇLARI ===")
print(f"MAE  (Ortalama Mutlak Hata): {mae_lr:,.0f} €")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse_lr:,.0f} €")
print(f"R² Skoru: {r2_lr:.4f}")

# ============================================================
# MODEL 2: Random Forest
# ============================================================
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n=== RANDOM FOREST SONUÇLARI ===")
print(f"MAE  (Ortalama Mutlak Hata): {mae_rf:,.0f} €")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse_rf:,.0f} €")
print(f"R² Skoru: {r2_rf:.4f}")