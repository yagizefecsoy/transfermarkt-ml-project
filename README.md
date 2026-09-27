# Transfermarkt ML Project

Transfermarkt verisi kullanılarak futbol oyuncularının piyasa değerini tahmin eden (supervised) ve oyuncuları özelliklerine göre gruplara ayıran (unsupervised) makine öğrenmesi modelleri.

---

## Proje Yapısı

```
transfermarkt-ml-project/
├── data/
│   ├── raw/               ← Ham veri (gitignore'da, repoya dahil değil)
│   └── processed/         ← Temizlenmiş ve işlenmiş veri
├── src/
│   ├── explore_data.py    ← Ham verilerin ilk incelenmesi
│   ├── prepare_data.py    ← Veri temizleme, birleştirme, feature engineering
│   ├── train_model.py     ← Supervised model eğitimi (regresyon)
│   └── cluster_players.py ← Unsupervised model (kümeleme)
└── models/                ← Eğitilmiş modeller (henüz boş)
```

---

## Veri Seti

Kaynak: [Transfermarkt Kaggle Dataset](https://www.kaggle.com/)

Kullanılan tablolar:
- `players.csv` → Oyuncu demografik bilgileri ve piyasa değeri
- `appearances.csv` → Maç bazlı performans istatistikleri
- `clubs.csv` → Kulüp ve lig bilgileri

Ham veri boyutu: ~50.000 oyuncu
Temizleme sonrası: **13.406 aktif oyuncu** (last_season ≥ 2023 filtresi)

---

## Kullanılan Özellikler

| Özellik | Açıklama |
|---|---|
| age | Oyuncunun yaşı (date_of_birth'ten hesaplandı) |
| height_in_cm | Boy |
| foot | Baskın ayak (sol/sağ/çift) |
| position / sub_position | Pozisyon bilgisi |
| international_caps | Milli takım maç sayısı |
| international_goals | Milli takım gol sayısı |
| total_matches | Kariyer boyunca toplam maç |
| total_goals | Kariyer boyunca toplam gol |
| total_assists | Kariyer boyunca toplam asist |
| total_minutes | Kariyer boyunca toplam dakika |
| goals_per_match | Maç başına gol oranı |
| minutes_per_match | Maç başına ortalama dakika |
| domestic_competition_id | Oynadığı lig |
| country_of_citizenship | Milliyet |

**Hedef değişken:** `market_value_in_eur` (€ cinsinden piyasa değeri)

---

## Supervised Model: Piyasa Değeri Tahmini

**Problem:** Regresyon (sürekli sayısal değer tahmini)

**Modeller ve Sonuçlar:**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Lineer Regresyon (baseline) | 0.34 | 4.086.186 € | 6.971.090 € |
| Random Forest | **0.62** | **2.245.246 €** | **5.306.288 €** |

**Notlar:**
- Lineer Regresyon baseline olarak kullanıldı — yaş-değer ilişkisinin doğrusal olmaması nedeniyle düşük performans beklenen bir sonuç.
- Log dönüşümü (np.log1p) denendi ama Random Forest için R²'yi düşürdüğü (0.62 → 0.57) için geri alındı.
- R²=0.62 bu problem tipi için makul bir sonuç — piyasa değeri kısmen insan kararına dayandığı için %90+ skor beklentisi gerçekçi değil.

---

## Unsupervised Model: Oyuncu Segmentasyonu

**Problem:** K-Means kümeleme (K=5)

**Elbow Method ile K seçimi:** K=4-5 aralığında dirsek noktası gözlemlendi, K=5 seçildi.

**Küme Sonuçları:**

| Küme | Oyuncu Sayısı | Ort. Yaş | Ort. Değer | Yorum |
|---|---|---|---|---|
| 4 | 197 | 33.5 | 14.9M € | Deneyimli yıldızlar |
| 0 | 1708 | 32.5 | 7.4M € | Tecrübeli, düzenli oynayan |
| 1 | 2472 | 27.6 | 5.6M € | Orta yaş, orta seviye |
| 2 | 4618 | 28.2 | 3.0M € | Uzun boylu, düzenli oynayan |
| 3 | 4411 | 23.6 | 1.8M € | Genç, az oynayan (potansiyel) |

**Silhouette Score:** 0.2145

Model piyasa değerini hiç görmeden gruplar oluşturdu — ama gruplar arasında 1.8M € ile 14.9M € arasında net bir değer farklılaşması ortaya çıktı. Bu, kümelerin gerçek hayatta anlamlı karşılığı olduğunun göstergesi.

---

## Kurulum ve Çalıştırma

```bash
# Repoyu klonla
git clone https://github.com/yagizefecsoy/transfermarkt-ml-project.git
cd transfermarkt-ml-project

# Virtual environment oluştur
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Kütüphaneleri kur
pip install pandas numpy scikit-learn matplotlib seaborn

# Scriptleri sırayla çalıştır
python src/explore_data.py
python src/prepare_data.py
python src/train_model.py
python src/cluster_players.py
```

**Not:** `data/raw/` klasörüne Transfermarkt veri setini manuel olarak eklemeniz gerekir (boyutu nedeniyle repoya dahil edilmedi).

---

## Teknolojiler

- Python 3.10+
- pandas, numpy
- scikit-learn
- Git / GitHub
