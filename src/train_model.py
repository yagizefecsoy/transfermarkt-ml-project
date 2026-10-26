"""
AMAÇ: Temizlenmiş veriyle piyasa değeri tahmini için regresyon modeli eğitmek (supervised learning).

Adımlar:
1. Eğitim/test ayrımı (%80/%20). Test verisi en sona kadar kenarda durur.
2. Ön işleme pipeline'ın İÇİNDE: eksik doldurma, nadir ülkeleri "Other" yapma, one-hot encoding.
   Pipeline çapraz doğrulamanın her katında sadece o katın eğitim parçasından öğrenir.
3. Hedef log ölçeğinde öğrenilir (log1p), tahminler € ölçeğine geri çevrilir (expm1).
   Piyasa değeri çok çarpık: medyan 0,8M €, en yüksek 200M €. Log olmadan model hatası
   birkaç yıldız oyuncuya göre şekillenir.
4. Üç model 5-katlı CV ile karşılaştırılır; seçim ölçütü € cinsinden MAE.
5. Seçilen model test verisinde BİR KEZ değerlendirilir ve kaydedilir.
   Karşılaştırma için aynı test oyuncularında iki eski sürüm de ölçülür:
   ilk yaklaşım (Random Forest, log yok) ve ek özelliklerden önceki model (Gradient Boosting).

Çalıştırma (proje klasöründen): python src/train_model.py   (~3-5 dk)
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from paths import FIGURES_DIR, MODELS_DIR, PROCESSED_FILE, RANDOM_STATE, REPORTS_DIR

TARGET = "market_value_in_eur"
CATEGORICAL = ["position", "sub_position", "foot", "country_of_citizenship", "domestic_competition_id"]
BASE_NUMERIC = [
    "age", "height_in_cm", "international_caps", "international_goals",
    "total_matches", "total_goals", "total_assists", "total_minutes",
    "total_yellow_cards", "total_red_cards",
    "goals_per_match", "assists_per_match", "minutes_per_match",
]
# Ek özellikler (prepare_data.py adım 9-10): bir yıl önceki değer ve sözleşmenin bitmesine kalan yıl
EXTRA_NUMERIC = ["value_1y_ago", "value_1y_ago_age_days", "contract_years_left"]
NUMERIC = BASE_NUMERIC + EXTRA_NUMERIC


def build_preprocessor(numeric_columns=NUMERIC):
    numeric = Pipeline([
        # Eksikler eğitim verisinin medyanıyla doldurulur; add_indicator "bu değer eksikti"
        # bilgisini ayrı bir sütun olarak saklar (ör. bir yıl önceki değeri olmayan oyuncu)
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),                   # ağaçlar için gereksiz ama Ridge için şart
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # Eğitim verisinde 100'den az oyuncusu olan ülkeler/ligler tek bir "infrequent" sütununda toplanır
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                 sparse_output=False)),
    ])
    # Euro cinsinden önceki değer çok çarpık: doğrusal model (Ridge) için log ölçeğine alınır.
    # Ağaç modelleri sıralamaya baktığı için bu dönüşümden etkilenmez.
    money = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    money_columns = [c for c in numeric_columns if c == "value_1y_ago"]
    other_columns = [c for c in numeric_columns if c not in money_columns]
    return ColumnTransformer([("num", numeric, other_columns), ("money", money, money_columns),
                              ("cat", categorical, CATEGORICAL)])


def log_target(regressor, numeric_columns=NUMERIC):
    """Modeli log1p(değer) üzerinde eğitir; predict() sonucu yine € cinsindendir."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor(numeric_columns)), ("model", regressor)]),
        func=np.log1p, inverse_func=np.expm1,
    )


# Aday modeller ve denenecek ayarlar
CANDIDATES = {
    "Ridge Regression (baseline)": (
        log_target(Ridge()),
        {"regressor__model__alpha": [0.1, 1, 10, 100, 1000]},
    ),
    "Random Forest": (
        log_target(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        {"regressor__model__min_samples_leaf": [1, 3, 10],
         "regressor__model__max_features": [0.33, 1.0]},
    ),
    "Gradient Boosting": (
        log_target(HistGradientBoostingRegressor(max_iter=500, random_state=RANDOM_STATE)),
        {"regressor__model__learning_rate": [0.03, 0.1],
         "regressor__model__max_leaf_nodes": [15, 31, 63]},
    ),
}


def test_metrics(y_true, y_pred):
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "r2": r2_score(y_true, y_pred),
        "mae_eur": mean_absolute_error(y_true, y_pred),
        "mse_eur2": float(mean_squared_error(y_true, y_pred)),
        "rmse_eur": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_abs_error_eur": median_absolute_error(y_true, y_pred),
        "median_pct_error": float(np.median(ape)),
        "within_50pct": float(np.mean(ape <= 0.5)),
        "off_by_2x_or_more": float(np.mean((y_pred / y_true < 0.5) | (y_pred / y_true > 2))),
    }


def main():
    df = pd.read_csv(PROCESSED_FILE)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Eğitim: {len(X_train)}  Test: {len(X_test)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Karşılaştırma 1 - ilk yaklaşım: ek özellik YOK, log dönüşümü YOK, ayar araması YOK
    old_rf = Pipeline([("prep", build_preprocessor(BASE_NUMERIC)),
                       ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
    old_rf.fit(X_train, y_train)
    # Karşılaştırma 2 - ek özelliklerden önceki model: log hedefli Gradient Boosting, ek özellik YOK
    # (ayarlar, o sürümün CV ile seçtiği değerler: learning_rate=0.1, max_leaf_nodes=15)
    previous_gb = log_target(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=15,
                                                           random_state=RANDOM_STATE), BASE_NUMERIC)
    previous_gb.fit(X_train, y_train)

    results = {}
    for name, (model, grid) in CANDIDATES.items():
        search = GridSearchCV(model, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        i = search.best_index_
        results[name] = {
            "search": search,
            "cv_mae_eur": float(-search.cv_results_["mean_test_score"][i]),
            "cv_mae_std": float(search.cv_results_["std_test_score"][i]),
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
        }
        print(f"{name:28s} CV MAE = {results[name]['cv_mae_eur']:>11,.0f} € "
              f"± {results[name]['cv_mae_std']:>9,.0f}  {results[name]['best_params']}")

    best_name = min(results, key=lambda n: results[n]["cv_mae_eur"])
    best_model = results[best_name]["search"].best_estimator_
    print(f"\nSeçilen model: {best_name}")

    y_pred = best_model.predict(X_test)
    metrics = {
        "data": {"players": len(df), "train": len(X_train), "test": len(X_test),
                 "target_median_eur": float(y.median()), "target_max_eur": float(y.max())},
        "cv_comparison": {n: {k: v for k, v in r.items() if k != "search"} for n, r in results.items()},
        "selected_model": best_name,
        "test": {
            "old_rf_raw_target": test_metrics(y_test, old_rf.predict(X_test)),
            "previous_gb_without_extra_features": test_metrics(y_test, previous_gb.predict(X_test)),
            "selected_model": test_metrics(y_test, y_pred),
        },
    }
    for name, m in metrics["test"].items():
        print(f"{name:35s} R²={m['r2']:.3f}  MAE={m['mae_eur']:>11,.0f} €  "
              f"MedAE={m['median_abs_error_eur']:>9,.0f} €  MedAPE={m['median_pct_error']:.1%}  "
              f"±%50 içinde={m['within_50pct']:.1%}  2x+ yanlış={m['off_by_2x_or_more']:.1%}")

    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "market_value_model.joblib")

    # Grafik 1: gerçek vs tahmin (log eksen; köşegen = kusursuz tahmin)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, y_pred, s=5, alpha=0.3)
    lims = [1e4, 3e8]
    ax.plot(lims, lims, color="#E45756", lw=1)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel="Gerçek piyasa değeri (€)", ylabel="Tahmin (€)",
           title=f"Test verisi: {best_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gercek_vs_tahmin.png", dpi=150)
    plt.close(fig)

    # Grafik 2: özellik önemi (permutation importance, test verisinde, MAE artışı)
    imp = permutation_importance(best_model, X_test, y_test, scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    metrics["permutation_importance_mae_eur"] = importance.sort_values(ascending=False).round(0).to_dict()
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.plot.barh(ax=ax, color="#4C78A8")
    ax.set(xlabel="Karıştırılınca MAE artışı (€)", title="Özellik önemi (test verisi)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ozellik_onemi.png", dpi=150)
    plt.close(fig)

    # Grafik 3: öğrenme eğrisi. Eğitim verisi büyüdükçe eğitim ve doğrulama hatası (MAE) nasıl değişiyor?
    # Doğrulama hatası hâlâ düşüyorsa daha fazla veri işe yarar; iki eğri arasındaki açıklık ezberlemeyi gösterir.
    sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error", train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
    train_mae, val_mae = -train_scores / 1e6, -val_scores / 1e6
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for score, label, color in [(train_mae, "Eğitim", "#4C78A8"), (val_mae, "Doğrulama (CV)", "#E45756")]:
        ax.plot(sizes, score.mean(axis=1), "o-", color=color, label=label)
        ax.fill_between(sizes, score.mean(axis=1) - score.std(axis=1), score.mean(axis=1) + score.std(axis=1),
                        color=color, alpha=0.15)
    ax.set(xlabel="Eğitimdeki oyuncu sayısı", ylabel="MAE (milyon €)", title=f"Öğrenme eğrisi: {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ogrenme_egrisi.png", dpi=150)
    plt.close(fig)
    metrics["learning_curve"] = {"train_size": sizes.tolist(), "train_mae_eur": (train_mae.mean(axis=1) * 1e6).tolist(),
                                 "validation_mae_eur": (val_mae.mean(axis=1) * 1e6).tolist()}

    (REPORTS_DIR / "metrics_supervised.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nKaydedildi: models/market_value_model.joblib, reports/metrics_supervised.json, reports/figures/")


if __name__ == "__main__":
    main()
