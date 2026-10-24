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
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paths import FIGURES_DIR, MODELS_DIR, PROCESSED_FILE, RANDOM_STATE, REPORTS_DIR

TARGET = "market_value_in_eur"
CATEGORICAL = ["position", "sub_position", "foot", "country_of_citizenship", "domestic_competition_id"]
NUMERIC = [
    "age", "height_in_cm", "international_caps", "international_goals",
    "total_matches", "total_goals", "total_assists", "total_minutes",
    "total_yellow_cards", "total_red_cards",
    "goals_per_match", "assists_per_match", "minutes_per_match",
]


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # boy eksikse eğitim verisinin medyanı
        ("scale", StandardScaler()),                   # ağaçlar için gereksiz ama Ridge için şart
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # Eğitim verisinde 100'den az oyuncusu olan ülkeler/ligler tek bir "infrequent" sütununda toplanır
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def log_target(regressor):
    """Modeli log1p(değer) üzerinde eğitir; predict() sonucu yine € cinsindendir."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor()), ("model", regressor)]),
        func=np.log1p, inverse_func=np.expm1,
    )


# Aday modeller ve denenecek ayarlar
CANDIDATES = {
    "Ridge Regression (baseline)": (
        log_target(Ridge()),
        {"regressor__model__alpha": [0.1, 1, 10, 100]},
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
        "rmse_eur": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_abs_error_eur": median_absolute_error(y_true, y_pred),
        "median_pct_error": float(np.median(ape)),
        "within_50pct": float(np.mean(ape <= 0.5)),
    }


def main():
    df = pd.read_csv(PROCESSED_FILE)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Eğitim: {len(X_train)}  Test: {len(X_test)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Eski yaklaşım (karşılaştırma için): log dönüşümü YOK, ayar araması YOK
    old_rf = Pipeline([("prep", build_preprocessor()),
                       ("model", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])
    old_rf.fit(X_train, y_train)

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
            "selected_model": test_metrics(y_test, y_pred),
        },
    }
    for name, m in metrics["test"].items():
        print(f"{name:20s} R²={m['r2']:.3f}  MAE={m['mae_eur']:>11,.0f} €  "
              f"MedAE={m['median_abs_error_eur']:>9,.0f} €  MedAPE={m['median_pct_error']:.1%}  "
              f"±%50 içinde={m['within_50pct']:.1%}")

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

    (REPORTS_DIR / "metrics_supervised.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nKaydedildi: models/market_value_model.joblib, reports/metrics_supervised.json, reports/figures/")


if __name__ == "__main__":
    main()
