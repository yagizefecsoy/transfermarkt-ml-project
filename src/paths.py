"""Proje klasör yolları. Scriptler hangi klasörden çalıştırılırsa çalıştırılsın doğru dosyaları bulur."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "players_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42


def raw_file(name):
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Transfermarkt veri setini README'deki adımlarla data/raw/ klasörüne koyun."
        )
    return path
