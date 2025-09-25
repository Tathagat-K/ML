from pathlib import Path

# project roots
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# ensure dirs exist
for p in (DATA, RAW, INTERIM, PROCESSED, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)
