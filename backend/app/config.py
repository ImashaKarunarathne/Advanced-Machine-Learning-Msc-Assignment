from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"

# One seed for every split and model so the dashboard numbers are reproducible.
RANDOM_STATE = 42
