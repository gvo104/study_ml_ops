from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "fin_data.csv"

MODEL_DIR = BASE_DIR / "saved_models" / "xgboost"

RANDOM_STATE = 101

TEST_SIZE = 0.2

TFIDF_MAX_FEATURES = 5000

NGRAM_RANGE = (1, 2)

SVD_COMPONENTS = 300