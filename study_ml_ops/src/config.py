from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATA_PATH = PROCESSED_DATA_DIR / "fin_data.csv"

MODEL_DIR = PROJECT_DIR / "models" / "xgboost"

RANDOM_STATE = 101
TEST_SIZE = 0.2

TEXT_COLUMN = "tokens_stemmed"
TARGET_COLUMN = "status"
NUMERICAL_COLUMNS = [
    "num_of_characters",
    "num_of_sentences",
]

TFIDF_MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 300
