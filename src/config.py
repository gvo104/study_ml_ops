import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PATH = PROCESSED_DATA_DIR / "fin_data.csv"

MODEL_DIR = PROJECT_DIR / "models" / "xgboost"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "xgboost_baseline.yaml"

RANDOM_STATE = 101
TEST_SIZE = 0.2

TEXT_COLUMN = "tokens_stemmed"
RAW_TEXT_COLUMN = "statement"
TARGET_COLUMN = "status"
NUMERICAL_COLUMNS = [
    "num_of_characters",
    "num_of_sentences",
]

REPREPROCESS_TEXT = False
COMPUTE_NUMERICAL_FROM_RAW = False

TFIDF_MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 300

FEATURE_SCHEMA_VERSION = "1.0"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "mental_health_classification",
)
MLFLOW_ENABLED = os.getenv(
    "MLFLOW_ENABLED",
    "true",
).lower() in {"1", "true", "yes"}

# MinIO / S3 defaults for local MLflow artifact store
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://localhost:9000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", AWS_ENDPOINT_URL)
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def configure_mlflow_environment() -> None:
    """Expose MinIO/S3 credentials to boto3 used by MLflow artifact uploads."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY)
    os.environ.setdefault("AWS_ENDPOINT_URL", AWS_ENDPOINT_URL)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)
    os.environ.setdefault("AWS_DEFAULT_REGION", AWS_DEFAULT_REGION)
