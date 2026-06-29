from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.config import DATA_PATH, MODEL_DIR, PROJECT_DIR


MODEL_FILES = (
    "xgboost_model.pkl",
    "label_encoder.pkl",
    "metadata.json",
)


def dataset_available(data_path: Path = DATA_PATH) -> bool:
    return data_path.exists() and data_path.is_file() and data_path.stat().st_size > 0


def model_available(model_dir: Path = MODEL_DIR) -> bool:
    return all((model_dir / filename).exists() for filename in MODEL_FILES)


def _configure_dvc_remote() -> None:
    endpoint_url = os.environ.get("DVC_S3_ENDPOINT_URL") or os.environ.get(
        "MLFLOW_S3_ENDPOINT_URL"
    )
    if not endpoint_url:
        return

    subprocess.run(
        ["dvc", "remote", "modify", "origin", "endpointurl", endpoint_url],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_dataset(data_path: Path = DATA_PATH) -> tuple[bool, str]:
    if dataset_available(data_path):
        return True, f"Dataset is ready at {data_path}."

    dvc_file = Path(f"{data_path}.dvc")
    if dvc_file.exists():
        try:
            _configure_dvc_remote()
            completed = subprocess.run(
                ["dvc", "pull", str(dvc_file.relative_to(PROJECT_DIR))],
                cwd=PROJECT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            if dataset_available(data_path):
                details = completed.stdout.strip() or completed.stderr.strip()
                message = f"Dataset restored with DVC from {dvc_file}."
                if details:
                    message = f"{message} {details}"
                return True, message
        except FileNotFoundError:
            return False, "DVC is not installed in the web application container."
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            message = "DVC pull failed while restoring the training dataset."
            if details:
                message = f"{message} {details}"
            return False, message

    raw_dir = PROJECT_DIR / "data" / "raw"
    raw_file = raw_dir / "fin_data.csv"
    if raw_file.exists():
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.data.make_dataset",
                    str(raw_dir),
                    str(data_path.parent),
                ],
                cwd=PROJECT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            if dataset_available(data_path):
                details = completed.stdout.strip() or completed.stderr.strip()
                message = f"Dataset prepared from raw data in {raw_dir}."
                if details:
                    message = f"{message} {details}"
                return True, message
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            message = "Dataset preparation from raw data failed."
            if details:
                message = f"{message} {details}"
            return False, message

    return (
        False,
        "Training dataset is unavailable. Add data/raw/fin_data.csv or make "
        "data/processed/fin_data.csv available through DVC.",
    )
