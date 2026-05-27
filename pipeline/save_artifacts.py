import os
import json
import joblib

from datetime import datetime

from pipeline.config import MODEL_DIR


def save_artifacts(
    model,
    feature_builder,
    encoder,
    accuracy
):

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "xgboost_model.pkl"
    )

    joblib.dump(
        feature_builder.vectorizer,
        MODEL_DIR / "vectorizer.pkl"
    )

    joblib.dump(
        feature_builder.svd,
        MODEL_DIR / "svd.pkl"
    )

    joblib.dump(
        encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    metadata = {
        "accuracy": accuracy,
        "created_at": str(datetime.now()),
        "classes": list(encoder.classes_)
    }

    with open(
        MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Artifacts saved successfully.")