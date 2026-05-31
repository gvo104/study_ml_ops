import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from src.config import FEATURE_SCHEMA_VERSION, MODEL_DIR
from src.features.build_features import FeatureBuilder
from src.features.numerical import extract_numerical_features
from src.features.preprocess import preprocess_single


@dataclass
class TrainResult:
    accuracy: float
    macro_f1: float
    model_dir: Path


class MentalHealthPredictor:
    def __init__(self, model_dir: Path | str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "xgboost_model.pkl")
        self.encoder = joblib.load(self.model_dir / "label_encoder.pkl")

        feature_builder_path = self.model_dir / "feature_builder.pkl"
        if feature_builder_path.exists():
            self.feature_builder = joblib.load(feature_builder_path)
        else:
            self.feature_builder = self._load_legacy_feature_builder()

        with open(self.model_dir / "metadata.json", encoding="utf-8") as file:
            self.metadata = json.load(file)

    def _load_legacy_feature_builder(self) -> FeatureBuilder:
        """Load vectorizer + SVD from legacy separate pickle files."""
        builder = FeatureBuilder()
        builder.vectorizer = joblib.load(self.model_dir / "vectorizer.pkl")
        builder.svd = joblib.load(self.model_dir / "svd.pkl")
        return builder

    def extract_features(self, text: str) -> np.ndarray:
        if hasattr(self.feature_builder, "transform_single"):
            return self.feature_builder.transform_single(text)

        processed = preprocess_single(text)
        tfidf = self.feature_builder.vectorizer.transform([processed])
        reduced = self.feature_builder.svd.transform(tfidf)
        numerical = extract_numerical_features(text)

        return np.hstack([reduced, numerical])

    def predict(self, text: str) -> dict:
        features = self.extract_features(text)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        predicted_class = self.encoder.inverse_transform([prediction])[0]

        probability_map = {
            self.encoder.inverse_transform([index])[0]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        return {
            "prediction": predicted_class,
            "confidence": probability_map[predicted_class],
            "probabilities": probability_map,
        }


def save_artifacts(
    model,
    feature_builder: FeatureBuilder,
    encoder,
    accuracy: float,
    macro_f1: float,
    model_dir: Path | str = MODEL_DIR,
) -> Path:
    """Persist model, feature pipeline and metadata to disk."""
    model_dir = Path(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_dir / "xgboost_model.pkl")
    joblib.dump(feature_builder, model_dir / "feature_builder.pkl")
    joblib.dump(feature_builder.vectorizer, model_dir / "vectorizer.pkl")
    joblib.dump(feature_builder.svd, model_dir / "svd.pkl")
    joblib.dump(encoder, model_dir / "label_encoder.pkl")

    metadata = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "created_at": str(datetime.now()),
        "classes": list(encoder.classes_),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return model_dir


def load_predictor(model_dir: Path | str = MODEL_DIR) -> MentalHealthPredictor:
    return MentalHealthPredictor(model_dir=model_dir)
