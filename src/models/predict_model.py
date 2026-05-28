import json
import re

import click
import joblib
import numpy as np

from src.config import MODEL_DIR
from src.features.preprocess import preprocess_single


class MentalHealthPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model = joblib.load(self.model_dir / "xgboost_model.pkl")
        self.vectorizer = joblib.load(self.model_dir / "vectorizer.pkl")
        self.svd = joblib.load(self.model_dir / "svd.pkl")
        self.encoder = joblib.load(self.model_dir / "label_encoder.pkl")

        with open(self.model_dir / "metadata.json", "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

    def extract_features(self, text: str):
        processed = preprocess_single(text)
        tfidf = self.vectorizer.transform([processed])
        reduced = self.svd.transform(tfidf)

        num_chars = len(text)
        num_sentences = len(re.findall(r"[.!?]+", text)) or 1
        numerical = np.array([[num_chars, num_sentences]])

        return np.hstack([reduced, numerical])

    def predict(self, text: str):
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


@click.command()
@click.argument("text")
def main(text):
    predictor = MentalHealthPredictor()
    result = predictor.predict(text)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
