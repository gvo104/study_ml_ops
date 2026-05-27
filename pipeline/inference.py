import re
import json
import joblib
import numpy as np

from pipeline.config import MODEL_DIR
from pipeline.preprocess import preprocess_single


class MentalHealthPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_DIR / "xgboost_model.pkl"
        )

        self.vectorizer = joblib.load(
            MODEL_DIR / "vectorizer.pkl"
        )

        self.svd = joblib.load(
            MODEL_DIR / "svd.pkl"
        )

        self.encoder = joblib.load(
            MODEL_DIR / "label_encoder.pkl"
        )

        with open(
            MODEL_DIR / "metadata.json",
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

    def extract_features(self, text: str):

        processed = preprocess_single(text)

        tfidf = self.vectorizer.transform([processed])

        reduced = self.svd.transform(tfidf)

        num_chars = len(text)

        num_sentences = len(
            re.findall(r"[.!?]+", text)
        ) or 1

        numerical = np.array([
            [num_chars, num_sentences]
        ])

        combined = np.hstack([
            reduced,
            numerical
        ])

        return combined

    def predict(self, text: str):

        features = self.extract_features(text)

        prediction = self.model.predict(features)[0]

        probabilities = self.model.predict_proba(
            features
        )[0]

        predicted_class = self.encoder.inverse_transform(
            [prediction]
        )[0]

        probability_map = {
            self.encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        return {
            "prediction": predicted_class,
            "confidence": probability_map[predicted_class],
            "probabilities": probability_map
        }


if __name__ == "__main__":

    predictor = MentalHealthPredictor()

    text = "I feel sad and anxious and cannot sleep."

    result = predictor.predict(text)

    print(json.dumps(result, indent=2))