import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    NGRAM_RANGE,
    RANDOM_STATE,
    SVD_COMPONENTS,
    TFIDF_MAX_FEATURES,
)
from src.config_loader import FeaturesConfig
from src.features.numerical import extract_numerical_features
from src.features.preprocess import preprocess_single


class FeatureBuilder:
    def __init__(self, features_config: FeaturesConfig | None = None):
        cfg = features_config or FeaturesConfig(
            tfidf_max_features=TFIDF_MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            svd_components=SVD_COMPONENTS,
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=cfg.ngram_range,
            max_features=cfg.tfidf_max_features,
        )
        self.svd_components = cfg.svd_components
        self.svd = self._make_svd(cfg.svd_components)

    def _make_svd(self, n_components: int) -> TruncatedSVD:
        return TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

    def fit_transform(self, texts, numerical_features):
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.svd_components, tfidf.shape[1])
        self.svd = self._make_svd(n_components)
        reduced = self.svd.fit_transform(tfidf)

        return np.hstack([reduced, numerical_features])

    def transform(self, texts, numerical_features):
        tfidf = self.vectorizer.transform(texts)
        reduced = self.svd.transform(tfidf)

        return np.hstack([reduced, numerical_features])

    def transform_single(self, raw_text: str) -> np.ndarray:
        """Build feature vector from raw input text (inference path)."""
        processed = preprocess_single(raw_text)
        tfidf = self.vectorizer.transform([processed])
        reduced = self.svd.transform(tfidf)
        numerical = extract_numerical_features(raw_text)

        return np.hstack([reduced, numerical])
