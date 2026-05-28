import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import NGRAM_RANGE, RANDOM_STATE, SVD_COMPONENTS, TFIDF_MAX_FEATURES


class FeatureBuilder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
        )

        self.svd = TruncatedSVD(
            n_components=SVD_COMPONENTS,
            random_state=RANDOM_STATE,
        )

    def fit_transform(self, texts, numerical_features):
        tfidf = self.vectorizer.fit_transform(texts)
        reduced = self.svd.fit_transform(tfidf)

        return np.hstack([reduced, numerical_features])

    def transform(self, texts, numerical_features):
        tfidf = self.vectorizer.transform(texts)
        reduced = self.svd.transform(tfidf)

        return np.hstack([reduced, numerical_features])
