import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class FeatureBuilder:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000
        )

        self.svd = TruncatedSVD(
            n_components=300,
            random_state=101
        )

    def fit_transform(self, texts, numerical_features):

        tfidf = self.vectorizer.fit_transform(texts)

        reduced = self.svd.fit_transform(tfidf)

        return np.hstack([reduced, numerical_features])

    def transform(self, texts, numerical_features):

        tfidf = self.vectorizer.transform(texts)

        reduced = self.svd.transform(tfidf)

        return np.hstack([reduced, numerical_features])