import re
import multiprocessing as mp

import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()


def clean_text(text: str) -> str:

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def stem_text(text: str):

    tokens = word_tokenize(text)

    stemmed = [stemmer.stem(token) for token in tokens]

    return " ".join(stemmed)


def preprocess_single(text: str):

    cleaned = clean_text(text)

    return stem_text(cleaned)


def preprocess_parallel(texts, n_jobs=None):

    n_jobs = n_jobs or mp.cpu_count()

    with mp.Pool(n_jobs) as pool:

        processed = pool.map(
            preprocess_single,
            texts
        )

    return processed