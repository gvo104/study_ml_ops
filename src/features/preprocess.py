import multiprocessing as mp
import re

from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize


stemmer = PorterStemmer()


def ensure_nltk_resources():
    return None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def stem_text(text: str):
    tokens = wordpunct_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens]

    return " ".join(stemmed)


def preprocess_single(text: str):
    ensure_nltk_resources()
    cleaned = clean_text(text)

    return stem_text(cleaned)


def preprocess_parallel(texts, n_jobs=None):
    ensure_nltk_resources()
    n_jobs = n_jobs or mp.cpu_count()

    with mp.Pool(n_jobs) as pool:
        processed = pool.map(preprocess_single, texts)

    return processed
