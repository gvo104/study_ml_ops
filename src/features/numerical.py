import re

import numpy as np


def extract_numerical_features(text: str) -> np.ndarray:
    """Extract num_chars and num_sentences from raw text.

    Same formula used at training (when computed from text) and inference.
    """
    num_chars = len(text)
    num_sentences = len(re.findall(r"[.!?]+", text)) or 1

    return np.array([[num_chars, num_sentences]])
