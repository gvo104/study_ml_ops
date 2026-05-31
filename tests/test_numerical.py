import pytest

from src.features.numerical import extract_numerical_features


def test_single_sentence():
    features = extract_numerical_features("No punctuation here")
    assert features[0, 1] == 1


def test_multiple_sentences():
    features = extract_numerical_features("One. Two! Three?")
    assert features[0, 1] == 3
