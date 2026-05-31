from src.features.preprocess import clean_text, preprocess_single


def test_clean_text_lowercase():
    assert clean_text("HELLO World") == "hello world"


def test_preprocess_single_returns_stemmed_string():
    result = preprocess_single("I am feeling sad today.")
    assert isinstance(result, str)
    assert len(result) > 0
