import pytest
from email_classifier.data_handling.vectorizer.base_text_vectorizer import BaseTextVectorizer
from email_classifier.data_handling.vectorizer.bow_vectorizer import BOWVectorizer
from email_classifier.data_handling.vectorizer.tfidf_vectorizer import TfidfVectorizer
from email_classifier.data_handling.vectorizer.factory import VectorizerFactory

def test_base_text_vectorizer_instantiation():
    """
    Test that BaseTextVectorizer cannot be instantiated directly.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseTextVectorizer"):
        BaseTextVectorizer()


def test_base_text_vectorizer_feature_names_error():
    """
    Test that get_feature_names raises an error when not supported.
    """
    class DummyVectorizer(BaseTextVectorizer):
        def vectorize(self, documents):
            return []

    dummy_vectorizer = DummyVectorizer()
    with pytest.raises(NotImplementedError, match="This vectorizer does not support feature extraction"):
        dummy_vectorizer.get_feature_names()


def test_bow_vectorizer_vectorize():
    """
    Test that BOWVectorizer vectorizes documents correctly.
    """
    vectorizer = BOWVectorizer(max_features=3, stop_words=None)
    documents = ["unique alpha beta gamma", "unique delta epsilon zeta"]
    transformed = vectorizer.vectorize(documents)

    # Assert transformed output shape
    assert transformed.shape == (2, 3), "Expected 2 documents with 3 features."


def test_bow_vectorizer_feature_names():
    """
    Test that BOWVectorizer returns correct feature names.
    """
    vectorizer = BOWVectorizer(max_features=3, stop_words=None)
    documents = ["alpha beta gamma", "delta epsilon zeta"]
    vectorizer.vectorize(documents)

    feature_names = vectorizer.get_feature_names()
    assert len(feature_names) == 3, "Expected 3 feature names."
    assert "alpha" in feature_names, "Expected 'alpha' in feature names."


def test_tfidf_vectorizer_vectorize():
    """
    Test that TfidfVectorizer vectorizes documents correctly.
    """
    vectorizer = TfidfVectorizer(max_features=3, stop_words=None)
    documents = ["unique alpha beta gamma", "unique delta epsilon zeta"]
    transformed = vectorizer.vectorize(documents)

    # Assert transformed output shape
    assert transformed.shape == (2, 3), "Expected 2 documents with 3 features."


def test_tfidf_vectorizer_feature_names():
    """
    Test that TfidfVectorizer returns correct feature names.
    """
    vectorizer = TfidfVectorizer(max_features=3, stop_words=None)
    documents = ["alpha beta gamma", "delta epsilon zeta"]
    vectorizer.vectorize(documents)

    feature_names = vectorizer.get_feature_names()
    assert len(feature_names) == 3, "Expected 3 feature names."
    assert "beta" in feature_names, "Expected 'beta' in feature names."


def test_vectorizer_factory_get_vectorizer():
    """
    Test that VectorizerFactory returns the correct vectorizer instance.
    """
    tfidf_vectorizer = VectorizerFactory.get_vectorizer("tfidf", max_features=3, stop_words=None)
    bow_vectorizer = VectorizerFactory.get_vectorizer("bow", max_features=3, stop_words=None)

    assert isinstance(tfidf_vectorizer, TfidfVectorizer), "Expected TfidfVectorizer instance."
    assert isinstance(bow_vectorizer, BOWVectorizer), "Expected BOWVectorizer instance."


def test_vectorizer_factory_unsupported_type():
    """
    Test that VectorizerFactory raises an error for unsupported vectorizer types.
    """
    with pytest.raises(ValueError, match="Unsupported vectorizer type: unsupported"):
        VectorizerFactory.get_vectorizer("unsupported")
