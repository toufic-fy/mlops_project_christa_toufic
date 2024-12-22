from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from .base_text_vectorizer import BaseTextVectorizer

class TfidfVectorizer(BaseTextVectorizer):

    def __init__(self, max_features: int = 1000, stop_words: str = "english"):
        """
        Initialize the TF-IDF vectorizer.

        Args:
            max_features (int): Maximum number of features to keep.
            stop_words (str): Stop words to remove during vectorization.
        """
        self.vectorizer = SklearnTfidfVectorizer(max_features=max_features, stop_words=stop_words)

    def vectorize(self, documents: list[str]) -> list[str]:
        """
        Vectorize a list of documents using TF-IDF.

        Args:
            documents (List[str]): List of text documents to vectorize.

        Returns:
            scipy.sparse.csr_matrix: Transformed document-term matrix.
        """
        return self.vectorizer.fit_transform(documents)

    def get_feature_names(self):
        """Get feature names (vocabulary) from the vectorizer."""
        return self.vectorizer.get_feature_names_out()
