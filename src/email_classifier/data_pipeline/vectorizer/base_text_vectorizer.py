from abc import ABC, abstractmethod
from typing import List


class BaseTextVectorizer(ABC):
    @abstractmethod
    def vectorize(self, documents: List) -> List:
        """Vectorize a list of documents."""
        pass

    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary) from the vectorizer if it supports it.
        Raises an exception in case the vectorizer doesn't support it.

        Returns:
            List[str]: A list of feature names.
        """
        if hasattr(self, "vectorizer") and hasattr(self.vectorizer, "get_feature_names_out"):
            return self.vectorizer.get_feature_names_out().tolist()
        raise NotImplementedError("This vectorizer does not support feature extraction.")

    def fit(self, X, y=None):
        """
        Fit the vectorizer to the data.

        Args:
            X (list of str): Input text data.
            y (ignored): Not used, present for API consistency.
        """
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        """
        Transform the input data to TF-IDF feature vectors.

        Args:
            X (list of str): Input text data.

        Returns:
            sparse matrix: Transformed feature vectors.
        """
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Args:
            X (list of str): Input text data.
            y (ignored): Not used, present for API consistency.

        Returns:
            sparse matrix: Transformed feature vectors.
        """
        return self.vectorizer.fit_transform(X)
