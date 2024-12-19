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