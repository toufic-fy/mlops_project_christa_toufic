from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Union, Any

class VectorizerFactory:
    """Factory class to create vectorizers based on configuration."""
    vectorizers = {
        "tfidf": TfidfVectorizer,
        "bow": CountVectorizer,
    }
    @staticmethod
    def get_vectorizer(self: Any, vectorizer_type: str, **kwargs) -> Union[TfidfVectorizer, CountVectorizer]:
        """Retrieve the specified vectorizer class."""
        
        vectorizer_cls = self.vectorizers.get(vectorizer_type.lower())
        if not vectorizer_cls:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
        
        return vectorizer_cls(**kwargs)