from .tfidf_vectorizer import TfidfVectorizer
from .bow_vectorizer import BOWVectorizer
from typing import Union

class VectorizerFactory:
    """Factory class to create vectorizers based on configuration."""
    _vectorizers = {
        "tfidf": TfidfVectorizer,
        "bow": BOWVectorizer,
    }
    @staticmethod
    def get_vectorizer(vectorizer_type: str, **kwargs) -> Union[TfidfVectorizer, BOWVectorizer]:
        """Retrieve the specified vectorizer class."""
        
        vectorizer_cls = VectorizerFactory._vectorizers.get(vectorizer_type.lower())
        if not vectorizer_cls:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
        
        return vectorizer_cls(**kwargs)