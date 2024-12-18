from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from .base_text_vectorizer import BaseTextVectorizer
from typing import List

class TfidfVectorizer(BaseTextVectorizer):
    def vectorize(self, documents: List[str]) -> List[str]:
        return self.vectorizer.fit_transform(documents)