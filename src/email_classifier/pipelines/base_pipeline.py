from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger

from ..training.classifier_model.base_classifier import BaseClassifier
from ..data_handling.vectorizer.base_text_vectorizer import BaseTextVectorizer

class BasePipeline(ABC):
    def __init__(self, model: BaseClassifier, vectorizer: BaseTextVectorizer):
        self._model = model
        self._vectorizer = vectorizer
        self.logger = logger
        self.pd = pd

    def load_data(self, file_type: str, file_path: str):
        """Load data based on the provided file path."""
        # Implement shared data loading logic here
        print(f"Loading data from {file_path}")
        # Replace with actual data loading logic
        return []

    @abstractmethod
    def run(self, *args, **kwargs):
        """Abstract method to define the main pipeline logic."""
        pass
