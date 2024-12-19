from sklearn.linear_model import LogisticRegression
from .base_classifier import BaseClassifier
from typing import Dict, Any

class LogisticTextClassifier(BaseClassifier):
    """SGD Classifier implementation."""
    
    def get_classifier(self, **kwargs) -> LogisticRegression:
        """Return an SGDClassifier instance."""
        return LogisticRegression(**kwargs)

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "C": [0.1, 1, 10],
            "max_iter": [100, 200]
        }