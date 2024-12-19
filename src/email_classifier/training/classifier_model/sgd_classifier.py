from sklearn.linear_model import SGDClassifier
from .base_classifier import BaseClassifier
from typing import Dict, Any

class SGDTextClassifier(BaseClassifier):
    """SGD Classifier implementation."""

    def get_classifier(self, **kwargs) -> SGDClassifier:
        """Return an SGDClassifier instance."""
        return SGDClassifier(**kwargs)

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "loss": ["hinge", "log"],
            "alpha": [0.0001, 0.001],
            "max_iter": [1000, 2000],
            "tol": [1e-3, 1e-4]
        }
