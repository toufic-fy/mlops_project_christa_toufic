from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseClassifier(ABC):
    """Abstract base class for classifiers."""
    
    @abstractmethod
    def get_classifier(self, **kwargs) -> Any:
        """Return an instance of the classifier."""
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return a dictionary of hyperparameters for Grid Search."""
        pass
