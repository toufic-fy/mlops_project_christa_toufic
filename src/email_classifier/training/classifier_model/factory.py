from .logistic_classifier import LogisticTextClassifier
from .sgd_classifier import SGDTextClassifier
from .base_classifier import BaseClassifier

class ClassifierFactory:
    """Factory class to retrieve classifiers dynamically."""

    _classifiers = {
            "sgd": SGDTextClassifier,
            "logistic": LogisticTextClassifier,
    }

    @staticmethod
    def get_classifier(classifier_type: str) -> BaseClassifier:


        classifier_cls = ClassifierFactory._classifiers.get(classifier_type.lower())
        if not classifier_cls:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

        return classifier_cls()
