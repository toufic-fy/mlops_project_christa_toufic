from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ...data_pipeline.vectorizer.base_text_vectorizer import BaseTextVectorizer as Vectorizer
import pandas as pd

class Trainer:
    """Handles training and evaluation of classifiers."""

    def __init__(self, classifier, vectorizer: Vectorizer, hyperparams=None):
        """
        Initialize the trainer.

        Args:
            classifier (BaseClassifier): An instance of a classifier.
            vectorizer: A vectorizer instance (e.g., TfidfVectorizer).
            hyperparams (dict, optional -> overrides classifier's get_hyperparameters): Hyperparameter grid for GridSearchCV
        """
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.hyperparams = hyperparams or classifier.get_hyperparameters()

    def train(self, X_train, y_train):
        """
        Train the model using GridSearchCV.

        Args:
            X_train (list[str]): Training data (list of email bodies).
            y_train (list): Training labels.

        Returns:
            Trained model.
        """
        # Create a pipeline with the vectorizer and classifier
        pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier.get_classifier())
        ])

        # Perform Grid Search for hyperparameter optimization
        grid_search = GridSearchCV(
            pipeline,
            param_grid={"classifier__" + key: value for key, value in self.hyperparams.items()},
            cv=5,
            scoring="accuracy",
            verbose=1
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_)

        return grid_search.best_estimator_

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate the trained model.

        Args:
            model: Trained model pipeline.
            X_test (list[str]): Test data.
            y_test (list): Test labels.

        Returns:
            dict: Evaluation metrics (accuracy, classification report, confusion matrix).
        """
        # Predict on the test data
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(pd.DataFrame(report).transpose())
        print("Confusion Matrix:")
        print(confusion)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion
        }
