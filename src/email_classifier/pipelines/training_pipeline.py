from .base_pipeline import BasePipeline
from sklearn.model_selection import train_test_split
from ..training.trainer.trainer import Trainer
from utils.mlf_utils import save_model_with_vectorizer
import mlflow
import mlflow.sklearn
import pandas as pd
import os

class TrainingPipeline(BasePipeline):

    def fetch_best_accuracy(self, experiment_name: str) -> float:
        """
        Fetch the best accuracy from past runs in MLflow for the given experiment.

        Args:
            experiment_name (str): The name of the MLflow experiment.

        Returns:
            float: The best accuracy across all runs, or 0 if no runs exist.
        """
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return 0

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.accuracy DESC"],
            max_results=1,
        )

        if runs:
            return runs[0].data.metrics.get("accuracy", 0)
        return 0


    def run(self, data: list[str], labels: list, experiment_name: str, log_best_accuracy: bool = False):
        """Perform training, including vectorization and model fitting."""
        self.logger.info("Running training pipeline...")

        if not experiment_name:
            raise ValueError("Experiment name must be specified in the configuration.")

        # TODO: next major - make test_size, stratify and radom_seed configurable
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        trainer = Trainer(self._model, self._vectorizer)

        with mlflow.start_run():
            # Log model hyperparameters
            mlflow.log_param("vectorizer", self._vectorizer.__class__.__name__)
            mlflow.log_param("model", self._model.get_classifier().__class__.__name__)
            mlflow.log_params(self._model.get_hyperparameters())

            # Train the model
            self.logger.info("Training the model...")
            trained_model = trainer.train(X_train.tolist(), y_train.tolist())

            # Evaluate the model
            self.logger.info("Evaluating the model...")
            eval = trainer.evaluate(trained_model, X_test.tolist(), y_test.tolist())

            # Log evaluation metrics
            mlflow.log_metric("accuracy", eval["accuracy"])
            mlflow.log_metrics({
                "precision": eval["classification_report"]["weighted avg"]["precision"],
                "recall": eval["classification_report"]["weighted avg"]["recall"],
                "f1-score": eval["classification_report"]["weighted avg"]["f1-score"]
            })

            # Log confusion matrix as an artifact
            confusion_matrix_df = pd.DataFrame(eval["confusion_matrix"])
            confusion_matrix_path = "confusion_matrix.csv"
            confusion_matrix_df.to_csv(confusion_matrix_path, index=False)
            mlflow.log_artifact(confusion_matrix_path)

            # Debugging
            if not os.path.exists(confusion_matrix_path):
                print(f"File {confusion_matrix_path} was not created.")
            else:
                print(f"File {confusion_matrix_path} exists at {os.path.abspath("confusion_matrix.csv")} and is ready to log.")
                mlflow.log_artifact(confusion_matrix_path)

            # Log the model after evaluation
            if log_best_accuracy:
                best_accuracy = self.fetch_best_accuracy(experiment_name)
                print("training debug: ", eval["accuracy"], best_accuracy)
                if eval["accuracy"] > best_accuracy:
                    self.logger.info("New best model found, logging to MLflow.")
                    save_model_with_vectorizer(trained_model, self._vectorizer)
                else:
                    self.logger.info("Model did not surpass best accuracy, not logging.")
            else:
                self.logger.info("Logging the model...")
                save_model_with_vectorizer(trained_model, self._vectorizer)

            self.logger.info("Training and evaluation logged to MLflow.")

        return eval

