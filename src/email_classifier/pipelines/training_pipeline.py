from .base_pipeline import BasePipeline
from sklearn.model_selection import train_test_split
from ..training.trainer.trainer import Trainer

class TrainingPipeline(BasePipeline):

    def run(self, data: list[str], labels: list):
        """Perform training, including vectorization and model fitting."""
        self.logger.info("Running training pipeline...")
        # TODO: next major - make test_size, stratify and radom_seed configurable
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        trainer = Trainer(self._model, self._vectorizer)
        trained_model = trainer.train(X_train.tolist(), y_train.tolist())

        self.logger.info("Training done, evaluating...")

        eval = trainer.evaluate(trained_model, X_test.tolist(), y_test.tolist())
        self.logger.debug(f"Accuracy: {eval["accuracy"]}")
        self.logger.debug(f"Classification Report: {self.pd.DataFrame(eval["classification_report"]).transpose()}")
        self.logger.debug(f"Confusion Matrix: {eval["confusion_matrix"]}")

