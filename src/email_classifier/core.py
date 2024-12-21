from __future__ import annotations

import pandas as pd
from loguru import logger

from email_classifier.config import VectorizationConfig, ClassificationConfig
from email_classifier.data_handling.vectorizer.base_text_vectorizer import BaseTextVectorizer
from email_classifier.data_handling.vectorizer.factory import VectorizerFactory
from email_classifier.training.classifier_model.base_classifier import BaseClassifier
from email_classifier.training.classifier_model.factory import ClassifierFactory


def load_pipeline(
    vectorization_config: VectorizationConfig, model_config: ClassificationConfig
) -> InferencePipeline:
    data_transformer = VectorizerFactory.get_vectorizer(vectorization_config.type, **vectorization_config.params)
    model = ClassifierFactory.get_classifier(model_config.type)
    return InferencePipeline(data_transformer, model)


class InferencePipeline:
    _data_vecotrizer: BaseTextVectorizer
    _model: BaseClassifier

    def __init__(self, data_vecotrizer: BaseTextVectorizer, model: BaseClassifier):
        self._data_vecotrizer = data_vecotrizer
        self._model = model

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs the inference data pipeline on the input data.

        Args:
            data (pd.DataFrame): The input data to process.

        Returns:
            pd.DataFrame: The processed data.
        """

        try:
            logger.info("Pipeline execution started.")

            logger.info("Applying Data vectorization.")
            vectorized_data = self._data_vecotrizer.vectorize(data)
            logger.debug(f"Data: {vectorized_data.head()}")
            logger.info("Data vectorized successfully.")

            logger.info("Running Inference.")
            predictions = self._model.predict(vectorized_data)
            logger.debug(f"Predictions: {predictions.head()}")
            logger.info("Model prediction completed successfully.")

            logger.info("Pipeline execution completed.")
            return predictions

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            return
