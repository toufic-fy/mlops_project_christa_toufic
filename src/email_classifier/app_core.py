from __future__ import annotations

from email_classifier.config import VectorizationConfig, ClassificationConfig
from email_classifier.data_handling.vectorizer.factory import VectorizerFactory
from email_classifier.training.classifier_model.factory import ClassifierFactory
from email_classifier.pipelines.base_pipeline import BasePipeline
from email_classifier.pipelines.factory import PipelineFactory


def load_pipeline(pipeline_type: str,
    vectorization_config: VectorizationConfig, model_config: ClassificationConfig
) -> BasePipeline:
    vectorizer = VectorizerFactory.get_vectorizer(vectorization_config.type, **vectorization_config.params)
    model = ClassifierFactory.get_classifier(model_config.type)
    return PipelineFactory.get_pipeline(pipeline_type, model, vectorizer)


# class InferencePipeline:
#     _data_vecotrizer: BaseTextVectorizer
#     _model: BaseClassifier

#     def __init__(self, data_vecotrizer: BaseTextVectorizer, model: BaseClassifier):
#         self._data_vecotrizer = data_vecotrizer
#         self._model = model

#     def run(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Runs the inference data pipeline on the input data.

#         Args:
#             data (pd.DataFrame): The input data to process.

#         Returns:
#             pd.DataFrame: The processed data.
#         """

#         try:
#             logger.info("Pipeline execution started.")

#             logger.info("Applying Data vectorization.")
#             vectorized_data = self._data_vecotrizer.vectorize(data)
#             logger.debug(f"Data: {vectorized_data.head()}")
#             logger.info("Data vectorized successfully.")

#             logger.info("Running Inference.")
#             predictions = self._model.predict(vectorized_data)
#             logger.debug(f"Predictions: {predictions.head()}")
#             logger.info("Model prediction completed successfully.")

#             logger.info("Pipeline execution completed.")
#             return predictions

#         except Exception as e:
#             logger.error(f"Failed in Pipeline Execution: {e}")
#             return
