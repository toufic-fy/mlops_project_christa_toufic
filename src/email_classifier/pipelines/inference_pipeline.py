from .base_pipeline import BasePipeline

class InferencePipeline(BasePipeline):

    def run(self, data: list[str], include_confidence: bool = False):
        """
        Perform vectorization, inference, and optionally compute confidence scores.

        Args:
            data (list[str]): List of email bodies.
            include_confidence (bool): Whether to compute confidence scores.

        Returns:
            dict: A dictionary containing predictions and optionally confidence scores.
        """
        self.logger.info("Running inference pipeline...")

        # Vectorize input data
        vectorized_data = self._vectorizer.transform(data)
        self.logger.info("Data vectorized successfully.")

        # Perform inference
        predictions = self._model.predict(vectorized_data)
        self.logger.info("Inference successful.")

        # Compute confidence scores if requested
        response = {"predictions": predictions}
        if include_confidence:
            confidences = self._model.predict_proba(vectorized_data).max(axis=1)
            response["confidences"] = confidences
            self.logger.debug(f"Confidences: {confidences}")

        self.logger.debug(f"Predictions: {predictions}")
        return response
