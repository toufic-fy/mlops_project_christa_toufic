from .base_pipeline import BasePipeline

class InferencePipeline(BasePipeline):

    def run(self, data):
        """Perform vectorization and inference."""
        self.logger.info("Running inference pipeline...")

        vectorized_data = self._vectorizer.transform(data)
        self.logger.debug(f"Data: {vectorized_data.head()}")
        self.logger.info("Data vectorized successfully.")

        self.logger.info("Running Inference.")
        predictions = self._model.predict(vectorized_data)
        self.logger.info("Inference Successfull")
        self.logger.debug(f"Predictions: {predictions.head()}")
        return predictions
