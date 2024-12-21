# src/email_classifier/data_loader/csv_loader.py
import pandas as pd
from .base_loader import DataLoader

class CSVLoader(DataLoader):
    """A data loader for loading CSV files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
            Load and preprocess the email dataset.

            Args:
                file_path (str): Path to the CSV file.
                preprocessor (BasePreprocessor, optional): Preprocessor instance for preprocessing the data.
                                                        Defaults to None.

            Returns:
                pd.DataFrame: Preprocessed dataset if a preprocessor is provided, otherwise raw dataset.
        """
        try:
            data = pd.read_csv(file_path)
            print("CSV data loaded successfully.")
            return data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
