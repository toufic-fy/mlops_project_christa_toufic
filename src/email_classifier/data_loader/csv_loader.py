# src/email_classifier/data_loader/csv_loader.py
import pandas as pd
from .base_loader import DataLoader


class CSVLoader(DataLoader):
    """A data loader for loading CSV files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file."""
        try:
            print(f"Loading CSV data from: {file_path}")
            data = pd.read_csv(file_path)
            print("CSV data loaded successfully.")
            return data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
