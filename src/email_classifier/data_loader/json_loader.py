# src/email_classifier/data_loader/json_loader.py
import pandas as pd
from .base_loader import DataLoader


class JSONLoader(DataLoader):
    """A data loader for loading JSON files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a JSON file."""
        try:
            print(f"Loading JSON data from: {file_path}")
            data = pd.read_json(file_path)
            print("JSON data loaded successfully.")
            return data
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")
