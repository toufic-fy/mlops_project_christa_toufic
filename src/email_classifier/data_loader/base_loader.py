# src/email_classifier/data_loader/base_loader.py
from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self, file_path: str) -> List[str]:
        """
        Loads data from the specified file path.

        Args:
            file_path (str): The path to the file to load data from.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        pass
