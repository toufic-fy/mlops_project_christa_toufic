# src/email_classifier/data_loader/base_loader.py
from abc import ABC, abstractmethod
from typing import List
from ..preprocessor.base_preprocessor import BasePreprocessor
import pandas as pd

class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from the specified file path.

        Args:
            file_path (str): The path to the file to load data from.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        pass

    def load_and_preprocess_data(self, file_path: str,  preprocessors: list[BasePreprocessor] = None) -> pd.DataFrame:
            """
            Load and preprocess the email dataset.

            Args:
                file_path (str): Path to the CSV file.
                preprocessor (BasePreprocessor, optional): Preprocessor instance for preprocessing the data. 
                                                        Defaults to None.

            Returns:
                pd.DataFrame: Preprocessed dataset if a preprocessor is provided, otherwise raw dataset.
            """
            # Load the raw dataset
            df = self.load_data(file_path)
            
            if preprocessors:
                for preprocessor in preprocessors:
                    df = preprocessor.preprocess(df)
            
            return df