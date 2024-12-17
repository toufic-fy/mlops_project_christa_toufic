# src/email_classifier/data_loader/factory.py
from .base_loader import DataLoader
from .csv_loader import CSVLoader
from .json_loader import JSONLoader


class DataLoaderFactory:
    """Factory class to create data loader instances based on the file type."""

    _loaders = {
        "csv": CSVLoader,
        "json": JSONLoader,
    }

    @staticmethod
    def get_data_loader(file_type: str) -> DataLoader:
        """
        Returns an instance of a data loader based on the provided file type.

        Args:
            file_type (str): The type of file to load (e.g., "csv", "json").

        Returns:
            DataLoader: An instance of the requested data loader.

        Raises:
            ValueError: If the file type is unsupported.
        """
        file_type = file_type.lower()
        loader_class = DataLoaderFactory._loaders.get(file_type)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_type}. Supported types: {list(DataLoaderFactory._loaders.keys())}")
        return loader_class()
