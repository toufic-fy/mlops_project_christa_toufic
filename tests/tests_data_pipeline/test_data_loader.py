import pytest
import pandas as pd
from unittest.mock import patch
from email_classifier.data_pipeline.data_loader.csv_loader import CSVLoader
from email_classifier.data_pipeline.preprocessor.email_preprocessor import EmailPreprocessor

@pytest.fixture
def sample_csv_file(tmp_path):
    """
    Creates a temporary CSV file for testing.
    """
    data = pd.DataFrame({
        "body": ["Email body 1", "Email body 2", "Email body 3"],
        "label": [0, -1, 1]
    })
    file_path = tmp_path / "emails.csv"
    data.to_csv(file_path, index=False)
    return file_path

def test_load_data(sample_csv_file):
    """
    Test the `load_data` method of `CSVDataLoader`.
    """
    # Arrange
    loader = CSVLoader()

    # Act
    df = loader.load_data(sample_csv_file)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["body", "label"]
    assert df["body"].iloc[0] == "Email body 1"
    assert df["label"].iloc[1] == -1

def test_load_empty_file(tmp_path):
    """
    Test that `load_data` raises an error when the CSV file is empty.
    """
    # Arrange: Create an empty CSV file
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    loader = CSVLoader()

    # Act & Assert
    with pytest.raises(ValueError, match="Error loading CSV file: No columns to parse from file"):
        loader.load_data(empty_file)

def test_invalid_file_path():
    """
    Test that `load_data` raises an error when the file path is invalid.
    """
    loader = CSVLoader()

    # Act & Assert
    with pytest.raises(ValueError, match="Error loading CSV file: .*No such file or directory.*"):
        loader.load_data("non_existent_file.csv")



def test_sampling_logic():
    """
    Test the sampling logic in the preprocessor directly.
    """
    data = pd.DataFrame({
        "body": ["Email body 1", "Email body 2", "Email body 3"],
        "label": [0, 0, 1]
    })

    preprocessor = EmailPreprocessor()

    # Act: Apply sampling logic
    sampled_data = preprocessor.preprocess(data)

    # Assert
    assert len(sampled_data) == 2, "Expected 2 rows after sampling."
    assert sampled_data["label"].value_counts()[0] == 1, "Expected balanced labels."

def test_load_and_preprocess_data(sample_csv_file):
    """
    Test the `load_and_preprocess_data` method with sampling mocked.
    """
    # Arrange
    loader = CSVLoader()
    preprocessor = EmailPreprocessor()

    # Mock the sampling step in the preprocessor
    with patch.object(preprocessor, "preprocess", return_value=pd.DataFrame({
        "body": ["Email body 1", "Email body 2"],
        "label": [0, 1]
    })) as mock_preprocess:
        # Act
        preprocessed_data = loader.load_and_preprocess_data(file_path=sample_csv_file, preprocessors=[preprocessor])
        # Assert
        mock_preprocess.assert_called_once()
        assert isinstance(preprocessed_data, pd.DataFrame)
        assert len(preprocessed_data) == 2
        assert list(preprocessed_data.columns) == ["body", "label"]
        assert preprocessed_data["body"].iloc[0] == "Email body 1"
