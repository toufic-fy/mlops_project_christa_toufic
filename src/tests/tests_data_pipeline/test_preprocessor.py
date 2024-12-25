import pytest
import pandas as pd
from email_classifier.data_handling.preprocessor.base_preprocessor import BasePreprocessor
from email_classifier.data_handling.preprocessor.email_preprocessor import EmailPreprocessor

def test_base_preprocessor_instantiation():
    """
    Test that BasePreprocessor cannot be instantiated directly.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class BasePreprocessor without an implementation for abstract method 'preprocess'"):
        BasePreprocessor()

def test_email_preprocessor_drops_missing_values():
    """
    Test that EmailPreprocessor drops rows with missing 'label' or 'body'.
    """
    preprocessor = EmailPreprocessor()
    input_data = pd.DataFrame({
        "label": [0, 1, None, 1],
        "body": ["email 1", "email 2", "email 3", None]
    })
    processed_data = preprocessor.preprocess(input_data)

    # Assert missing rows are dropped
    assert len(processed_data) == 2, "Expected 2 rows after dropping rows with missing values."
    assert processed_data.isnull().sum().sum() == 0, "No missing values should remain."

def test_email_preprocessor_balances_dataset():
    """
    Test that EmailPreprocessor balances the dataset by sampling.
    """
    preprocessor = EmailPreprocessor()
    input_data = pd.DataFrame({
        "label": [1, 1, 1, 1, 0, 0],
        "body": ["email 1", "email 2", "email 3", "email 4", "email 5", "email 6"]
    })
    processed_data = preprocessor.preprocess(input_data)

    # Assert balanced dataset
    phishing_count = (processed_data["label"] == 1).sum()
    safe_count = (processed_data["label"] == 0).sum()
    assert phishing_count == safe_count, "Phishing and safe emails should be balanced."
    assert len(processed_data) == 4, "The resulting dataset should contain the same number of phishing and safe emails."

def test_email_preprocessor_handles_empty_dataframe():
    """
    Test that EmailPreprocessor handles empty input gracefully.
    """
    preprocessor = EmailPreprocessor()
    empty_data = pd.DataFrame(columns=["label", "body"])
    processed_data = preprocessor.preprocess(empty_data)

    # Assert the output is also empty
    assert processed_data.empty, "Expected an empty DataFrame for empty input."

def test_email_preprocessor_all_missing_rows():
    """
    Test that EmailPreprocessor handles DataFrames where all rows are missing 'label' or 'body'.
    """
    preprocessor = EmailPreprocessor()
    input_data = pd.DataFrame({
        "label": [None, None],
        "body": [None, None]
    })
    processed_data = preprocessor.preprocess(input_data)

    # Assert the output is empty
    assert processed_data.empty, "Expected an empty DataFrame after dropping all missing rows."

def test_email_preprocessor_handles_unbalanced_data():
    """
    Test that EmailPreprocessor correctly handles unbalanced datasets.
    """
    preprocessor = EmailPreprocessor()
    input_data = pd.DataFrame({
        "label": [1, 1, 1, 0],
        "body": ["email 1", "email 2", "email 3", "email 4"]
    })
    processed_data = preprocessor.preprocess(input_data)

    # Assert balanced dataset
    phishing_count = (processed_data["label"] == 1).sum()
    safe_count = (processed_data["label"] == 0).sum()
    assert phishing_count == safe_count, "Phishing and safe emails should be balanced."

