import pandas as pd
from .base_preprocessor import BasePreprocessor

class EmailPreprocessor(BasePreprocessor):
    """Preprocessor for email data."""

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the email dataset.

        Steps:
        - Drop records with missing values.
        - Balance the dataset by sampling phishing emails to match the number of safe emails.

        Args:
            df (pd.DataFrame): Input DataFrame with at least "Label" and "Body" columns.

        Returns:
            pd.DataFrame: Preprocessed and balanced DataFrame.
        """
        # Drop missing values
        df = df.dropna(subset=["Label", "Body"])
        
        # Separate phishing and safe emails
        phishing_emails = df[df["Label"] == "Phishing"]
        safe_emails = df[df["Label"] == "Safe"]
        
        # Balance the dataset by sampling phishing emails
        if phishing_emails.shape[0] > safe_emails.shape[0]:
            phishing_emails = phishing_emails.sample(safe_emails.shape[0], random_state=42)
        elif safe_emails.shape[0] > phishing_emails.shape[0]:
            safe_emails = safe_emails.sample(phishing_emails.shape[0], random_state=42)
        
        # Combine the balanced datasets
        balanced_df = pd.concat([phishing_emails, safe_emails], axis=0).reset_index(drop=True)
        
        return balanced_df
