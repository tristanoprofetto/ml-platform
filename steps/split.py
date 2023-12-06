import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits dataset into train and test sets

    Args:
        df (pd.DataFrame): Dataframe containing text and label columns
        test_size (float): Size of test set
        random_state (int): Random state for reproducibility
    
    Returns:
        df_train (pd.DataFrame): Training set
        df_test (pd.DataFrame): Test set
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test