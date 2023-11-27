import pandas as pd


def preprocess_data(df):
    """
    Preprocess data for model training
    """
    # Drop rows with missing values
    df = df.dropna()
    # Lowercasing
    df['text'] = df['text'].str.lower()
    # Create label to category mapping
    label2cat = {1: 'positive', 0: 'neutral', -1: 'negative'}
    cat2label = {cat: label for label, cat in label2cat.items()}
    df['label'] = df['label'].map(cat2label)
    # Drop rows with missing values
    df = df[~df['label'].isna()]
    df = df[~df['text'].isna()]

    return df