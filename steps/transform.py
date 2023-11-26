import mlflow
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def transform_data(df):
    """
    Transform data for modeling
    
    Args:
        df (pandas.DataFrame): DataFrame containing raw data
    
    Returns:
        mlflow.run.info.RunInfo: MLFlow run information
    """
    label2cat = {1: 'positive', 0: 'neutral', -1: 'negative'}
    cat2label = {cat: label for label, cat in label2cat.items()}

    df['vectorized_text'] = df['text'].apply(lambda x: ' '.join(x))


if __name__ == "__main__":
    df = pd.read_csv('feedback.csv')
    cv = CountVectorizer(
        max_features=1000, 
        ngram_range=(1, 2), 
        max_df=0.9,
        min_df=0.01
    )
    vectorized_text = cv.fit_transform(df['text'].tolist())
    print(vectorized_text)