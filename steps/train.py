import argparse
import datetime
import mlflow
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_experiment(run_name: str, 
                   model_params: dict = None, 
                   data_params: dict = None, 
                   df: pd.DataFrame = None
    ):

    """
    Runs MLFlow experiment

    Args:
        run_name (str): Name of run
        experiment_name (str): Name of experiment
        df (pd.DataFrame): Dataframe containing text and label columns
    
    Returns:
        None
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment('hi-jokes')
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=data_params['test_size'], 
        random_state=data_params['random_state']
    )
    # Vectorize text
    cv = CountVectorizer(
        max_features=model_params['max_features'], 
        ngram_range=model_params['ngram_range'], 
        max_df=model_params['max_df'],
        min_df=model_params['min_df'],
    )
    X = cv.fit_transform(x_train)
    # Initialize and fit model
    model = MultinomialNB()
    model.fit(X.toarray(), y_train)
    # Generate predicitons on test set
    predictions = model.predict(cv.transform(x_test).toarray())
    # Get test metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    # Initialize mlflow dataset
    train = pd.concat([pd.DataFrame({'text': x_train}), pd.DataFrame({'label': y_train})], axis=1)
    test = pd.concat([pd.DataFrame({'text': x_test}), pd.DataFrame({'label': y_test})], axis=1)
    test['predictions'] = predictions
    train_set = mlflow.data.from_pandas(train)
    test_set = mlflow.data.from_pandas(test, predictions='predictions', targets='label')
    # Generate current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Start MLFlow run
    with mlflow.start_run(run_name=f'{run_name}:{timestamp}') as run:
        # Log hyperparameters
        mlflow.log_params(model_params)
        # Log data parameters
        mlflow.log_params(data_params)
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        # Log datasets
        mlflow.log_input(train_set, "train")
        mlflow.log_input(test_set, "test")
        # Log model and vectorizer artifacts
        mlflow.sklearn.log_model(
            cv,
            "tokenizer",
            #registered_model_name="vectorizer",
        )
        mlflow.sklearn.log_model(
            model, 
            "student-model",
            #registered_model_name="student-model",
        )
        
    #return run.info.run_id, accuracy


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()

    #parser.add_argument('--tracking_uri', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--experiment_name', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import os
    import warnings
    warnings.filterwarnings("ignore")
    # Get command-line arguments
    #args = get_args()
    # Load dataset
    df = pd.read_csv('feedback.csv')
    df = df[~df['text'].isna()]
    # Define model and data parameters
    model_params = {
        "max_features": 1000,
        "ngram_range": (1, 2),
        "max_df": 0.9,
        "min_df": 0.01,
    }
    data_params = {
        "test_size": 0.05,
        "random_state": 42,
    }
    TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    print(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('hi-jokes')
    # Run MLFlow experiment
    run_experiment(
        'whats-up',
        model_params,
        data_params,
        df
    )


