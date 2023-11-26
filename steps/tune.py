import argparse
import datetime
import mlflow
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_experiment(model_params: dict = None):

    """
    Runs MLFlow experiment

    Args:
        run_name (str): Name of run
        experiment_name (str): Name of experiment
        df (pd.DataFrame): Dataframe containing text and label columns
    
    Returns:
        None
    """
    mlflow.set_tracking_uri('http://10.0.0.117:8080')
    mlflow.set_experiment('hi-jokes')
    df = pd.read_csv('./data/feedback.csv')
    df = df[~df['text'].isna()]
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.1, 
        random_state=42
    )
    # Generate current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Start MLFlow run
    with mlflow.start_run(nested=True, run_name=f'run-test:{timestamp}', log_system_metrics=True) as run:
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
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        # Log hyperparameters
        mlflow.log_params(model_params)
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        # Log model and vectorizer artifacts
        mlflow.sklearn.log_model(
            cv,
            "tokenizer"
        )
        mlflow.sklearn.log_model(
            model, 
            "student-model"
        )
        
    return {"accuracy": accuracy, "model": model, "tokenizer": cv, "params": model_params, "run_id": run.info.run_id}


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
    from steps.params import get_params
    warnings.filterwarnings("ignore")
    from concurrent.futures import ProcessPoolExecutor
    # Get command-line arguments
    #args = get_args()
    # Load dataset
    # df = pd.read_csv('./data/feedback.csv')
    # df = df[~df['text'].isna()]
    # Define model and data parameters
    NUM_WORKERS = os.cpu_count() - 8
    model_params_lst, data_params_lst = get_params(NUM_WORKERS)
    with mlflow.start_run():
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i in range(NUM_WORKERS):
                try:
                    futures.append(executor.submit(run_experiment, model_params_lst[i]))
                except Exception as e:
                    raise e
            results = []
            for future in futures:
                results.append(future.result())
        print(results)
        print('****************************')
        best_run = max(results, key=lambda x: x['accuracy'])
        #best_run = max(results, key=lambda x: x['accuracy'])
        # mlflow.set_tracking_uri('http://10.0.0.117:8080')
        # mlflow.set_experiment('hi-jokes')
        # mlflow.log_params(best_run['params'])
        # mlflow.sklearn.log_model(best_run['model'], "best-model")
        # mlflow.sklearn.log_model(best_run['tokenizer'], "best-vectorizer")
        #mlflow.log_params(max(results, key=lambda x: x['accuracy']))
        # mlflow.log_params(best)
        # mlflow.log_metric("accuracy", best_run['loss'])
        # mlflow.sklearn.log_model(best_run['model'], "best-model")
        # mlflow.sklearn.log_model(best_run['vectorizer'], "best-vectorizer")


