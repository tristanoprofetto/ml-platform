import sys
import os
import logging
import argparse
import datetime
import random
import mlflow
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from exceptions.runs_errors import ParallelRunsError, RunExperimentError
from logger.get_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_tuning(tracking_uri: str,
               experiment_name: str,
               run_name: str = None,
               model_params: dict = None,
               tokenizer_params: dict = None,
               data_params: dict = None,
               df: pd.DataFrame = None
    ):

    """
    Runs trainig as MLFlow experiment

    Args:
        tracking_uri (str): URI of MLFlow tracking server
        experiment_name (str): Name of experiment
        run_name (str): Name of current mlflow run
        model_params (dict): Dictionary of model parameters
        tokenizer_params (dict): Dictionary of tokenizer parameters
        data_params (dict): Dictionary of data parameters
        df (pd.DataFrame): Dataframe containing text and label columns
    
    Returns:
        results (dict): dictionary of results
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            df['text'].tolist(), 
            df['label'].tolist(), 
            test_size=data_params['test_size'], 
            random_state=data_params['random_state']
        )
        # Generate current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Start MLFlow run
        with mlflow.start_run(nested=True, run_name=f'{run_name}:{timestamp}') as run:
            # Vectorize text
            cv = CountVectorizer(
                max_features=tokenizer_params['max_features'], 
                ngram_range=tokenizer_params['ngram_range'], 
                max_df=tokenizer_params['max_df'],
                min_df=tokenizer_params['min_df'],
            )
            X = cv.fit_transform(x_train)
            # Initialize and fit model
            model = MultinomialNB(alpha=model_params['alpha'], fit_prior=model_params['fit_prior'])
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
            # Log model and tokenizer artifacts
            mlflow.sklearn.log_model(
                cv,
                "tokenizer"
            )
            mlflow.sklearn.log_model(
                model, 
                "student-model"
            )
            
        return {"accuracy": accuracy, "model": model, "tokenizer": cv, "model_params": model_params, "tokenizer_params": tokenizer_params, "run_id": run.info.run_id}
    
    except RunExperimentError as e:
        logger.error(f"Error in run_experiment: {e}")


def get_tune_params(size: int = 1):
    """
    Generates random parameters depeneding on the number of workers

    Args:
        size (int): number of workers (CPU cores available to process)

    Returns:
        params_list (dict): dictionary of model, tokenizer, and data parameters
    """
    model_params_lst = []
    tokenizer_params_lst = []
    data_params_lst = []
    for _ in range(size):
        model_params ={
            "alpha": round(random.uniform(0.1, 1 ), 2),
            "fit_prior": random.choice([True, False])
        }
        tokenizer_params = {
            "max_features": 1000,
            "max_df": round(random.uniform(0.7, 0.9 ), 2),
            "ngram_range": (random.randrange(0, 2), random.randrange(3, 5)),
            "min_df": round(random.uniform(0.01, 0.1 ), 2)
        }
        data_params = {
            "test_size": round(random.uniform(0.05, 0.2 ), 2),
            "random_state": random.randrange(0, 100, 2),
        }
        model_params_lst.append(model_params)
        tokenizer_params_lst.append(tokenizer_params)
        data_params_lst.append(data_params)
        
    return {"model_params": model_params_lst, "tokenizer_params": tokenizer_params_lst, "data_params": data_params_lst}


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracking_uri', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--run_name', type=str)

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    import sys
    import os
    import warnings
    warnings.filterwarnings("ignore")
    from concurrent.futures import ProcessPoolExecutor
    from preprocess import preprocess_data
    # Get command-line arguments
    args = get_args()
    # Load dataset
    logger.info('Loading dataset...')
    df = pd.read_csv('./data/feedback.csv')
    # Preprocess dataset
    logger.info('Preprocessing dataset...')
    df = preprocess_data(df)
    # Define model and data parameters
    NUM_WORKERS = os.cpu_count() 
    logger.info(f'Running {NUM_WORKERS} parallel runs')
    params = get_tune_params(NUM_WORKERS)
    try:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run():
            # Running mlflow runs in parallel
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = []
                for i in range(NUM_WORKERS):
                    futures.append(executor.submit(
                        run_tuning,
                        args.tracking_uri,
                        args.experiment_name,
                        args.run_name,
                        params['model_params'][i],
                        params['tokenizer_params'][i],
                        params['data_params'][i],
                        df
                    ))
                results = []
                for future in futures:
                    results.append(future.result())
            logger.info(f'Successfully ran {NUM_WORKERS} parallel runs')
            logger.info('Finding the best run...')
            best_run = max(results, key=lambda x: x['accuracy'])
            logger.info(f'Best run: {best_run}')
            mlflow.set_tracking_uri(args.tracking_uri)
            mlflow.set_experiment(args.experiment_name)
            mlflow.log_params(best_run['model_params'])
            mlflow.log_params(best_run['tokenizer_params'])
            mlflow.sklearn.log_model(best_run['model'], "best-model")
            mlflow.sklearn.log_model(best_run['tokenizer'], "best-tokenizer")
            logger.info(f'Successfully logged best model and tokenizer artifacts with run_id: {best_run["run_id"]}')

    except ParallelRunsError as e:
        logger.error(f"Error in run_experiment: {e}")
        


