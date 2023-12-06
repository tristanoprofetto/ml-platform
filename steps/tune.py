import sys
import os
import logging
import argparse
import datetime
import random
import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from params import get_model_params_tune, get_tokenizer_params_tune
from models import get_model, get_tokenizer
from exceptions.runs_errors import ParallelRunsError, RunExperimentError
from logger.get_logger import setup_logging
# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def run_tuning(tracking_uri: str,
               experiment_name: str,
               run_name: str = None,
               model_name: str = 'nb',
               tokenizer_name: str = 'count',
               model_params: dict = None,
               tokenizer_params: dict = None,
               df_train: pd.DataFrame = None,
               df_test: pd.DataFrame = None,
               logger: logging.Logger = None
    ):
    """
    Runs a model tuning experiment as an MLFlow experiment

    Args:
        tracking_uri (str): URI of MLFlow tracking server
        experiment_name (str): Name of experiment
        run_name (str): Name of current mlflow run
        model_name (str): Name of the model to select and train
        tokenizer_name (str): Name of the vectorizer to select and train
        model_params (dict): Dictionary of model parameters
        tokenizer_params (dict): Dictionary of tokenizer parameters
        df_train (pd.DataFrame): training set containing text and label columns
        df_test (pd.DataFrame): testing set containing text and label columns
        logger (logging.Logger): logger instance
    
    Returns:
        results (dict): dictionary of results
    """
    try:
        logger.info(f'Loading {model_name} model instance...')
        model = get_model(model_name=model_name, params=model_params)
        logger.info(f'Loading {tokenizer_name} tokenizer instance...')
        tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, params=tokenizer_params)
        # Generate current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Start MLFlow run
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(nested=True, run_name=f'{run_name}-{model_name}:{timestamp}', description='Parallel parameter tuning jobs') as run:
            # Vectorize text
            logger.info(f'Fitting training data on {tokenizer_name} tokenizer...')
            X = tokenizer.fit_transform(df_train['text'])
            # Initialize and fit model
            logger.info(f'Fitting training data on {model_name} model...')
            model.fit(X.toarray(), df_train['label'])
            # Generate predictions on test set
            logger.info('Generating predictions on test set...')
            predictions = model.predict(tokenizer.transform(df_test['text']).toarray())
            # Get test metrics
            logger.info('Evaluating model...')
            accuracy = accuracy_score(df_test['label'], predictions)
            precision = precision_score(df_test['label'], predictions, average='weighted')
            recall = recall_score(df_test['label'], predictions, average='weighted')
            f1 = f1_score(df_test['label'], predictions, average='weighted')
            logger.info('Logging metrics, model, and tokenizer artifacts...')
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
                tokenizer,
                "tokenizer"
            )
            mlflow.sklearn.log_model(
                model, 
                "student-model"
            )
            logger.info(f'Successfully completed a run with accuracy: {accuracy}')

        return {"accuracy": accuracy, "model": model, "tokenizer": tokenizer, "model_params": model_params, "tokenizer_params": tokenizer_params, "run_id": run.info.run_id}
    
    except RunExperimentError as e:
        logger.error(f"Error in run_experiment: {e}")


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracking_uri', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model_name', type=str, default='nb')
    parser.add_argument('--tokenizer_name', type=str, default='count')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    import sys
    import os
    import configparser
    import warnings
    warnings.filterwarnings("ignore")
    from concurrent.futures import ProcessPoolExecutor
    from preprocess import preprocess_data
    from split import split_dataset
    # Get command-line arguments
    args = get_args()
    config = configparser.ConfigParser()
    config.read('conf.ini')
    # Load dataset
    logger.info('Loading dataset...')
    df = pd.read_csv('./data/feedback.csv')
    # Preprocess dataset
    logger.info('Preprocessing dataset...')
    df = preprocess_data(df)
    # Splitting dataset into train and test sets
    df_train, df_test = split_dataset(df=df,
                                      test_size=config.getfloat('dataset', 'test_size'),
                                      random_state=config.getint('dataset', 'random_state')
    )
    # Define model and data parameters
    NUM_WORKERS = os.cpu_count() - 4
    logger.info(f'Running {NUM_WORKERS} parallel runs')
    model_params = get_model_params_tune(NUM_WORKERS, args.model_name)
    tokenizer_params = get_tokenizer_params_tune(NUM_WORKERS)
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
                        args.model_name,
                        args.tokenizer_name,
                        model_params[i],
                        tokenizer_params[i],
                        df_train,
                        df_test,
                        logger
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
            mlflow.sklearn.log_model(best_run['model'], "best-model")
            mlflow.sklearn.log_model(best_run['tokenizer'], "best-tokenizer")
            logger.info(f'Successfully logged best model and tokenizer artifacts with run_id: {best_run["run_id"]}')

    except ParallelRunsError as e:
        logger.error(f"Error in run_experiment: {e}")
        


