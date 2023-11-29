import sys
import os
import logging
import ast
import argparse
import datetime
import pandas as pd
import numpy as np
import configparser
import mlflow

from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from exceptions.runs_errors import RunExperimentError
from logger.get_logger import setup_logging
# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def run_training(tracking_uri: str,
                 experiment_name: str,
                 run_name: str = None,
                 model_params: dict = None,
                 tokenizer_params: dict = None, 
                 data_params: dict = None, 
                 df: pd.DataFrame = None
    ):
    """
    Runs model training as an MLFlow experiment

    Args:
        tracking_uri (str): URI of MLFlow tracking server
        experiment_name (str): Name of experiment
        run_name (str): Name of run
        model_params (dict): Dictionary of model parameters
        tokenizer_params (dict): Dictionary of tokenizer parameters
        data_params (dict): Dictionary of data parameters
        df (pd.DataFrame): Dataframe containing text and label columns
    
    Returns:
        results (dict): dictionary of training results
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
        # Evaluate model
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
                registered_model_name="tokenizer",
            )
            mlflow.sklearn.log_model(
                model, 
                "student-model",
                registered_model_name="student-model",
            )
            
        return {"run_id": run.info.run_id, "model": model, "tokenizer": cv, "model_params": model_params, "tokenizer_params": tokenizer_params, "data_params": data_params}

    except RunExperimentError as e:
        logger.error(e)


def get_train_params(config: configparser.ConfigParser):
    """
    Get model, tokenizer, and data parameters from config file
    """
    model_params = {
        "alpha": config.getfloat("model", "alpha"),
        "fit_prior": config.getboolean("model", "fit_prior")
    }
    tokenizer_params = {
        "max_features": config.getint("tokenizer", "max_features"),
        "ngram_range": ast.literal_eval(config.get("tokenizer", "ngram_range")),
        "max_df": config.getfloat("tokenizer", "max_df"),
        "min_df": config.getfloat("tokenizer", "min_df"),
    }
    data_params = {
        "test_size": config.getfloat("dataset", "test_size"),
        "random_state": config.getint("dataset", "random_state"),
    }
    return { "model_params": model_params, "tokenizer_params": tokenizer_params, "data_params": data_params}


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000')
    parser.add_argument('--experiment_name', type=str, default='test-experiment')
    parser.add_argument('--run_name', type=str, default='test-run')
    parser.add_argument('--input_data_path', type=str, default='./data/feedback.csv')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    import os
    import configparser
    import warnings
    warnings.filterwarnings("ignore")
    from preprocess import preprocess_data
    # Get command-line arguments
    args = get_args()
    # Load dataset
    logger.info('Loading dataset...')
    df = pd.read_csv(args.input_data_path)
    # Preprocess dataset
    logger.info('Preprocessing dataset...')
    df = preprocess_data(df)
    # Get model, tokenizer, and data parameters from config file
    config = configparser.ConfigParser()
    config.read('conf.ini')
    params = get_train_params(config)
    # Run MLFlow experiment
    logger.info('Starting model training with MLflow...')
    result = run_training(
        args.tracking_uri,
        args.experiment_name,
        args.run_name,
        params['model_params'],
        params['tokenizer_params'],
        params['data_params'],
        df
    )
    logger.info(f'Successfully ran mlflow training experiment with run_id: {result["run_id"]}')


