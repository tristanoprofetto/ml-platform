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

# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from params import get_train_params
from models import get_model, get_tokenizer
from exceptions.runs_errors import RunExperimentError
from logger.get_logger import setup_logging
# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def run_training(tracking_uri: str,
                 experiment_name: str,
                 run_name: str = None,
                 model_name: str = 'nb',
                 tokenizer_name: str = 'count',
                 params: dict = None,
                 df_train: pd.DataFrame = None,
                 df_test: pd.DataFrame = None,
                 logger: logging.Logger = None
    ):
    """
    Runs model training as an MLFlow experiment

    Args:
        tracking_uri (str): URI of MLFlow tracking server
        experiment_name (str): Name of experiment
        run_name (str): Name of the current run
        model_name (str): Name of the model to select and train
        tokenizer_name (str): Name of the vectorizer to select and train
        params (dict): Dictionary of model and tokenizer parameters
        df_train (pd.DataFrame): train dataset containing text and label columns
        df_test (pd.DataFrame): test dataset containing text and label columns
        logger (logging.Logger): logger instance
    
    Returns:
        results (dict): dictionary of training results
    """
    try:
        # Load model and tokenizer instances
        logger.info(f'Loading {model_name} model instance..')
        model = get_model(model_name=model_name, params=params['model_params'])
        logger.info(f'Loading {tokenizer_name} tokenizer instance..')
        tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, params=params['tokenizer_params'])
        # Set MLFlow tracking URI and experiment name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        # Generate current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Start MLFlow run
        with mlflow.start_run(run_name=f'{run_name}-{model_name}:{timestamp}') as run:
            # Fit tokenizer on training data
            logger.info(f'Fitting {tokenizer_name} tokenizer on training data...')
            X = tokenizer.fit_transform(df_train['text'])
            # Initialize and fit model
            logger.info(f'Fitting {model_name} model on training data...')
            model.fit(X.toarray(), df_train['label'])
            # Generate predicitons on test set
            predictions = model.predict(tokenizer.transform(df_test['text']).toarray())
            # Evaluate model
            logger.info(f'Evaluating {model_name} model...')
            df_test['predictions'] = predictions
            mlflow.evaluate(model_type='classifier', data=df_test, predictions='predictions', targets='label')
            # Initialize mlflow dataset
            train_set = mlflow.data.from_pandas(df_train)
            test_set = mlflow.data.from_pandas(df_test)
            # Log model and tokenizer parameters
            model_params, tokenizer_params = model.get_params(), tokenizer.get_params()
            mlflow.log_params(model_params)
            mlflow.log_params(tokenizer_params)
            # Log datasets
            mlflow.log_input(dataset=train_set, context="train")
            mlflow.log_input(dataset=test_set, context="test")
            # Log model and vectorizer artifacts
            logger.info(f'Logging {model_name} model and {tokenizer_name} tokenizer artifacts...')
            mlflow.sklearn.log_model(
                tokenizer,
                "tokenizer",
                #registered_model_name=tokenizer_name
            )
            mlflow.sklearn.log_model(
                model, 
                "student-model",
                #registered_model_name=model_name,
            )
            
        return {"run_id": run.info.run_id, "model": model, "tokenizer": tokenizer, "model_params": params['model_params'], "tokenizer_params": params['tokenizer_params']}

    except RunExperimentError as e:
        logger.error(e)


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000')
    parser.add_argument('--experiment_name', type=str, default='dev-experiment-')
    parser.add_argument('--run_name', type=str, default='test-run')
    parser.add_argument('--model_name', type=str, default='nb')
    parser.add_argument('--tokenizer_name', type=str, default='count')
    parser.add_argument('--input_data_path', type=str, default='./data/feedback.csv')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    import os
    import configparser
    import warnings
    warnings.filterwarnings("ignore")
    from preprocess import preprocess_data
    from split import split_dataset
    # Get command-line arguments and read config file
    args = get_args()
    config = configparser.ConfigParser()
    config.read('conf.ini')
    # Load dataset
    logger.info('Loading dataset...')
    df = pd.read_csv(args.input_data_path)
    # Preprocess dataset
    logger.info('Preprocessing dataset...')
    df = preprocess_data(df)
    # Split data into train and test sets
    logger.info('Splitting dataset into train and test sets...')
    df_train, df_test = split_dataset(df=df, 
                                      test_size=config.getfloat('dataset', 'test_size'), 
                                      random_state=config.getint('dataset', 'random_state')
                                      )
    # Load data, model, and tokenizer env vars for training job
    env_vars = [os.environ.get('MODEL_NAME'), os.environ.get('TOKENIZER_NAME')]
    if None not in env_vars:
        model_name = env_vars[0]
        tokenizer_name = env_vars[1]
    else:
        model_name = args.model_name
        tokenizer_name = args.tokenizer_name
    params = get_train_params(model_name, config)
    # Run MLFlow experiment
    logger.info('Starting model training with MLflow...')
    result = run_training(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        params=params,
        df_train=df_train,
        df_test=df_test,
        logger=logger
    )
    logger.info(f'Successfully ran mlflow training experiment with run_id: {result["run_id"]}')