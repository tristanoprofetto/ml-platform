import os
import argparse
import datetime
import mlflow
import pandas as pd

from concurrent.futures import ProcessPoolExecutor

from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    subs.required = True
    # Train Parser
    train_parser = subs.add_parser("train")
    train_parser.add_argument(
        "--tracking_uri",
        type=str,
        default="127.0.0.1:8080",
        help="URI of the MLFlow tracking server",
    )
    train_parser.add_argument(
        "--experiment_name",
        type=str,
        default="feedback-sentiment",
        help="Name of the experiment to log to",
    )
    train_parser.add_argument(
        "--run_name",
        type=str,
        default="feedback-run",
        help="Name of the run to log to",
    )
    # train_parser.add_argument(
    #     "--run_id",
    #     type=str,
    #     default=None,
    #     help="ID of the run to log to",
    # )
    # Deploy Parser
    deploy_parser = subs.add_parser("deploy")

    args = parser.parse_args()
    return args


def run_training(
        model_params: dict = None, 
        data_params: dict = None, 
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
    mlflow.set_experiment('feedback-sentiment')
    df = pd.read_csv('./data/feedback.csv')
    #run = client.create_run(experiment.experiment_id, run_name=run_name)
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
    # Get timestamp for current run
    tstamp = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S.%f')
    run_name = 'feedback'
    # Start MLFlow run
    with mlflow.start_run(run_name=f'{run_name}-{tstamp}') as run:
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
            "vectorizer",
            registered_model_name="vectorizer",
        )
        mlflow.sklearn.log_model(
            model, 
            "student-model",
            registered_model_name="student-model",
        )
        
    #return run.info.run_id, accuracy


if __name__ == "__main__":
    import multiprocessing as mp
    import pandas as pd
    from steps.params import get_params
    from  mlflow.tracking import MlflowClient

    from concurrent.futures import ThreadPoolExecutor

    #NUM_WORKERS = os.cpu_count()
    
    args = get_args()
    if args.command == "train":
        client = MlflowClient(tracking_uri=args.tracking_uri)
        experiment = mlflow.get_experiment_by_name(name=args.experiment_name)
        df = pd.read_csv('./data/feedback.csv')
        NUM_WORKERS = 4
        model_params, data_params = get_params(size=NUM_WORKERS)
        mlflow.set_tracking_uri(args.tracking_uri)
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            executor.map(
                run_training,
                model_params,
                data_params
            )




