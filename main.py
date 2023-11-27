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
    # Tune Parser
    tune_parser = subs.add_parser("tune")
    tune_parser.add_argument(
        "--tracking_uri",
        type=str,
        default="http://localhost:5000"
    )
    tune_parser.add_argument(
        "--experiment_name",
        type=str,
        default="tune-student-experiment"
    )
    tune_parser.add_argument(
        "--run_name",
        type=str,
        default="tune-student-run"
    )
    # Deploy Parser
    deploy_parser = subs.add_parser("deploy")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from  mlflow.tracking import MlflowClient
    from exceptions import runs_errors
    import warnings
    print(os.getcwd())
    warnings.filterwarnings("ignore")
    from concurrent.futures import ThreadPoolExecutor
    # Get command-line arguments
    args = get_args()
    if args.command == "train" or args.command == "tune":
        import pandas as pd
        from steps.preprocess import preprocess_data
        # Load dataset
        df = pd.read_csv('./data/feedback.csv')
        # Preprocess dataset
        df = preprocess_data(df)
        if args.command == "train":
            import configparser
            from steps.train import run_training, get_train_params
            # Get model, tokenizer, and data parameters from config file
            config = configparser.ConfigParser()
            config.read('conf.ini')
            params = get_train_params(config)
            run_training(
                tracking_uri=args.tracking_uri,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                model_params=params['model_params'],
                tokenizer_params=params['tokenizer_params'],
                data_params=params['data_params'],
                df=df
            )
        elif args.command == "tune":
            from concurrent.futures import ProcessPoolExecutor
            from steps.tune import run_tuning, get_tune_params
            # Get model, tokenizer, and data parameters from config file
            NUM_WORKERS = os.cpu_count() - 8
            params = get_tune_params(NUM_WORKERS)
            with mlflow.start_run():
                with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = []
                    for i in range(NUM_WORKERS):
                        try:
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
                        except Exception as e:
                            print(e)
                    results = []
                    for future in futures:
                        results.append(future.result())
                print(results)
                print('****************************')
                best_run = max(results, key=lambda x: x['accuracy'])



