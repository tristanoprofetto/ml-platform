import os
import sys
import argparse
import sklearn

import mlflow


#src_uri = f"runs:/{run.info.run_id}/sklearn-model"

def register_model(client: mlflow.MlflowClient, run_id: str):
    """
    Register model with MLFlow

    Args:

    """
    pass


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
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    mlflow.set_tracking_uri('http://10.0.0.117:8080')
    #model = mlflow.sklearn.load_model(model_uri='runs:/dc13f7f6f4324295845f77c3865a544d/student-model')
    #tokenizer = mlflow.sklearn.load_model(model_uri='runs:/dc13f7f6f4324295845f77c3865a544d/tokenizer')
    model = mlflow.sklearn.load_model(model_uri='runs:/dc13f7f6f4324295845f77c3865a544d/student-model')
    tokenizer = mlflow.sklearn.load_model(model_uri='runs:/dc13f7f6f4324295845f77c3865a544d/tokenizer')
    print(model.predict(tokenizer.transform(['I hate this class'])))
