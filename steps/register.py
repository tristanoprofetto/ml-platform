import os
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
    client = mlflow.tracking.MlflowClient()
