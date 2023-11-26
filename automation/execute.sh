#!/bin/bash
# FILEPATH: /Users/tristano/Desktop/sown/automation/e2e.sh

# Import variables
source ./variables.sh

# Install dependencies
pip install -r requirements.txt

# Start the MLflow server
mlflow server --host $MLFLOW_SERVER_HOST --port $MLFLOW_SERVER_PORT --serve-artifacts

docker build -t mlflow-test .

