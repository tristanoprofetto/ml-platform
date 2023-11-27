#!/bin/bash
# FILEPATH: /Users/tristano/Desktop/sown/automation/e2e.sh

# Import variables
source ./automation/variables.sh

# Start the MLflow server
#mlflow server --host $MLFLOW_SERVER_HOST --port $MLFLOW_SERVER_PORT --serve-artifacts
if [ "$1" == "train" ]; then
    # Build the train image
    echo "Building the train image..."
    docker build -t $TRAIN_IMAGE_TAG -f ./dockerfiles/train/Dockerfile .
    # Run model training inside the container
    echo "Running model training inside the container..."
    docker run -e TRACKING_URI=$MLFLOW_TRACKING_URI -e EXPERIMENT_NAME=$MLFLOW_EXPERIMENT_NAME -e RUN_NAME=$MLFLOW_RUN_NAME $TRAIN_IMAGE_TAG

elif [ "$1" == "tune" ]; then
    # Build the tune image
    echo "Building the tune image..."
    docker build -t $TUNE_IMAGE_TAG -f ./dockerfiles/tune/Dockerfile .
    # Run model tuning inside the container
    echo "Running model tuning inside the container..."
    docker run -e TRACKING_URI=$MLFLOW_TRACKING_URI -e EXPERIMENT_NAME=$MLFLOW_EXPERIMENT_NAME -e RUN_NAME=$MLFLOW_RUN_NAME $TUNE_IMAGE_TAG

elif [ "$1" == "deploy" ]; then
    # Build the predict image
    echo "Building the serving image..."
    docker build -t $SERVE_IMAGE_TAG -f ./dockerfiles/serve/Dockerfile .
    # Run model prediction inside the container
    echo "Running model prediction inside the container..."
    docker run -p 8000:8000 -e TRACKING_URI=$MLFLOW_TRACKING_URI -e RUN_ID=$SERVE_RUN_ID $SERVE_IMAGE_TAG

else
    echo "Invalid command. Please use 'train', 'tune' or 'deploy'."
fi

