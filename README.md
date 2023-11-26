# Machine Learning Platform with MLflow

## Project Overview
This Machine Learning Platform is designed as a robust, scalable solution for training and serving machine learning models using Docker containers. The project leverages the open-source MLflow framework, focusing on a Naive Bayes classifier from the scikit-learn library to classify text data. It's specifically tailored for analyzing student feedback 'reflections', classifying them into one of three categories: negative, neutral, or positive.

## Features

- **MLflow Integration**: Utilizes MLflow for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.
- **Dockerized Components**: Separate Docker containers for training and serving the model, ensuring modularity and scalability.
- **Naive Bayes Classifier**: Employs a Naive Bayes classifier from scikit-learn for text classification.
- **Sentiment Analysis**: Classifies student feedback into negative, neutral, and positive categories.
- **Dataset Structure**: The dataset consists of two columns - 'text' (student feedback) and 'labels' (sentiment labels).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Be sure to have installed the following on your machine.
- Docker
- MLflow
- Python 3.x
- scikit-learn

### Getting Started

1. **Clone the Repository**
   ```sh
   git clone [repository URL]
   ```

2. **Define Environment Variables**
   Navigate to the automation folder and define the required environment variables in ./automation/variables.sh

3. **Run End-to-End ML Workflow**
   Start the containers using Docker Compose.
   ```sh
   bash ./automation/execute.sh
   ```

### Finding the best Model Parameters
1. Build the tuning image in ./dockerfile/tune by running:
```sh
docker build -t $TUNE_IMAGE_TAG -f ./dockerfiles/tune/Dockerfile .
```
2. Run the image as a container:
```sh
docker run -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI $TUNE_IMAGE_URI
```
3. Check and Compare MLFlow runs through the Tracking UI

### Training the Model
1. Access the training container's shell.
2. Run the training script.
   ```sh
   python train.py
   ```

### Serving the Model

1. The serving container starts automatically with Docker Compose.
2. Access the model's predictions at `http://localhost:5000`.

## Usage

Provide detailed instructions on how to use the platform, including example requests and responses for the model serving API.

