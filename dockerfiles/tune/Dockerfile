# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Copy files to the container
COPY ./requirements.txt ./
COPY ./steps/tune.py ./
COPY ./steps/preprocess.py ./
COPY ./steps/split.py ./
COPY ./exceptions ./exceptions
COPY ./logger ./logger
COPY ./models.py ./
COPY ./params.py ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run train.py when the container launches
CMD ["python3", "tune.py", \
    "--tracking_uri", "$TRACKING_URI", \
    "--experiment_name", "$EXPERIMENT_NAME", \
    "--run_name", "$RUN_NAME", \
    "--model_name", "$MODEL_NAME", \
    "--tokenizer_name", "$TOKENIZER_NAME"]


