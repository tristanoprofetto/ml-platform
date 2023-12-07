# Use an official Python runtime as a parent image
FROM python:3.9-slim

ARG MODEL_NAME
ARG TOKENIZER_NAME
# Copy files to the container
COPY ./requirements.txt ./
COPY ./steps/train.py ./
COPY ./steps/preprocess.py ./
COPY ./steps/split.py ./
COPY ./logger ./logger
COPY ./exceptions ./exceptions
COPY ./conf.ini ./
COPY ./models.py ./
COPY ./params.py ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

ENV MODEL_NAME=$MODEL_NAME
ENV TOKENIZER_NAME=$TOKENIZER_NAME

# Run train.py when the container launches
CMD ["python3", "train.py", \
    "--tracking_uri", "$TRACKING_URI", \
    "--experiment_name", "$EXPERIMENT_NAME", \
    "--run_name", "$RUN_NAME", \
    "--model_name", "$MODEL_NAME", \
    "--tokenizer_name", "$TOKENIZER_NAME"]


