# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Copy files to the container
COPY ./requirements.txt ./
COPY ./data ./data
COPY ./steps/train.py ./
COPY ./steps/preprocess.py ./
COPY ./logger ./logger
COPY ./exceptions ./exceptions
COPY ./conf.ini ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run train.py when the container launches
CMD ["python3", "train.py", "--tracking_uri", "$TRACKING_URI", "--experiment_name", "$EXPERIMENT_NAME", "--run_name", "$RUN_NAME"]


