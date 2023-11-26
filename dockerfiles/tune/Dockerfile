# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Copy files to the container
COPY ./requirements.txt ./
COPY ./data/feedback.csv ./
COPY ./train.py ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run train.py when the container launches
CMD ["python3", "train.py"]


