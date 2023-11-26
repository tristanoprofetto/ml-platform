# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Define build arguments
ARG MODEL_PATH
ARG VECTORIZER_PATH

# Copy the current directory contents into the container at /app
COPY ./dockerfiles/requirements.txt ./
COPY . /app
COPY ./mlartifacts/486720760588751436/${RUN_ID}/artifacts/student-model ./model
COPY ./mlartifacts/486720760588751436/${RUN_ID}/artifacts/vectorizer ./vectorizer

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Define environment variables (if needed)
ENV MODEL_PATH=$MODEL_PATH
ENV VECTORIZER_PATH=$VECTORIZER_PATH

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
