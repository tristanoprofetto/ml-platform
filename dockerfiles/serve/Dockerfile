# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./dockerfiles/serve/requirements.txt ./
COPY ./app.py ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port that the FastAPI application will run on
EXPOSE 5000

# Command to run the FastAPI application
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
