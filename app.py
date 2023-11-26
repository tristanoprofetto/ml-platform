import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Create a FastAPI app instance
app = FastAPI()
# Get model and vectorizer paths from environment variables
MODEL_PATH = os.environ.get("MODEL_PATH")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH")
# Load the trained MultinomialNB classifier and preprocessing transformers
model = mlflow.sklearn.load_model(model_uri=MODEL_PATH)
vectorizer = mlflow.sklearn.load_model(model_uri=VECTORIZER_PATH)
# Create label to category mapping
label2cat = {1: 'positive', 0: 'neutral', -1: 'negative'}

# Define input and output models using Pydantic
class InputData(BaseModel):
    text: List[str]

class OutputData(BaseModel):
    predictions: List[str]

# Define a route for making predictions
@app.post("/predict/", response_model=OutputData)
async def predict(input_data: InputData):
    print(input_data)
    # Preprocess the input data using the loaded vectorizer
    X = vectorizer.transform(input_data.text)

    # Make predictions using the loaded classifier
    predictions = model.predict(X)
    output = [label2cat[pred] for pred in predictions]
    # Return the predictions
    return {"predictions": output}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    # Use uvicorn to run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)