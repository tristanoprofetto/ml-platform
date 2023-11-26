import os
import json
import pandas as pd


def ingest_dataset(input_path: str, output_path: str = None):
    """
    Ingest dataset from folder
    """
    # Initialize lists to store data
    texts = []
    labels = []
    # Iterate through each file in folder
    for file in os.listdir(input_path):
        # Open file
        with open(os.path.join(input_path, file)) as f:
            # Load data from file
            data = json.load(f)
            # Iterate through each text/label pair
            for text, label in data:
                # Append text and label to respective lists
                texts.append(' '.join(text))
                labels.append(label)
    # Create dataframe from lists
    df = pd.DataFrame({'text': texts, 'label': labels})
    # Return dataframe
    #df.to_csv(output_path, index=False)
    return df