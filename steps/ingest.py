import os
import json
import pandas as pd


def ingest_dataset(input_path: str):
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
    return df

if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    df = ingest_dataset('./data')
    df.to_csv('./data/feedback.csv', index=False)