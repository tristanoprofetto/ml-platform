import requests


def get_predictions(texts: list, url: str = 'http://0.0.0.0:8000/predict/'):
    """
    Function to make inference requests to model serving container
    """
    r = requests.post(
        json={"text": texts},
        url='http://0.0.0.0:8000/predict/'
    )

    print(r.json())
    return r.json()

if __name__ == "__main__":
    # Get the predictions
    url = 'http://0.0.0.0:8000/predict/'
    predictions = get_predictions(
        texts=["I love this class so much lol!", "I hate this class so much lol!"],
        url=url
    )
    print(predictions)
