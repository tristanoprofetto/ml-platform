import os
import mlflow
import requests
from sklearn.feature_extraction.text import CountVectorizer

# #mlflow.set_tracking_uri("http://127.0.0.1:8080")
# model = mlflow.sklearn.load_model("./mlartifacts/486720760588751436/c0aeae4bd6f64952a8a07cdf756fa885/artifacts/student-model")
# vectorizer = mlflow.sklearn.load_model("./mlartifacts/486720760588751436/c0aeae4bd6f64952a8a07cdf756fa885/artifacts/vectorizer")
# #cv = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=0.9)
# vectorized = vectorizer.transform(["I love this class so much lol!", "I hate this class so much lol!"])
# print(model.predict(vectorized.toarray()))

r = requests.post(
    json={"text": ["I love this class so much lol!", "I hate this class so much lol!"]},
    url="http://0.0.0.0:8000/predict/"
)

print(r.json())
