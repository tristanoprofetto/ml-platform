import os
import sys


def get_model(model_name: str, params: dict = None):
    """
    Gets scikit-learn model by corresponding name

    Args:
        model_name (str): name of the model to retrieve
    
    Returns:
        model
    """

    if model_name == 'nb':
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(
            alpha=params['alpha'],
            fit_prior=params['fit_prior']
        )
    elif model_name == 'svc':
        from sklearn.svm import SVC
        model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=params['gamma'],
            shrinking=params['shrinking'],
            probability=params['probability'],
            class_weight=params['class_weight'],
        )
    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            bootstrap=params['bootstrap']
        )
    elif model_name == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            penalty=params['penalty'],
            C=params['C'],
            fit_intercept=params['fit_intercept'],
            class_weight=params['class_weight'],
            solver=params['optimizer'],
            max_iter=params['max_iter'],
            multi_class=params['multi_class']
        )
    else:
        raise ValueError(f"Model name: {model_name} not recognized... must be either 'nb', 'svc', 'rf', or 'lr'")

    return model


def get_tokenizer(tokenizer_name: str, params: dict = None):
    """
    Gets scikit-learn vectorizer by corresponding name

    Args:
        tokenizer_name (str): name of the vectorizer to retrieve
    
    Returns:
        tokenizer
    """

    if tokenizer_name == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        tokenizer = TfidfVectorizer(
            ngram_range=params['ngram_range'],
            max_df=params['max_df'],
            min_df=params['min_df'],
            max_features=params['max_features']
        )
    elif tokenizer_name == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        tokenizer = CountVectorizer(
            ngram_range=params['ngram_range'],
            max_df=params['max_df'],
            min_df=params['min_df'],
            max_features=params['max_features']
        )
    else:
        raise ValueError(f"Tokenizer name: {tokenizer_name} not recognized... must be either 'tfidf' or 'count'")

    return tokenizer


