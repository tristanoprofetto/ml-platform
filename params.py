import configparser
import ast
import random


def get_train_params(model_name: str, config: configparser.ConfigParser):
    """
    Get model parameters

    Args:
        model_name (str): name of the specified model

    Returns:
        params (dict): dictionary of model parameters and tokenizer parameters
    """
    if model_name == 'nb':
        model_params = {
            'alpha': config.getfloat(f'model_{model_name}', 'alpha'),
            'fit_prior': config.getboolean(f'model_{model_name}', 'fit_prior')
        }
    elif model_name == 'svc':
        model_params = {
            'C': config.getfloat(f'model_{model_name}', 'C'),
            'kernel': config.get(f'model_{model_name}', 'kernel'),
            'gamma': config.get(f'model_{model_name}', 'gamma')
        }
    elif model_name == 'rf':
        model_params = {
            'n_estimators': config.getint(f'model_{model_name}', 'n_estimators'),
            'max_depth': config.getint(f'model_{model_name}', 'max_depth'),
            'min_samples_split': config.getint(f'model_{model_name}', 'min_samples_split'),
            'min_samples_leaf': config.getint(f'model_{model_name}', 'min_samples_leaf'),
            'max_features': config.get(f'model_{model_name}', 'max_features'),
            'bootstrap': config.getboolean(f'model_{model_name}', 'bootstrap')
        }
    elif model_name == 'lr':
        model_params = {
            'C': config.getfloat(f'model_{model_name}', 'C'),
            'penalty': config.get(f'model_{model_name}', 'penalty'),
            'fit_intercept': config.getboolean(f'model_{model_name}', 'fit_intercept'),
            'max_iter': config.getint(f'model_{model_name}', 'max_iter'),
            'optimizer': config.get(f'model_{model_name}', 'optimizer'),
            'class_weight': ast.literal_eval(config.get(f'model_{model_name}', 'class_weight')),
            'multi_class': config.get(f'model_{model_name}', 'multi_class')
        }
    else:
        raise ValueError(f"Model name: {model_name} not recognized... must be either 'nb', 'svc', 'rf', or 'lr'")
    tokenizer_params = {
        'ngram_range': ast.literal_eval(config.get('tokenizer', 'ngram_range')),
        'max_df': config.getfloat('tokenizer', 'max_df'),
        'min_df': config.getfloat('tokenizer', 'min_df'),
        'max_features': config.getint('tokenizer', 'max_features')
    }
    return {"model_params": model_params, "tokenizer_params": tokenizer_params}


def get_tokenizer_params_tune(size: int):
    """
    Generate random variables for model and tokenizer hyperparameters
    """
    tokenizer_params_lst = []
    for _ in range(size):
        tokenizer_params = {
            "max_features": 1000,
            "max_df": round(random.uniform(0.7, 0.9 ), 2),
            "ngram_range": (random.randrange(0, 2), random.randrange(3, 5)),
            "min_df": round(random.uniform(0.01, 0.1 ), 2)
        }
        tokenizer_params_lst.append(tokenizer_params)
    
    return tokenizer_params_lst


def get_model_params_tune(size: int, model_name: str):
    """
    Generate random variables for model and tokenizer hyperparameters
    """
    model_params_lst = []
    for _ in range(size):
        if model_name == 'nb':
            model_params ={
                "alpha": round(random.uniform(0.1, 1 ), 2),
                "fit_prior": random.choice([True, False])
            }
        elif model_name == 'svc':
            model_params = {
                "C": round(random.uniform(0.1, 1 ), 2),
                "kernel": random.choice(['linear', 'poly', 'rbf', 'sigmoid']),
                "gamma": random.choice(['scale', 'auto']),
                "shrinking": random.choice([True, False]),
                "probability": random.choice([True, False]),
                "class_weight": random.choice([None, 'balanced'])
            }
        elif model_name == 'rf':
            model_params = {
                "n_estimators": random.randrange(10, 100, 10),
                "max_depth": random.randrange(10, 100, 10),
                "min_samples_split": random.randrange(2, 10, 2),
                "min_samples_leaf": random.randrange(1, 10, 2),
                "max_features": random.choice(['sqrt', 'log2', None]),
                "bootstrap": random.choice([True, False])
            }
        elif model_name == 'lr':
            model_params = {
                "penalty": random.choice(['l1', 'l2', 'elasticnet', 'none']),
                "C": round(random.uniform(0.1, 1 ), 2),
                "fit_intercept": random.choice([True, False]),
                "class_weight": random.choice([None, 'balanced']),
                "optimizer": random.choice(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                "max_iter": random.randrange(100, 1000, 100),
                "multi_class": random.choice(['auto', 'ovr', 'multinomial'])
            }
        else:
            raise ValueError(f"Model name: {model_name} not recognized... must be either 'nb', 'svc', 'rf', or 'lr'")
        model_params_lst.append(model_params)
    
    return model_params_lst