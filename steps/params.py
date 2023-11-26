import random


def get_params(size: int = 1):
    """
    Generates random parameters depeneding on the number of workers

    Args:
        size (int): number of workers (CPU cores available to process)

    Returns:
        params_list (list): list of tuples of model and data parameters
    """
    model_params_lst = []
    data_params_lst = []
    params_list = []
    for _ in range(size):
        model_params = {
            "max_features": 1000,
            "max_df": round(random.uniform(0.7, 0.9 ), 2),
            "ngram_range": (random.randrange(0, 2), random.randrange(3, 5)),
            "min_df": round(random.uniform(0.01, 0.1 ), 2)
        }
        data_params = {
            "test_size": round(random.uniform(0.05, 0.2 ), 2),
            "random_state": random.randrange(0, 100, 2),
        }
        model_params_lst.append(model_params)
        data_params_lst.append(data_params)
        
    return model_params_lst, data_params_lst


    
