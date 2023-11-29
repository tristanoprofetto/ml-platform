import pytest
import sys
import os
import ast
import configparser
# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from steps.train import get_train_params


def test_config_functionality(mock_config):
    params = get_train_params(mock_config)
    assert params["model_params"]["alpha"] == 0.5
    assert params["model_params"]["fit_prior"] == True
    assert params["tokenizer_params"]["max_features"] == 1000
    assert params["tokenizer_params"]["max_df"] == 0.9
    assert params["tokenizer_params"]["ngram_range"] == (1,1)
    assert params["tokenizer_params"]["min_df"] == 0.01
    assert params["data_params"]["test_size"] == 0.2
    assert params["data_params"]["random_state"] == 42


def test_missing_keys(mock_config):
    with pytest.raises(configparser.NoOptionError):
        mock_config.remove_option("model", "alpha")
        _ = get_train_params(mock_config)

    with pytest.raises(configparser.NoOptionError):
        mock_config.remove_option("model", "fit_prior")
        _ = get_train_params(mock_config)

    with pytest.raises(configparser.NoOptionError):
        mock_config.remove_option("tokenizer", "max_features")
        _ = get_train_params(mock_config)
    
    with pytest.raises(configparser.NoOptionError):
        mock_config.remove_option("tokenizer", "max_df")
        _ = get_train_params(mock_config)






    
