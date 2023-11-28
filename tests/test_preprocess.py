import pytest
import os
import sys
import pandas as pd
# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from steps.preprocess import preprocess_data


@pytest.fixture
def df():
    return pd.DataFrame({
        'text': ['foo', 'bar', 'baz'],
        'label': [1, 0, -1]
    })

@pytest.fixture
def df_with_nan():
    return pd.DataFrame({
        'text': ['foo', 'bar', None, 'baz'],
        'label': [1, 0, -1, None]
    })


def test_drop_missing_vals(df_with_nan):
    """
    Ensure rows with missing values are correctly dropped.
    """
    preprocesed_df = preprocess_data(df_with_nan)
    assert preprocesed_df.isna().sum().sum() == 0


def test_lowercasing():
    pass