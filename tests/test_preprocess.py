import pytest
import os
import sys
import pandas as pd
# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from steps.preprocess import preprocess_data


@pytest.fixture()
def mock_df():
    return pd.DataFrame({
        'text': ['FOO', 'BAR', 'BAZ', None, None, 'YOLO'],
        'label': ["positive", "negative", None, "neutral", "neutral", "neutral"]
    })


def test_drop_missing_vals(mock_df):
    """
    Ensure rows with missing values are correctly dropped.
    """
    df = preprocess_data(mock_df)
    assert df['text'].str.islower().all()
    assert df['label'].isin([-1, 0, 1]).all()
    assert df.isna().sum().sum() == 0

