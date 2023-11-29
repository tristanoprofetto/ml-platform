import pytest
import os
import sys
import pandas as pd
# Set path to parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from steps.ingest import ingest_dataset


@pytest.fixture()
def mock_df():
    return pd.DataFrame({
        'text': ['FOO', 'BAR', 'BAZ', None, None, 'YOLO'],
        'label': ["positive", "negative", None, "neutral", "neutral", "neutral"]
    })


def test_invalid_input_path():
    """
    Ensure error is raised when invalid input path is provided.
    """
    with pytest.raises(FileNotFoundError):
        ingest_dataset('./invalid_path')
