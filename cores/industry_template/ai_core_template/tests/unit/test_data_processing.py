import pandas as pd
import pytest
from aicore.pipelines.data_processing.nodes import clean_data, validate_data

def test_clean_data():
    df = pd.DataFrame({
        "a": [1, 2, None],
        "b": ["x", "x", "y"]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned.isnull().sum().sum() == 0

def test_validate_data():
    df = pd.DataFrame({"a": [1]})
    validated = validate_data(df, {})
    assert not validated.empty

    with pytest.raises(ValueError, match="Data is empty"):
        validate_data(pd.DataFrame(), {})
