import pandas as pd
import pytest

from src.schema_analyzer import analyze_schema


def test_analyze_schema():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "product": ["Laptop", "Mouse", "Keyboard"],
            "quantity": [2, 5, 3],
            "is_returned": [False, True, False],
        }
    )

    schema = analyze_schema(df)

    assert schema["order_id"]["role"] == "identifier"
    assert schema["product"]["role"] == "categorical"
    assert schema["quantity"]["role"] == "numeric"
    assert schema["is_returned"]["role"] == "boolean"


def test_analyze_schema_invalid_input():
    with pytest.raises(TypeError):
        analyze_schema("not a dataframe")