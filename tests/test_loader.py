import numpy as np
import pandas as pd
import pytest
from autoprep.loader import DataLoader

EXPECTED_COLUMNS = ["id", "date", "age", "salary", "city", "gender", "product_category", "score", "notes"]

def test_load_csv():
    loader = DataLoader()
    df = loader.load_data("data/sample.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS

def test_file_not_found():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_data("data/non_existent_file.csv")


def test_unsupported_format(tmp_path):
    fake_file = tmp_path / "test.txt"
    fake_file.write_text("This is a fake file")

    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.load_data(str(fake_file))


def test_save_data_csv(tmp_path):
    loader = DataLoader()
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    out_file = tmp_path / "saved.csv"
    loader.save_data(df, str(out_file))

    loaded = pd.read_csv(out_file)
    assert loaded.shape == (2, 2)
    assert list(loaded.columns) == ["a", "b"]


def test_save_data_parquet(tmp_path):
    loader = DataLoader()
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    out_file = tmp_path / "saved.parquet"
    loader.save_data(df, str(out_file))

    loaded = pd.read_parquet(out_file)
    assert loaded.shape == (2, 2)
    assert list(loaded.columns) == ["a", "b"]


def test_save_data_unsupported_extension(tmp_path):
    loader = DataLoader()
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        loader.save_data(df, str(tmp_path / "saved.json"))


def test_type_inference_with_quoted_numeric_values(tmp_path):
    from autoprep.cleaner import DataCleaner
    
    df = pd.DataFrame({
        "mixed_numeric": ["10", "563", "20", "22000", '"475"', "23", "15"]
    })
    
    cleaner = DataCleaner()
    result = cleaner.fit_transform(df)
    
    assert pd.api.types.is_numeric_dtype(result["mixed_numeric"])
    assert result["mixed_numeric"].iloc[4] == 475.0


def test_no_int64_dtype_with_decimal_values():
    from autoprep.cleaner import DataCleaner
    
    df = pd.DataFrame({
        "mixed_decimals": [10, 563, 20, 22000, 152.5, 23, 15]
    })
    
    cleaner = DataCleaner()
    result = cleaner.fit_transform(df)
    
    # Should be float64, not Int64
    assert result["mixed_decimals"].dtype in [np.float64, 'float64', 'Float64']
    assert result["mixed_decimals"].iloc[4] == 152.5


def test_loader_converts_int64_to_float64():
    import tempfile
    loader = DataLoader()
    
    # Create a CSV with missing numeric values (pandas infers as Int64)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n")
        f.write("10,100.5\n")
        f.write(",200.3\n")  # Missing value
        f.write("563,300.0\n")
        f.write('"475",152.5\n')  # Quoted value
        temp_path = f.name
    
    try:
        df = loader.load_data(temp_path)
        
        # Both columns should be float64, not Int64
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            assert "Int" not in dtype_str, f"{col} has dtype {dtype_str} instead of float64"
            assert df[col].dtype in [np.float64, 'float64', 'Float64']
        
        # Values should be preserved
        assert df["col2"].iloc[3] == 152.5
    finally:
        import os
        os.unlink(temp_path)
