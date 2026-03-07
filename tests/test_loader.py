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
