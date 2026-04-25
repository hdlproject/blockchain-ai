import pandas as pd
import pytest
from blockchain_ai.ingest import load_and_clean


def test_load_and_clean_drops_nulls(tmp_path):
    raw_csv = tmp_path / "input.csv"
    raw_csv.write_text("feature1,feature2,target\n1,2,3\n4,,6\n7,8,9\n")
    out_path = tmp_path / "processed.csv"

    load_and_clean(str(raw_csv), str(out_path))

    result = pd.read_csv(out_path)
    assert len(result) == 2
    assert list(result.columns) == ["feature1", "feature2", "target"]


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.csv"), str(tmp_path / "out.csv"))
