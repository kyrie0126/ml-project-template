import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.prepare_data import (
    csv_to_df,
    drop_columns,
    split_train_eval_test
)

class TestCsvToDf:
    def test_reads_csv_returns_dataframe(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("A,B,C\n1,2,3\n4,5,6\n")
            path = f.name
        try:
            result = csv_to_df(path)
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (2,3)
            assert list(result.columns) == ["A","B","C"]
        finally:
            os.unlink(path)
    
    def test_preserves_column_types(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("NAME,VALUE\nalpha,1.5\nbeta,2.5\n")
            path = f.name
        try:
            result = csv_to_df(path)
            assert result["NAME"].dtype == object
            assert result["VALUE"].dtype == float
        finally:
            os.unlink(path)
            
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            csv_to_df("nonexistent.csv")


class TestDropColumns:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    
    def test_drop_single_column(self, df):
        result = drop_columns(df, columns=["B"])
        assert list(result.columns) == ["A", "C"]
        
    def test_drop_multiple_columns(self, df):
        result = drop_columns(df, columns=["A","C"])
        assert list(result.columns) == ["B"]
        
    def test_does_not_modify_original(self, df):
        drop_columns(df, columns=["B"])
        assert "B" in df.columns
        
    def test_missing_column_raises(self, df):
        with pytest.raises(KeyError):
            drop_columns(df, columns=["Z"])
            
            
class TestSplitTrainEvalTest:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(123)
        return pd.DataFrame({
            "FEATURE1": np.random.randn(100),
            "FEATURE2": np.random.randn(100),
            "LABEL": np.random.randint(0, 2, 100)
        })
        
    def test_returns_all_expected_keys(self, sample_df):
        result = split_train_eval_test(sample_df, target_col="LABEL")
        assert set(result.keys()) == {"X_train", "X_eval", "X_test", "y_train", "y_eval", "y_test"}
        
    def test_target_column_not_in_features(self, sample_df):
        result = split_train_eval_test(sample_df, target_col="LABEL")
        assert "LABEL" not in result["X_train"].columns
        assert "LABEL" not in result["X_eval"].columns
        assert "LABEL" not in result["X_test"].columns

    def test_all_rows_accounted_for(self, sample_df):
        result = split_train_eval_test(sample_df, target_col="LABEL")
        total = result["X_train"].shape[0] + result["X_eval"].shape[0] + result["X_test"].shape[0]
        assert sample_df.shape[0] == total
        
    def test_all_cols_accounted_for(self, sample_df):
        result = split_train_eval_test(sample_df, target_col="LABEL")
        assert sample_df.shape[1] == result["X_train"].shape[1] + result["y_train"].ndim
        assert sample_df.shape[1] == result["X_eval"].shape[1] + result["y_eval"].ndim
        assert sample_df.shape[1] == result["X_test"].shape[1] + result["y_test"].ndim
        
    def test_xy_index_alignment(self, sample_df):
        result = split_train_eval_test(sample_df, target_col="LABEL")
        assert list(result["X_train"].index) == list(result["y_train"].index)
        assert list(result["X_eval"].index) == list(result["y_eval"].index)
        assert list(result["X_test"].index) == list(result["y_test"].index)