import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.create_model_pipeline import (
    # preprocessing fns
    standard_scaler,
    onehot_encode,
    
    # model fns
    random_forest_classifier,
    random_forest_regressor,
    xgboost_classifier,
    xgboost_regressor,
    
    # init fns
    init_pipeline,
    
    # build fns
    build_pipeline,
    
    # fit fns
    fit_pipeline,
)

class TestPreprocessing:
    def test_standard_scaler_returns_tuple(self):
        result = standard_scaler(["col1", "col2"])
        assert result[0] == "standard_scaler"
        assert isinstance(result[1], StandardScaler)
        assert result[2] == ["col1","col2"]
        
    def test_onehot_encode_returns_tuple(self):
        result = onehot_encode(["col1"])
        assert result[0] == "onehot_encode"
        assert isinstance(result[1], OneHotEncoder)
        assert result[1].handle_unknown == "ignore"


class TestModel:    
    # xgb classifier
    def test_xgboost_classifier_default(self):
        name, model = xgboost_classifier()
        assert name == "model"
        assert isinstance(model, XGBClassifier)
        
    def test_xgboost_classifier_with_params(self):
        name, model = xgboost_classifier({"max_depth": 5, "n_estimators": 100})
        assert model.max_depth == 5
        assert model.n_estimators == 100
    
    # xgb regressor
    def test_xgboost_regressor_default(self):
        name, model = xgboost_regressor()
        assert name == "model"
        assert isinstance(model, XGBRegressor)
        
    def test_xgboost_regressor_with_params(self):
        name, model = xgboost_regressor({"max_depth": 5, "n_estimators": 100})
        assert model.max_depth == 5
        assert model.n_estimators == 100
    
    # rf classifier
    def test_random_forest_classifier_default(self):
        name, model = random_forest_classifier()
        assert name == "model"
        assert isinstance(model, RandomForestClassifier)
        
    def test_random_forest_classifier_with_params(self):
        name, model = random_forest_classifier({"max_depth": 5, "n_estimators": 100})
        assert model.max_depth == 5
        assert model.n_estimators == 100
    
    # rf regressor
    def test_random_forest_regressor_default(self):
        name, model = random_forest_regressor()
        assert name == "model"
        assert isinstance(model, RandomForestRegressor)
        
    def test_random_forest_regressor_with_params(self):
        name, model = random_forest_regressor({"max_depth": 5, "n_estimators": 100})
        assert model.max_depth == 5
        assert model.n_estimators == 100


class TestInitPipeline:
    def test_returns_pipeline(self):
        pipe = init_pipeline()
        assert isinstance(pipe, Pipeline)
    
    def test_starts_empty(self):
        pipe = init_pipeline()
        assert len(pipe.steps) == 0
        

class TestBuildPipeline:
    def test_with_preprocessing(self):
        pipe = init_pipeline()
        preprocessing = [standard_scaler(["num_col"])]
        model = xgboost_classifier({"n_estimators": 10})
        result = build_pipeline(pipe=pipe, model=model, preprocessing=preprocessing)
        assert len(result.steps) == 2
        assert result.steps[0][0] == "preprocessing"
        assert result.steps[1][0] == "model"
        
    def test_without_preprocessing(self):
        pipe = init_pipeline()
        model = xgboost_classifier({"n_estimators": 10})
        result = build_pipeline(pipe=pipe, model=model)
        assert len(result.steps) == 1
        assert result.steps[0][0] == "model"
    
    def test_single_tuple_preprocessing(self):
        pipe = init_pipeline()
        preprocessing = standard_scaler(["num_col"])
        model = xgboost_classifier({"n_estimators": 10})
        result = build_pipeline(pipe=pipe, model=model, preprocessing=preprocessing)
        assert len(result.steps) == 2
        
    def test_multiple_preprocessing(self):
        pipe = init_pipeline()
        preprocessing = [
            standard_scaler(["num_col"]),
            onehot_encode(["cat_col"])
        ]
        model = xgboost_classifier({"n_estimators": 10})
        result = build_pipeline(pipe=pipe, model=model, preprocessing=preprocessing)
        assert len(result.steps) == 2


class TestFitPipeline:
    @pytest.fixture
    def numeric_data(self):
        np.random.seed(123)
        X = pd.DataFrame({
            "num_1": np.random.randn(50),
            "num_2": np.random.randint(1, 11, 50)
        })
        y = pd.Series(np.random.randint(0, 2, 50))
        return X, y
    
    @pytest.fixture
    def mixed_data(self):
        np.random.seed(123)
        X = pd.DataFrame({
            "cat_1": np.random.choice(["a", "b", "c"], 50),
            "num_1": np.random.randn(50)
        })
        y = pd.Series(np.random.randint(0, 2, 50))
        return X, y
    
    def test_train_pred_numeric_data(self, numeric_data):
        X, y = numeric_data
        pipe = init_pipeline()
        preprocessing = [standard_scaler(["num_2"])]
        model = xgboost_classifier({"max_depth": 5, "n_estimators": 10})
        built_pipe = build_pipeline(pipe, model=model, preprocessing=preprocessing)
        fit_pipe = fit_pipeline(built_pipe, X, y)
        preds = fit_pipe.predict(X)
        assert len(preds) == 50
        assert set(preds).issubset({0,1})
        
    def test_train_pred_mixed_data(self, mixed_data):
        X, y = mixed_data
        pipe = init_pipeline()
        preprocessing = [
            onehot_encode(["cat_1"]),
            standard_scaler(["num_1"])
        ]
        model = random_forest_classifier({"max_depth": 5, "n_estimators": 10})
        built_pipe = build_pipeline(pipe, model=model, preprocessing=preprocessing)
        fit_pipe = fit_pipeline(built_pipe, X, y)
        preds = fit_pipe.predict(X)
        assert len(preds) == 50
        assert set(preds).issubset({0,1})