from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

import numpy as np
import pandas as pd




#############################################
############### preprocessing ###############
#############################################
def standard_scaler(columns: list) -> tuple:
    return ("standard_scaler", StandardScaler(), columns)


def onehot_encode(columns: list) -> tuple:
    return ("onehot_encode", OneHotEncoder(handle_unknown="ignore"), columns)




#############################################
################### model ###################
#############################################
def random_forest_classifier(params: dict=None) -> tuple:
    return ("model", RandomForestClassifier(**(params or {})))


def random_forest_regressor(params: dict=None) -> tuple:
    return ("model", RandomForestRegressor(**(params or {})))


def xgboost_classifier(params: dict=None) -> tuple:
    return ("model", XGBClassifier(**(params or {})))


def xgboost_regressor(params: dict=None) -> tuple:
    return ("model", XGBRegressor(**(params or {})))


#############################################
################## pipeline #################
#############################################
def init_pipeline() -> Pipeline:
    return Pipeline(steps=[])


def build_pipeline(pipe: Pipeline, model: tuple, preprocessing: list=None) -> Pipeline:
    if preprocessing:
        if isinstance(preprocessing, tuple):
            preprocessing = [preprocessing]
            
        preprocessor = ColumnTransformer(transformers=preprocessing, remainder="drop")
        pipe.steps.append(("preprocessing", preprocessor))
        
    pipe.steps.append(model)
    
    return pipe


def fit_pipeline(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe.fit(X_train, y_train)
    return pipe