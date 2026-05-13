from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import numpy as np
import pandas as pd

#############################################
################ predictions ################
#############################################
METRIC_REGISTRY = {
    "accuracy": lambda y, y_pred: accuracy_score(y, y_pred),
    "precision": lambda y, y_pred: precision_score(y, y_pred, average="binary"),
    "recall": lambda y, y_pred: recall_score(y, y_pred, average="binary"),
    "f1": lambda y, y_pred: f1_score(y, y_pred, average="binary")
}

PROBA_METRIC_REGISTRY = {
    "auc_roc": lambda y, y_proba: roc_auc_score(y, y_proba),
}


def predict(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, dataset: str, metrics: list) -> dict:
    y_pred = pipe.predict(X)
    results = {}
    for metric in metrics:
        results[f"{metric}_{dataset}"] = METRIC_REGISTRY[metric](y, y_pred)
    return results


def predict_proba(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, dataset: str, metrics: list) -> dict:
    y_proba = pipe.predict_proba(X)[:,1]
    results = {}
    for metric in metrics:
        results[f"{metric}_{dataset}"] = PROBA_METRIC_REGISTRY[metric](y, y_proba)
    return results


#############################################
############## postprocessing ###############
#############################################
def threshold_cutoff(probabilities, cutoff: float=0.5) -> np.ndarray:
    return (np.ndarray(probabilities) >= cutoff).astype(int)


#############################################
############## print results ################
#############################################
def print_results(metrics: list) -> dict:
    results = {}
    for metric_result in metrics:
        results.update(metric_result)
    print('\n        _____ Printed Metrics _____\n')
    for key, value in results.items():
        print(f"        |   {key}: {value:.4f}")
    
    print('        ___________________________\n')
    return results