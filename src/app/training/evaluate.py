# src/app/training/evaluate.py

from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np


def evaluate_model(
    model,
    X_test,
    y_test,
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    This function:
    - assumes the model is already trained
    - computes standard classification metrics
    - returns metrics as a dictionary

    It does NOT:
    - train the model
    - read data
    - print results
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label='Yes',zero_division=0),
        "recall": recall_score(y_test, y_pred,pos_label='Yes', zero_division=0),
        "f1": f1_score(y_test, y_pred,pos_label='Yes', zero_division=0),
    }

    return metrics


def compute_confusion_matrix(
    model,
    X_test,
    y_test,
) -> np.ndarray:
    """
    Compute confusion matrix for a trained model.
    """

    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
