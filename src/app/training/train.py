# src/app/training/train.py

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


def train_model(
    X_train,
    y_train,
    preprocessor: ColumnTransformer,
    model: BaseEstimator,
) -> Pipeline:
    """
    Trains a machine learning pipeline.

    This function:
    - builds a pipeline
    - fits it on training data
    - returns the trained pipeline

    It does NOT:
    - read files
    - split data
    - evaluate
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline
