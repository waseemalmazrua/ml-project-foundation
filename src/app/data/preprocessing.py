# src/app/data/preprocessing.py

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline for tabular data.

    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder

    This function does NOT:
    - read data
    - split data
    - fit models

    It only defines transformations.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor
