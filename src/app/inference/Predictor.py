from typing import Optional
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn


class AttritionPredictor:
    """
    Predictor class for employee attrition inference.

    Responsibilities:
    - Load trained ML pipeline from MLflow
    - Run predictions on new data
    - Apply explicit decision threshold
    """

    def __init__(
        self,
        model_uri: str,
        threshold: float = 0.3,
    ):
        """
        Initialize predictor by loading model from MLflow.

        Parameters
        ----------
        model_uri : str
            MLflow model URI (e.g. runs:/<RUN_ID>/model
            or models:/employee_attrition/Production)

        threshold : float
            Decision threshold for positive class (Attrition = Yes)
        """
        self.model_uri = model_uri
        self.threshold = threshold
        self.pipeline = mlflow.sklearn.load_model(model_uri)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on new data.

        Parameters
        ----------
        data : pd.DataFrame
            New observations with the same feature schema
            used during training.

        Returns
        -------
        pd.DataFrame
            Predictions with class labels and probabilities.
        """

        # ==================================================
        # 1. Predict probabilities (required for thresholding)
        # ==================================================
        if not hasattr(self.pipeline, "predict_proba"):
            raise RuntimeError(
                "Loaded model does not support probability predictions."
            )

        probabilities = self.pipeline.predict_proba(data)[:, 1]

        # ==================================================
        # 2. Apply decision threshold explicitly
        # ==================================================
        predictions = np.where(
            probabilities >= self.threshold,
            "Yes",
            "No",
        )

        # ==================================================
        # 3. Build output dataframe
        # ==================================================
        result = pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities,
        })

        return result
