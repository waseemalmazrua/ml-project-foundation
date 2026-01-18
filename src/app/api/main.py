from fastapi import FastAPI
import pandas as pd

from app.api.schemas import AttritionRequest, AttritionResponse
from app.inference.Predictor import AttritionPredictor


app = FastAPI(
    title="Employee Attrition Prediction API",
    version="1.0.0",
)


# ======================================
# Load model ONCE at startup
# ======================================
MODEL_URI ="models:/employee_attrition_model@production"

predictor = AttritionPredictor(
    model_uri=MODEL_URI,
    threshold=0.3,
)
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=AttritionResponse)
def predict_attrition(request: AttritionRequest):
    # 1. Convert request to DataFrame
    df = pd.DataFrame([request.dict()])

    # 2. Run inference
    result = predictor.predict(df)

    # 3. Return first row
    return AttritionResponse(
        prediction=result.iloc[0]["prediction"],
        probability=float(result.iloc[0]["probability"]),
    )
