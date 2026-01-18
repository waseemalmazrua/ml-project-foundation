from pydantic import BaseModel
from typing import List


class AttritionRequest(BaseModel):
    age: int
    monthly_salary: float
    years_at_company: int
    distance_from_home_km: float
    num_projects_last_year: int
    training_hours_last_year: int
    num_promotions: int
    last_promotion_years_ago: int
    performance_score: float
    job_satisfaction: float
    environment_satisfaction: float
    work_life_balance: float
    gender: str
    education: str
    department: str
    overtime: str


class AttritionResponse(BaseModel):
    prediction: str
    probability: float
