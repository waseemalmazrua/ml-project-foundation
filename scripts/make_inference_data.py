import pandas as pd
import os

df = pd.read_csv("data/raw/employee_attrition_performance.csv")

df_sample = df.sample(n=10, random_state=42)
df_inference = df_sample.drop(columns=["attrition"])

os.makedirs("data/inference", exist_ok=True)
df_inference.to_csv("data/inference/new_employees.csv", index=False)

print("Inference data created successfully.")
