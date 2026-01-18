# scripts/predict.py

import pandas as pd
from app.inference.Predictor import AttritionPredictor


def main():
    # ==========================================
    # 1. تحديد مسار النموذج من MLflow
    # ==========================================
    # استخدم RUN_ID الحقيقي من MLflow UI
    model_uri = "runs:/455c2236d4c74c2d816332c38da40615/model"

    # ==========================================
    # 2. تحميل بيانات جديدة (Inference only)
    # ==========================================
    input_path = "data/inference/new_employees.csv"
    df_new = pd.read_csv(input_path)

    # ==========================================
    # 3. إنشاء Predictor
    # ==========================================
    predictor = AttritionPredictor(model_uri=model_uri)

    # ==========================================
    # 4. تنفيذ التنبؤ
    # ==========================================
    predictions = predictor.predict(df_new)

    # ==========================================
    # 5. حفظ النتائج
    # ==========================================
    output_path = "data/inference/predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
