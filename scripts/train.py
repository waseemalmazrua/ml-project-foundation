# scripts/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# ==== استيراد منطق المشروع ====
from app.data.preprocessing import build_preprocessor
from app.models.classifier import classifier_model
from app.training.train import train_model
from app.training.evaluate import evaluate_model


def main():
    # ==================================================
    # 1. إعداد MLflow
    # ==================================================
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("employee_attrition_baseline")

    with mlflow.start_run(run_name="random_forest_baseline"):

        # ==================================================
        # 2. تحميل البيانات (IO مسموح هنا فقط)
        # ==================================================
        data_path = "data/raw/employee_attrition_performance.csv"
        df = pd.read_csv(data_path)

        # ==================================================
        # 3. تحديد X و y
        # ==================================================
        target_col = "attrition"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ==================================================
        # 4. Train / Test Split
        # ==================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # ==================================================
        # 5. تعريف الأعمدة (قرارات نهائية من EDA)
        # ==================================================
        num_cols = [
            "age",
            "monthly_salary",
            "years_at_company",
            "distance_from_home_km",
            "num_projects_last_year",
            "training_hours_last_year",
            "num_promotions",
            "last_promotion_years_ago",
            "performance_score",
            "job_satisfaction",
            "environment_satisfaction",
            "work_life_balance",
        ]

        cat_cols = [
            "gender",
            "education",
            "department",
            "overtime",
        ]

        # ==================================================
        # 6. بناء Preprocessor و Model
        # ==================================================
        preprocessor = build_preprocessor(
            numeric_features=num_cols,
            categorical_features=cat_cols,
        )

        model = classifier_model(random_state=42)

        # ==================================================
        # 7. التدريب
        # ==================================================
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model=model,
        )

        # ==================================================
        # 8. التقييم
        # ==================================================
        metrics = evaluate_model(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
        )

        # ==================================================
        # 9. تسجيل النتائج في MLflow
        # ==================================================
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "random_state": 42,
        })

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # ==================================================
        # 10. حفظ النموذج (Pipeline كامل)
        # ==================================================
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="employee_attrition_model",
        )


if __name__ == "__main__":
    main()
