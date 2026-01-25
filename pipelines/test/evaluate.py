import json
import logging
import os
import tarfile

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.info("===== Evaluation step started =====")

    # --------------------
    # Extract model
    # --------------------
    model_tar = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_tar) as tar:
        tar.extractall(path=".")

    # --------------------
    # Load XGBoost model
    # --------------------
    booster = xgb.Booster()
    booster.load_model("xgboost-model")
    logger.info("XGBoost model loaded")

    # --------------------
    # Load test data
    # --------------------
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path)

    y_true = df["Transported"].astype(int)
    X_test = df.drop(columns=["Transported"])

    dtest = xgb.DMatrix(X_test)

    # --------------------
    # Prediction
    # --------------------
    y_prob = booster.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    # --------------------
    # Accuracy
    # --------------------
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {accuracy}")

    # --------------------
    # Save evaluation
    # --------------------
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy
            }
        }
    }

    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info("===== Evaluation step completed =====")