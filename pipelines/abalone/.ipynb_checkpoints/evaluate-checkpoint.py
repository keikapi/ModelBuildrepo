import json
import logging
import os
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.info("===== Evaluation step started =====")

    # --------------------
    # Load model artifact
    # --------------------
    model_tar_path = "/opt/ml/processing/model/model.tar.gz"
    if not os.path.exists(model_tar_path):
        raise FileNotFoundError(f"Model archive not found: {model_tar_path}")

    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path=".")

    logger.info("Loading XGBoost model")
    model = pickle.load(open("xgboost-model", "rb"))

    # --------------------
    # Load test data
    # --------------------
    test_path = "/opt/ml/processing/test/test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # XGBoost built-in 前提：
    # 1列目 = label、残り = feature
    df = pd.read_csv(test_path, header=None)

    y_true = df.iloc[:, 0].astype(int).to_numpy()
    X = df.iloc[:, 1:].values

    dtest = xgboost.DMatrix(X)

    # --------------------
    # Prediction
    # --------------------
    logger.info("Running prediction")
    y_prob = model.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    # --------------------
    # Accuracy
    # --------------------
    accuracy = accuracy_score(y_true, y_pred)
    logger.info("Accuracy: %f", accuracy)

    # --------------------
    # Save evaluation result
    # --------------------
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy
            }
        }
    }

    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info("Evaluation report written to %s", evaluation_path)
    logger.info("===== Evaluation step completed =====")
