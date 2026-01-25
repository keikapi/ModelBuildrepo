import argparse
import json
import logging
import os
import tarfile

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-path", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("===== Evaluation step started =====")

    # --------------------
    # Load model (XGBoost built-in)
    # --------------------
    model_tar = os.path.join(args.model_path, "model.tar.gz")
    if not os.path.exists(model_tar):
        raise FileNotFoundError(f"Model archive not found: {model_tar}")

    with tarfile.open(model_tar) as tar:
        tar.extractall(path=args.model_path)

    booster = xgb.Booster()
    booster.load_model(os.path.join(args.model_path, "xgboost-model"))
    logger.info("XGBoost model loaded successfully")

    # --------------------
    # Load test data
    # --------------------
    test_file = os.path.join(args.test_path, "test.csv")
    df_test = pd.read_csv(test_file)

    y_true = df_test["Transported"].astype(int)
    X_test = df_test.drop(columns=["Transported"])

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
    # Save evaluation result
    # --------------------
    os.makedirs(args.output_path, exist_ok=True)

    evaluation_result = {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy
            }
        }
    }

    output_file = os.path.join(args.output_path, "evaluation.json")
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)

    logger.info(f"Evaluation result saved to {output_file}")
    logger.info("===== Evaluation step completed =====")


if __name__ == "__main__":
    main()
