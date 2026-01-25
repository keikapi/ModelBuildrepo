import argparse
import json
import logging
import os

import pandas as pd
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-path",
        type=str,
        default="/opt/ml/processing/test",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/opt/ml/processing/evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("===== Evaluation step started =====")
    logger.info(f"Test path: {args.test_path}")

    # --------------------
    # Load test data
    # --------------------
    test_file = os.path.join(args.test_path, "test.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    df_test = pd.read_csv(test_file)
    logger.info(f"Test data shape: {df_test.shape}")

    if "Transported" not in df_test.columns:
        raise ValueError("Target column 'Transported' not found in test data")

    y_true = df_test["Transported"].astype(int)
    X_test = df_test.drop(columns=["Transported"])

    # --------------------
    # Load prediction result
    # --------------------
    # XGBoost built-in の推論結果を想定
    # （事前に predict.csv を作っている前提）
    pred_file = os.path.join(args.test_path, "predictions.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    y_prob = pd.read_csv(pred_file, header=None).iloc[:, 0]
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
                "value": float(accuracy)
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
