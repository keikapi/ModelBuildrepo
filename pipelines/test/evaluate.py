import argparse
import json
import logging
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/opt/ml/processing/model",
    )
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
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data path: {args.test_path}")

    # --------------------
    # Load model
    # --------------------
    model_file = os.path.join(args.model_path, "model.joblib")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = joblib.load(model_file)
    logger.info("Model loaded successfully")

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

    X_test = df_test.drop(columns=["Transported"])
    y_true = df_test["Transported"]

    # --------------------
    # Prediction
    # --------------------
    y_pred = model.predict(X_test)

    # LightGBM の predict は確率を返す場合がある
    if y_pred.dtype != int and y_pred.max() <= 1.0:
        y_pred = (y_pred >= 0.5).astype(int)

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
