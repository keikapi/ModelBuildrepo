"""Feature engineers the abalone dataset."""
import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def to_xgb_csv(df, target_col, path, header=False):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    out = pd.concat([y, X], axis=1)
    out.to_csv(path, index=False, header=header)

def feature_engineering(df):
    logger.info("Start feature engineering")

    # ---------- Target ----------
    logger.info("Processing target column Transported")
    df["Transported"] = df["Transported"].astype(int)

    # ---------- PassengerId -> GroupSize ----------
    logger.info("Creating GroupSize from PassengerId")
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupId")["PassengerId"].transform("count")
    df.drop(columns=["PassengerId", "GroupId"], inplace=True)

    # ---------- Name ----------
    logger.info("Dropping Name column")
    df.drop(columns=["Name"], inplace=True)

    # ---------- Age -> AgeGroup ----------
    logger.info("Converting Age to AgeGroup")
    df["Age"] = df["Age"].fillna(-1)

    def age_to_group(age):
        if age < 0:
            return "Unknown"
        elif age < 20:
            return "child"
        elif age < 60:
            return "adult"
        else:
            return "senior"

    df["AgeGroup"] = df["Age"].apply(age_to_group)
    df.drop(columns=["Age"], inplace=True)

    # ---------- ServiceTotal ----------
    logger.info("Creating ServiceTotal feature")
    service_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    df[service_cols] = df[service_cols].fillna(0)
    df["ServiceTotal"] = df[service_cols].sum(axis=1)
    df.drop(columns=service_cols, inplace=True)

    # ---------- Cabin ----------
    logger.info("Splitting Cabin into Deck / CabinNum / Side")
    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df.drop(columns=["Cabin"], inplace=True)

    # ---------- Boolean ----------
    logger.info("Processing boolean columns")
    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].fillna(False).astype(int)

    # ---------- Categorical ----------
    logger.info("Processing categorical columns")
    categorical_cols = [
        "HomePlanet",
        "Destination",
        "Deck",
        "Side",
        "AgeGroup",
    ]

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # ---------- Numerical ----------
    logger.info("Processing numerical columns")
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())

    logger.info("Feature engineering completed")
    return df


def main():
    input_dir = "/opt/ml/processing/input"
    files = os.listdir(input_dir)
    if len(files) == 0:
        raise RuntimeError("No input files found")
    input_file = os.path.join(input_dir, files[0])
    df = pd.read_csv(input_file)
    logger.info(f"Loaded input file: {input_file}")
    logger.info(f"Input shape: {df.shape}")

    logger.info("Applying feature engineering")
    df = feature_engineering(df)

    logger.info("Splitting dataset into train / validation / test")
    train, temp = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["Transported"]
    )
    validation, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["Transported"]
    )

    output_base = "/opt/ml/processing"
    train_path = os.path.join(output_base, "train", "train.csv")
    val_path = os.path.join(output_base, "validation", "validation.csv")
    test_path = os.path.join(output_base, "test", "test.csv")

    logger.info("Saving processed datasets")
    to_xgb_csv(df_train, "Transported", "/opt/ml/processing/train/train.csv")
    to_xgb_csv(df_val, "Transported", "/opt/ml/processing/validation/validation.csv")
    to_xgb_csv(df_test, "Transported", "/opt/ml/processing/test/test.csv")

    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    main()