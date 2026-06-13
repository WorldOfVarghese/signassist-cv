# Week 3 Day 1: Train first gesture classifier for SignAssist CV
# Goal:
# 1. Load landmark dataset from CSV
# 2. Split data into training and testing sets
# 3. Train a simple Random Forest classifier
# 4. Print accuracy and classification report
# 5. Save the trained model locally

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_PATH = "data/gesture_landmarks.csv"
MODEL_PATH = "models/gesture_classifier.pkl"


def load_dataset(dataset_path):
    """Load the gesture landmark dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run src/collect_landmark_data.py first."
        )

    df = pd.read_csv(dataset_path)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    return df


def check_dataset(df):
    """Print basic dataset information before training."""
    print("Dataset check")
    print("-------------")
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nSamples per label:")
    label_counts = df["label"].value_counts()
    print(label_counts)

    expected_columns = 1 + (21 * 3)

    print("\nColumn check:")
    print(f"Expected columns: {expected_columns}")
    print(f"Actual columns: {len(df.columns)}")

    if len(df.columns) != expected_columns:
        raise ValueError("Column count does not match expected landmark format.")

    if len(label_counts) < 2:
        raise ValueError("Need at least 2 gesture classes to train a classifier.")

    if label_counts.min() < 2:
        raise ValueError(
            "Each label needs at least 2 samples for train/test splitting. "
            "Collect more data for the smallest label."
        )

    print("\nDataset looks usable for a first baseline model.")


def train_model(df):
    """Train a Random Forest classifier."""
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nTraining result")
    print("---------------")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, list(X.columns), sorted(y.unique())


def save_model(model, feature_columns, labels, model_path):
    """Save the model and metadata locally."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "labels": labels,
    }

    with open(model_path, "wb") as file:
        pickle.dump(model_package, file)

    print(f"\nModel saved locally to: {model_path}")


def main():
    print("SignAssist CV - First Classifier Training")
    print("=========================================")

    df = load_dataset(DATASET_PATH)
    check_dataset(df)

    model, feature_columns, labels = train_model(df)
    save_model(model, feature_columns, labels, MODEL_PATH)

    print("\nDone.")
    print("Important: This is only a first baseline model.")
    print("The accuracy is not a final real-world performance claim.")


if __name__ == "__main__":
    main()